"""LLM annotation of the novel -> data/derived/llm/<task>__<model>.jsonl (resumable).

Tasks: attribution (speakers and addressees of the quotation groups), beats (actions,
reported speech, silences and thoughts), events (the T2 event store) and relations (the T1
relation graph). Every call is seeded, runs at temperature 0 and is cached."""

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from charsim.cast import Cast  # noqa: E402
from charsim.extract import attribution_jobs, item_jobs, parse_attribution, parse_items  # noqa: E402
from charsim.jsonfix import parse_json, salvage_objects  # noqa: E402
from charsim.llm import OllamaClient  # noqa: E402

MAX_TOKENS = {"attribution": 1200, "beats": 2000, "events": 2000, "relations": 1500}


def chapter_list(spec):
    out = []
    for part in spec.split(","):
        lo, _, hi = part.partition("-")
        out += list(range(int(lo), int(hi or lo) + 1))
    return out


def safe(model):
    return model.replace(":", "_").replace("/", "_")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=["attribution", "beats", "events", "relations"])
    ap.add_argument("--model", required=True)
    ap.add_argument("--base-url", default=os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434"))
    ap.add_argument("--chapters", default="1-34")
    ap.add_argument("--text", default="data/text/chapters.json")
    ap.add_argument("--quotes", default="data/derived/quotes.jsonl")
    ap.add_argument("--cast", default="config/cast.json")
    ap.add_argument("--out-dir", default="data/derived/llm")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--num-ctx", type=int, default=8192)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    with open(args.text, encoding="utf-8") as f:
        chapters = {c["chapter"]: c for c in json.load(f)["chapters"]}
    cast = Cast(args.cast)
    rows = {}
    if args.task == "attribution":
        with open(args.quotes, encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                rows.setdefault(r["chapter"], []).append(r)

    jobs = []
    for c in chapter_list(args.chapters):
        if args.task == "attribution":
            jobs += list(attribution_jobs(chapters[c], rows.get(c, []), cast))
        else:
            jobs += list(item_jobs(chapters[c], args.task, cast))
    n_words = sum(len(j["prompt"].split()) for j in jobs)
    print(f"{args.task}: {len(jobs)} windows, about {int(n_words * 1.35)} prompt tokens")
    if args.dry_run:
        print("=" * 30, "first prompt", "=" * 30)
        print(jobs[0]["prompt"])
        return

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, f"{args.task}__{safe(args.model)}.jsonl")
    done = set()
    if os.path.exists(out_path):
        with open(out_path, encoding="utf-8") as f:
            done = {json.loads(line)["window"] for line in f}
    todo = [j for j in jobs if j["window"] not in done]
    print(f"{len(done)} windows already done, {len(todo)} to run -> {out_path}")
    llm = OllamaClient(args.model, base_url=args.base_url, temperature=0.0, max_tokens=MAX_TOKENS[args.task],
                       num_ctx=args.num_ctx, cache_path=os.path.join(args.out_dir, f"cache__{safe(args.model)}.jsonl"))

    def run(job):
        rec = {"window": job["window"], "chapter": job["chapter"], "pids": job["pids"], "task": args.task,
               "model": args.model, "repaired": False, "error": None, "json_mode": True}
        # JSON-constrained decoding can loop on one token (Ollama aborts with a 500); the
        # window is then decoded without the constraint and the JSON is repaired when parsed.
        try:
            text = llm.generate(job["prompt"], seed=0, temperature=0.0, json_mode=True)
        except RuntimeError:
            rec["json_mode"] = False
            try:
                text = llm.generate(job["prompt"], seed=0, temperature=0.0, json_mode=False)
            except RuntimeError as e:
                text, rec["error"] = "", str(e)[:200]
        try:
            obj, rec["repaired"] = parse_json(text)
        except ValueError as e:
            obj, rec["error"] = None, rec["error"] or str(e)[:200]
            items = salvage_objects(text) if args.task != "attribution" else []
            if items:
                obj, rec["repaired"], rec["error"] = {args.task: items}, "salvaged", None
        if args.task == "attribution":
            rec["labels"] = parse_attribution(obj, job)
        else:
            rec["items"], ok = parse_items(obj, job, args.task)
            if not ok and not rec["error"]:
                rec["error"] = "no item list"
        return rec

    t0 = time.time()
    with open(out_path, "a", encoding="utf-8") as f, ThreadPoolExecutor(args.workers) as pool:
        for i, rec in enumerate(pool.map(run, todo)):
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            if (i + 1) % 10 == 0 or i + 1 == len(todo):
                print(f"  {i + 1}/{len(todo)} windows, {time.time() - t0:.0f} s")

    with open(out_path, encoding="utf-8") as f:
        recs = [json.loads(line) for line in f]
    errors = sum(1 for r in recs if r["error"])
    if args.task == "attribution":
        labels = [v for r in recs for v in r["labels"].values()]
        print(f"labels={len(labels)} valid={sum(v['ok'] for v in labels)} windows with parse errors={errors}")
    else:
        items = [it for r in recs for it in r["items"]]
        print(f"items={len(items)} grounded={sum(it['grounded'] for it in items)} "
              f"valid={sum(it['valid'] for it in items)} windows with errors={errors}")


if __name__ == "__main__":
    main()
