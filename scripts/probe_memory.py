"""Memorization probe: how much do the base models know about the two test scenes?

Variants per scene and model (temperature 0, seed 0):
  mc_title     the scene set-up with the title and the author, multiple choice over the
               four pre-registered ending classes in four rotations of the options
  mc_plain     the same set-up with pseudonyms and without title and author
  ctx_title    the text before the decision point with real names, multiple choice
  ctx_plain    the same text with pseudonyms, "a novel" as source
  verbatim     continue the last 60 words before the scene's final 40 words; the score is
               the share of the 40 original words recovered in order (LCS recall)
  who          one-sentence identification of the main characters by name
Output: data/derived/probe/<model>.jsonl and a summary table."""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from charsim import pseudonyms  # noqa: E402
from charsim.grounding import tokens  # noqa: E402
from charsim.jsonfix import parse_json  # noqa: E402
from charsim.llm import OllamaClient  # noqa: E402
from charsim.prompts import PROBE_CONTEXT_MC, PROBE_MC, PROBE_VERBATIM, PROBE_WHO  # noqa: E402

LETTERS = "ABCD"
CONTEXT_PARAS = 3
VERBATIM_PREFIX, VERBATIM_TARGET = 60, 40
WHO = {"O-lan": ["wife"], "Wang Lung": ["farmer", "peasant", "husband", "landowner"],
       "Lotus": ["concubine", "courtesan", "mistress"], "Pear Blossom": ["slave", "servant", "maid", "concubine"],
       "Cuckoo": ["servant", "slave", "maid"]}


def scene_text(chapters, scene):
    paras = next(c for c in chapters if c["chapter"] == scene["chapter"])["paragraphs"]
    i0 = next(i for i, p in enumerate(paras) if scene["start_anchor"] in p["text"])
    i1 = next(i for i, p in enumerate(paras) if scene["end_anchor"] in p["text"])
    return paras, i0, i1


def context_passage(paras, i0, i1, cut_anchor):
    ic = next(i for i in range(i0, i1 + 1) if cut_anchor in paras[i]["text"])
    parts = [p["text"] for p in paras[max(0, i0 - CONTEXT_PARAS):ic]]
    t = paras[ic]["text"]
    parts.append(t[:t.index(cut_anchor) + len(cut_anchor)])
    return "\n\n".join(parts)


def verbatim_split(paras, i0, i1, scene):
    words = " ".join(p["text"] for p in paras[i0:i1 + 1])
    end = words.index(scene["end_anchor"]) + len(scene["end_anchor"])
    w = words[:end].split()
    return " ".join(w[-(VERBATIM_PREFIX + VERBATIM_TARGET):-VERBATIM_TARGET]), " ".join(w[-VERBATIM_TARGET:])


def lcs_recall(pred, target):
    a = [t for t, _, _ in tokens(pred)]
    b = [t for t, _, _ in tokens(target)]
    if not b:
        return 0.0
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0]
        for j, y in enumerate(b):
            cur.append(prev[j] + 1 if x == y else max(prev[j + 1], cur[j]))
        prev = cur
    return prev[-1] / len(b)


def options(classes, rotation):
    keys = list(classes)
    order = keys[rotation:] + keys[:rotation]
    return order, "\n".join(f"{LETTERS[i]}. {classes[k]}" for i, k in enumerate(order))


def ask_mc(llm, prompt):
    text = llm.generate(prompt, seed=0, temperature=0.0, json_mode=True, max_tokens=50)
    try:
        obj, _ = parse_json(text)
        ans = str(obj.get("answer", "")).strip().upper()[:1]
    except (ValueError, AttributeError):
        ans = ""
    return ans if ans in LETTERS else None, text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--base-url", default=os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434"))
    ap.add_argument("--text", default="data/text/chapters.json")
    ap.add_argument("--scenes", default="config/scenes.json")
    ap.add_argument("--pseudonyms", default="config/pseudonyms.json")
    ap.add_argument("--out-dir", default="data/derived/probe")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    with open(args.text, encoding="utf-8") as f:
        chapters = json.load(f)["chapters"]
    with open(args.scenes, encoding="utf-8") as f:
        scenes = json.load(f)["scenes"]
    table = pseudonyms.load(args.pseudonyms)

    prompts = []
    for sc in scenes:
        sc = dict(sc, probe_setup=sc["probe_setup"][0].lower() + sc["probe_setup"][1:])
        paras, i0, i1 = scene_text(chapters, sc)
        ctx = context_passage(paras, i0, i1, sc["probe_cut_anchor"])
        classes_plain = {k: pseudonyms.apply(v, table) for k, v in sc["ending_classes"].items()}
        for r in range(4):
            order, opts = options(sc["ending_classes"], r)
            order_p, opts_p = options(classes_plain, r)
            prompts += [
                (sc["id"], "mc_title", r, order, PROBE_MC.format(
                    intro="In Pearl S. Buck's novel The Good Earth, ", setup=sc["probe_setup"], options=opts)),
                (sc["id"], "mc_plain", r, order_p, PROBE_MC.format(
                    intro="In a novel, ", setup=pseudonyms.apply(sc["probe_setup"], table), options=opts_p)),
                (sc["id"], "ctx_title", r, order, PROBE_CONTEXT_MC.format(
                    source="Pearl S. Buck's novel The Good Earth", passage=ctx, kind="novel", options=opts)),
                (sc["id"], "ctx_plain", r, order_p, PROBE_CONTEXT_MC.format(
                    source="a novel", passage=pseudonyms.apply(ctx, table), kind="novel", options=opts_p))]
        prefix, target = verbatim_split(paras, i0, i1, sc)
        prompts.append((sc["id"], "verbatim", 0, target, PROBE_VERBATIM.format(prefix=prefix)))
    for name in WHO:
        prompts.append(("all", "who", 0, name, PROBE_WHO.format(name=name)))
    print(f"{len(prompts)} probe prompts per model")
    if args.dry_run:
        for p in prompts:
            if p[2] == 0 and p[1] in ("ctx_plain", "verbatim", "mc_title"):
                print("=" * 20, p[0], p[1], "=" * 20)
                print(p[4])
        return

    os.makedirs(args.out_dir, exist_ok=True)
    summary = []
    for model in args.models:
        llm = OllamaClient(model, base_url=args.base_url, temperature=0.0, max_tokens=200,
                           cache_path=os.path.join(args.out_dir, f"cache__{model.replace(':', '_')}.jsonl"))
        recs = []
        for scene_id, variant, rot, ref, prompt in prompts:
            rec = {"model": model, "scene": scene_id, "variant": variant, "rotation": rot}
            if variant.startswith(("mc", "ctx")):
                ans, raw = ask_mc(llm, prompt)
                rec.update({"answer": ans, "class": ref[LETTERS.index(ans)] if ans else None, "raw": raw})
            elif variant == "verbatim":
                raw = llm.generate(prompt, seed=0, temperature=0.0, json_mode=False, max_tokens=120)
                rec.update({"score": round(lcs_recall(raw, ref), 3), "raw": raw})
            else:
                raw = llm.generate(prompt, seed=0, temperature=0.0, json_mode=False, max_tokens=80)
                rec.update({"name": ref, "hit": any(k in raw.lower() for k in WHO[ref]), "raw": raw})
            recs.append(rec)
        with open(os.path.join(args.out_dir, f"{model.replace(':', '_')}.jsonl"), "w", encoding="utf-8") as f:
            for rec in recs:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        for sc in scenes:
            row = {"model": model, "scene": sc["id"]}
            for v in ("mc_title", "mc_plain", "ctx_title", "ctx_plain"):
                sub = [r for r in recs if r["scene"] == sc["id"] and r["variant"] == v]
                row[v] = f"{sum(r['class'] == sc['original_class'] for r in sub)}/{len(sub)}"
            row["verbatim"] = next(r["score"] for r in recs if r["scene"] == sc["id"] and r["variant"] == "verbatim")
            summary.append(row)
        who = [r for r in recs if r["variant"] == "who"]
        print(f"{model}: who {sum(r['hit'] for r in who)}/{len(who)}")
    print("original ending chosen (of 4 rotations) and verbatim LCS recall:")
    for row in summary:
        print(row)
    with open(os.path.join(args.out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
