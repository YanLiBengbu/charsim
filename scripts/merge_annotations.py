"""Merge the rule-based and LLM annotations and build the narrative tools.

Speakers: two LLM extractors and the rules vote; an optional third LLM decides the
remaining disagreements. Beats, events and relations of the two extractors are matched by
their grounded evidence spans; the agreement between the extractors is reported as F1.

Outputs: data/derived/quotes_final.jsonl, data/derived/beats_final.jsonl,
data/tools/events.jsonl (T2) and data/tools/relations.jsonl (T1), and an agreement report."""

import argparse
import collections
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from charsim import gold as gold_quotes  # noqa: E402

MIN_OVERLAP = 0.5


def safe(model):
    return model.replace(":", "_").replace("/", "_")


def load_llm(path):
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def key(label):
    return json.dumps(label, sort_keys=True) if label is not None else None


def kappa(pairs):
    pairs = [(a, b) for a, b in pairs if a is not None and b is not None]
    if not pairs:
        return None, 0
    n = len(pairs)
    po = sum(a == b for a, b in pairs) / n
    ca, cb = collections.Counter(a for a, _ in pairs), collections.Counter(b for _, b in pairs)
    pe = sum(ca[k] * cb.get(k, 0) for k in ca) / (n * n)
    return (round((po - pe) / (1 - pe), 3) if pe < 1 else 1.0), n


def merge_speakers(rows, labels, adjudicator):
    out, tiers = [], collections.Counter()
    m1, m2 = list(labels)
    for r in rows:
        g = r["group"]
        a, b = labels[m1].get(g, {}).get("speaker"), labels[m2].get(g, {}).get("speaker")
        rule = r["rule_speaker"]
        if a is not None and key(a) == key(b):
            final, tier = a, "llm_agree"
        elif rule is not None and key(rule) in (key(a), key(b)):
            final, tier = rule, "majority"
        elif adjudicator and adjudicator.get(g, {}).get("speaker") is not None and \
                key(adjudicator[g]["speaker"]) in {key(a), key(b), key(rule)}:
            final, tier = adjudicator[g]["speaker"], "adjudicated"
        else:
            final, tier = None, "unresolved"
        to = labels[m1].get(g, {}).get("to") or labels[m2].get(g, {}).get("to") or []
        tiers[tier] += 1
        out.append({**r, "speakers": {m1: a, m2: b, "rule": rule}, "speaker": final, "to": to, "tier": tier})
    return out, tiers


def _overlap(x, y):
    if not x["span"] or not y["span"] or x["span"]["pid"] != y["span"]["pid"]:
        return 0.0
    s, e = max(x["span"]["start"], y["span"]["start"]), min(x["span"]["end"], y["span"]["end"])
    shorter = min(x["span"]["end"] - x["span"]["start"], y["span"]["end"] - y["span"]["start"]) or 1
    return max(0, e - s) / shorter


def _same(task, x, y):
    if task == "beats":
        return x["character"] == y["character"] and x["type"] == y["type"]
    if task == "events":
        px, py = set(x["participants"]), set(y["participants"])
        return len(px & py) / len(px | py) >= 0.5 if px | py else False
    sign = lambda v: (v > 0) - (v < 0)  # noqa: E731
    return (x["source"], x["target"]) == (y["source"], y["target"]) and \
        (x["relation"] == y["relation"] or sign(x["polarity"]) == sign(y["polarity"]))


def match_items(task, recs1, recs2):
    by_win2 = {r["window"]: r for r in recs2}
    merged, n1, n2, n_match = [], 0, 0, 0
    for r1 in recs1:
        a = [it for it in r1["items"] if it["grounded"] and it["valid"]]
        b = [it for it in by_win2.get(r1["window"], {"items": []})["items"] if it["grounded"] and it["valid"]]
        n1, n2 = n1 + len(a), n2 + len(b)
        used = set()
        for x in a:
            best, bo = None, MIN_OVERLAP
            for j, y in enumerate(b):
                if j in used or not _same(task, x, y):
                    continue
                o = _overlap(x, y)
                if o >= bo:
                    best, bo = j, o
            if best is not None:
                used.add(best)
                n_match += 1
            merged.append({"window": r1["window"], "chapter": r1["chapter"], **x, "support": 2 if best is not None else 1,
                           "partner": b[best] if best is not None else None})
        for j, y in enumerate(b):
            if j not in used:
                merged.append({"window": r1["window"], "chapter": r1["chapter"], **y, "support": 1, "partner": None})
    f1 = round(2 * n_match / (n1 + n2), 3) if n1 + n2 else None
    return merged, {"items_model1": n1, "items_model2": n2, "matched": n_match, "f1": f1}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs=2, required=True, help="the two extractor models")
    ap.add_argument("--adjudicator", default=None)
    ap.add_argument("--llm-dir", default="data/derived/llm")
    ap.add_argument("--quotes", default="data/derived/quotes.jsonl")
    ap.add_argument("--out-dir", default="data/derived")
    ap.add_argument("--tools-dir", default="data/tools")
    ap.add_argument("--gold", default=gold_quotes.GOLD)
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    report = {"models": args.models, "adjudicator": args.adjudicator}

    with open(args.quotes, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f]
    labels = {}
    for m in args.models:
        recs = load_llm(os.path.join(args.llm_dir, f"attribution__{safe(m)}.jsonl"))
        if recs is None:
            sys.exit(f"missing attribution output of {m}")
        labels[m] = {g: v for r in recs for g, v in r["labels"].items()}
    adj = None
    if args.adjudicator:
        recs = load_llm(os.path.join(args.llm_dir, f"attribution__{safe(args.adjudicator)}.jsonl"))
        adj = {g: v for r in recs for g, v in r["labels"].items()} if recs else None
    final, tiers = merge_speakers(rows, labels, adj)
    m1, m2 = args.models
    report["speakers"] = {
        "groups": len(final), "tiers": dict(tiers),
        "kappa_llm1_llm2": kappa([(key(r["speakers"][m1]), key(r["speakers"][m2])) for r in final]),
        "kappa_llm1_rule": kappa([(key(r["speakers"][m1]), key(r["speakers"]["rule"])) for r in final]),
        "kappa_llm2_rule": kappa([(key(r["speakers"][m2]), key(r["speakers"]["rule"])) for r in final]),
        "by_speaker": collections.Counter(key(r["speaker"]) for r in final).most_common()}
    if args.gold and os.path.exists(args.gold):
        hits = [(gid, r) for _, gid, r in gold_quotes.match(final, gold_quotes.load(args.gold)) if r]
        sources = {m1: lambda r: r["speakers"][m1], m2: lambda r: r["speakers"][m2],
                   "rule": lambda r: r["speakers"]["rule"], "final": lambda r: r["speaker"]}
        if adj:
            sources[args.adjudicator] = lambda r: adj.get(r["group"], {}).get("speaker")
        report["speakers"]["gold_ch26"] = {
            name: {"labelled": sum(get(r) is not None for _, r in hits), "correct": sum(get(r) == gid for gid, r in hits)}
            for name, get in sources.items()} | {"quotations": len(hits)}
    with open(os.path.join(args.out_dir, "quotes_final.jsonl"), "w", encoding="utf-8") as f:
        for r in final:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    os.makedirs(args.tools_dir, exist_ok=True)
    for task in ("beats", "events", "relations"):
        recs = [load_llm(os.path.join(args.llm_dir, f"{task}__{safe(m)}.jsonl")) for m in args.models]
        if None in recs:
            print(f"{task}: outputs of both extractors are needed, skipped")
            continue
        merged, stats = match_items(task, recs[0], recs[1])
        report[task] = stats
        if task == "beats":
            path = os.path.join(args.out_dir, "beats_final.jsonl")
        else:
            path = os.path.join(args.tools_dir, f"{task}.jsonl")
        with open(path, "w", encoding="utf-8") as f:
            for i, it in enumerate(merged):
                partner = it.pop("partner")
                if task == "events" and partner:
                    it["witnesses"] = sorted(set(it["witnesses"]) | set(partner["witnesses"]))
                it["id"] = f"{task[0]}{i:05d}"
                f.write(json.dumps(it, ensure_ascii=False) + "\n")

    with open(os.path.join(args.out_dir, "agreement_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=1)
    print(json.dumps({k: v for k, v in report.items() if k != "speakers"} |
                     {"speakers": {k: v for k, v in report["speakers"].items() if k != "by_speaker"}},
                     ensure_ascii=False, indent=1))
    print("speakers:", report["speakers"]["by_speaker"][:12])


if __name__ == "__main__":
    main()
