"""chapters.json -> data/derived/quotes.jsonl: quotation groups with rule-based speakers.

With --gold, the rule-based speakers are compared with the manually attributed quotations
of the chapter 26 excerpt used in the previous study (Zhao and Shi, PeerJ CS)."""

import argparse
import collections
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from charsim import gold as gold_quotes  # noqa: E402
from charsim.attribution import attribute  # noqa: E402
from charsim.cast import Cast  # noqa: E402
from charsim.quotes import find_quotes, groups  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chapters", default="data/text/chapters.json")
    ap.add_argument("--cast", default="config/cast.json")
    ap.add_argument("--out", default="data/derived/quotes.jsonl")
    ap.add_argument("--gold", default=gold_quotes.GOLD)
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    with open(args.chapters, encoding="utf-8") as f:
        chapters = json.load(f)["chapters"]
    cast = Cast(args.cast)
    rows = []
    for ch in chapters:
        quotes = find_quotes(ch)
        labels = attribute(ch, quotes, cast)
        for g, segs in groups(quotes).items():
            rows.append({"group": g, "chapter": ch["chapter"], "pids": sorted({s["pid"] for s in segs}),
                         "segments": [{k: s[k] for k in ("qid", "pid", "start", "end", "open_end")} for s in segs],
                         "text": " ".join(s["text"] for s in segs),
                         "rule_speaker": labels[g]["speaker"], "rule_method": labels[g]["method"]})
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    methods = collections.Counter(r["rule_method"] or "none" for r in rows)
    speakers = collections.Counter(r["rule_speaker"] or "none" for r in rows)
    words = collections.Counter()
    for r in rows:
        words[r["rule_speaker"] or "none"] += len(r["text"].split())
    print(f"quotation groups={len(rows)} segments={sum(len(r['segments']) for r in rows)} "
          f"words={sum(words.values())}")
    print("methods:", dict(methods))
    print("speakers (groups, words):", [(k, v, words[k]) for k, v in speakers.most_common()])
    for c in (19, 34):
        sub = [r for r in rows if r["chapter"] == c]
        print(f"chapter {c}: groups={len(sub)}", collections.Counter(r["rule_speaker"] or "none" for r in sub))

    if args.gold and os.path.exists(args.gold):
        gold = gold_quotes.load(args.gold)
        n_match = n_lab = n_ok = 0
        for gq, gid, hit in gold_quotes.match(rows, gold):
            if not hit:
                print("  gold quotation not found:", gq["text"][:60])
                continue
            n_match += 1
            if hit["rule_speaker"]:
                n_lab += 1
                ok = hit["rule_speaker"] == gid
                n_ok += ok
                if not ok:
                    print(f"  mismatch: gold={gq['speaker']} rule={hit['rule_speaker']} ({hit['rule_method']}) "
                          f"| {gq['text'][:50]}")
        print(f"gold (ch. 26 excerpt): {len(gold)} quotations, {n_match} found, {n_lab} labelled by rules, "
              f"{n_ok} correct")


if __name__ == "__main__":
    main()
