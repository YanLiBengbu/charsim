"""Exploratory tests of the share of sessions with an ending (class A, B or C by the majority of the
judges): all character tools against the soul sessions without them and against the director,
per base model and scene. Two-sided Fisher exact tests with the Holm correction over the eight
comparisons. The share is descriptive in the planned analysis, so these tests are exploratory.
Output: printed table and results/summary/decided_tests.json."""

import argparse
import collections
import csv
import json
import os
import sys
from math import comb

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from charsim.stats import holm  # noqa: E402

ALL = "soul_tools-on_t4_t5_t6_t7_t8_t9"
AGAINST = ["soul_tools-on", "soul_tools-on_director"]


def fisher(a, n1, b, n2):
    """Two-sided Fisher exact test of a/n1 against b/n2 (sum of the tables no more likely)."""
    k, total = a + b, n1 + n2

    def p(x):
        return comb(n1, x) * comb(n2, k - x) / comb(total, k)

    p0 = p(a)
    return min(1.0, sum(p(x) for x in range(max(0, k - n2), min(k, n1) + 1) if p(x) <= p0 * (1 + 1e-9)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sessions", default="results/summary/sessions.csv")
    ap.add_argument("--out", default="results/summary/decided_tests.json")
    args = ap.parse_args()
    rows = collections.defaultdict(list)
    with open(args.sessions, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows[(r["run"], r["scene"], r["condition"])].append(r["ending"] in ("A", "B", "C"))
    tests = []
    for b in AGAINST:
        for run in ("qwen", "llama"):
            for scene in ("pearls", "sell_the_land"):
                x, y = rows[(run, scene, ALL)], rows[(run, scene, b)]
                tests.append({"run": run, "scene": scene, "a": ALL, "b": b, "share_a": sum(x) / len(x),
                              "share_b": sum(y) / len(y), "n_a": len(x), "n_b": len(y),
                              "p": fisher(sum(x), len(x), sum(y), len(y))})
    for t, ph in zip(tests, holm([t["p"] for t in tests])):
        t["p_holm"] = ph
        print(f"{t['run']:6} {t['scene']:13} all tools {t['share_a']:.2f} (n={t['n_a']}) vs {t['b']:24} "
              f"{t['share_b']:.2f} (n={t['n_b']})  p = {t['p']:.5f}  Holm {t['p_holm']:.5f}")
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(tests, f, indent=1)


if __name__ == "__main__":
    main()
