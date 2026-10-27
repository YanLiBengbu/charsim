"""How often does a character spoken to move next? For every speech or reported speech addressed to
other characters of the scene, the share of cases in which the next move is made by one of the
addressees, per base model, scene and condition (the motivation of the turn-taking tool T9).
Output: printed table and results/summary/addressee_next.json."""

import argparse
import collections
import glob
import json
import os

CONDITIONS = ["prompted_tools-on", "world_tools-on", "soul_tools-on", "soul_tools-on_t4_t5_t6",
              "soul_tools-on_director", "prompted_tools-on_rich_coser"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim-root", default="results/sim")
    ap.add_argument("--out", default="results/summary/addressee_next.json")
    args = ap.parse_args()
    counts = collections.defaultdict(lambda: [0, 0])
    for path in glob.glob(os.path.join(args.sim_root, "*", "*", "*", "seed*.json")):
        run, scene, cond = os.path.relpath(path, args.sim_root).replace(os.sep, "/").split("/")[:3]
        if cond not in CONDITIONS or run.endswith("_pseudo"):
            continue
        with open(path, encoding="utf-8") as f:
            turns = json.load(f)["turns"]
        for a, b in zip(turns, turns[1:]):
            it = a["item"]
            to = [c for c in it.get("to") or [] if c != a["actor"]]
            if it["kind"] not in ("speech", "speak", "gist") or not to:
                continue
            c = counts[(run, scene, cond)]
            c[0] += b["actor"] in to
            c[1] += 1
    out = []
    for (run, scene, cond), (hit, n) in sorted(counts.items()):
        out.append({"run": run, "scene": scene, "condition": cond, "addressed": n, "next": hit, "share": hit / n})
        print(f"{run:6} {scene:13} {cond:30} {hit:4}/{n:4} = {hit / n:.2f}")
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
