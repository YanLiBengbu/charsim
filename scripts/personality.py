"""The personality edits against the predictions written before the simulations.

For every trait edit of config/scenes.json (card edit up or down, steering +1 or -1) the ending
classes of the edited sessions (majority of the judges, results/summary/sessions.csv) are
compared with the unedited soul sessions of the same base model and scene (soul_tools-on). A
prediction is read as follows:
  "X stays the most frequent class"     X is the most frequent class of the edited sessions
  "X becomes more frequent"             the share of X is higher than in the unedited sessions
  "C or B becomes more frequent than A" the share of B and C together is higher than that of A
  "no change of the class distribution" the class distribution does not differ (permutation test)
Every edited condition is also tested against the unedited one with a permutation test of the
total variation distance between the class distributions (sessions without a majority form a
class of their own), with the Holm correction over all edited conditions.
Output: results/summary/personality.json and a printed table."""

import argparse
import collections
import csv
import json
import os
import random
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from charsim.stats import holm  # noqa: E402

CLASSES = ("A", "B", "C", "D", "split")


def distribution(endings):
    c = collections.Counter(e or "split" for e in endings)
    return {k: c[k] / len(endings) for k in CLASSES}


def tv(a, b):
    da, db = distribution(a), distribution(b)
    return 0.5 * sum(abs(da[k] - db[k]) for k in CLASSES)


def tv_test(a, b, n_perm=10000, seed=12345):
    rng = random.Random(seed)
    obs = tv(a, b)
    pooled = list(a) + list(b)
    hits = 0
    for _ in range(n_perm):
        rng.shuffle(pooled)
        hits += tv(pooled[:len(a)], pooled[len(a):]) >= obs - 1e-12
    return obs, (hits + 1) / (n_perm + 1)


def holds(prediction, edited, base, p):
    """True or False for a prediction, None for "no prediction"."""
    de, db = distribution(edited), distribution(base)
    if prediction == "no prediction":
        return None
    if prediction == "no change of the class distribution":
        return p >= 0.05
    if m := re.fullmatch(r"(\w)( or (\w))? stays the most frequent class", prediction):
        top = max(de[k] for k in CLASSES)
        return any(de[x] == top for x in (m.group(1), m.group(3)) if x)
    if m := re.fullmatch(r"(\w) or (\w) becomes more frequent than (\w)", prediction):
        return de[m.group(1)] + de[m.group(2)] > de[m.group(3)]
    if m := re.fullmatch(r"(\w) becomes more frequent", prediction):
        return de[m.group(1)] > db[m.group(1)]
    raise ValueError(prediction)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sessions", default="results/summary/sessions.csv")
    ap.add_argument("--scenes", default="config/scenes.json")
    ap.add_argument("--out", default="results/summary/personality.json")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    with open(args.sessions, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    endings = collections.defaultdict(list)
    for r in rows:
        endings[(r["run"], r["scene"], r["condition"])].append(r["ending"])
    with open(args.scenes, encoding="utf-8") as f:
        scenes = json.load(f)["scenes"]

    out = []
    for sc in scenes:
        for edit in sc["trait_edits"]:
            trait = edit["trait"].replace(" ", "_")
            for run in sorted({k[0] for k in endings}):
                base = endings.get((run, sc["id"], "soul_tools-on"))
                for method, levels in (("card", ("up", "down")), ("steer", ("1", "-1"))):
                    for level in levels:
                        cond = f"soul_tools-on_{'edit' if method == 'card' else 'steer'}-{edit['character']}-{trait}-{level}"
                        edited = endings.get((run, sc["id"], cond))
                        if not base or not edited:
                            continue
                        direction = "up" if level in ("up", "1") else "down"
                        d, p = tv_test(edited, base)
                        out.append({"run": run, "scene": sc["id"], "character": edit["character"],
                                    "trait": edit["trait"], "control": bool(edit.get("control")),
                                    "method": method, "direction": direction, "n": len(edited),
                                    "prediction": edit[direction], "classes": distribution(edited),
                                    "base_classes": distribution(base), "tv": d, "p": p})
    for row, ph in zip(out, holm([r["p"] for r in out])):
        row["p_holm"] = ph
        row["holds"] = holds(row["prediction"], endings[(row["run"], row["scene"], "soul_tools-on_" + (
            "edit" if row["method"] == "card" else "steer") + f"-{row['character']}-{row['trait'].replace(' ', '_')}-"
            + {"card": {"up": "up", "down": "down"}, "steer": {"up": "1", "down": "-1"}}[row["method"]][row["direction"]])],
            endings[(row["run"], row["scene"], "soul_tools-on")], row["p"])
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)

    for r in out:
        cls = " ".join(f"{k}{r['classes'][k]:.2f}" for k in CLASSES if r["classes"][k])
        print(f"{r['run']:6} {r['scene']:13} {r['character']:10} {r['trait']:22} {r['method']:5} {r['direction']:4} "
              f"tv={r['tv']:.2f} p={r['p']:.3f} holm={r['p_holm']:.3f} holds={r['holds']}  [{cls}]  "
              f"pred: {r['prediction']}")
    judged = [r for r in out if r["holds"] is not None and not r["control"]]
    print(f"\nedited conditions {len(out)}, Holm < 0.05: {sum(r['p_holm'] < 0.05 for r in out)}, "
          f"raw p < 0.05: {sum(r['p'] < 0.05 for r in out)}")
    print(f"predictions of the edited traits that hold: {sum(r['holds'] for r in judged)} of {len(judged)}")
    for m in ("card", "steer"):
        sub = [r for r in judged if r["method"] == m]
        print(f"  {m}: {sum(r['holds'] for r in sub)} of {len(sub)}")
    ctrl = [r for r in out if r["control"]]
    print(f"control trait (no change): holds {sum(bool(r['holds']) for r in ctrl)} of {len(ctrl)}")


if __name__ == "__main__":
    main()
