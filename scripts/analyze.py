"""Summary of the simulated sessions and of the judges.

Per session: the speaker shares, the intentions, the outcome of the state machine, the
mention of a character who enters the novel only after the chapter of the scene
(anachronism), and, when the judges have run, the majority ending class, the propositions and
the ratings, and the share of repeated moves (a move whose words were already used in the
session, by any character), whether a transfer of the pearls is undone or contradicted later and
whether a character leaves the place or time of the scene (charsim/state.py; all three
descriptive, added on 2026-10-26 after the main results). Per condition: the class
distribution with bootstrap intervals, the share of the original class, the proposition rates,
the ratings, and permutation tests of the planned contrasts with the Holm correction (the
contrasts of the three follow-ups in families of their own; the share of sessions decided, with
an ending class other than D, is descriptive). The judges are checked on the
controls first (accuracy of the ending class, Krippendorff's alpha). Output: results/summary/."""

import argparse
import collections
import csv
import glob
import json
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from charsim import pseudonyms  # noqa: E402
from charsim.cast import Cast  # noqa: E402
from charsim.state import away, outcome, undone  # noqa: E402
from charsim.stats import bootstrap_ci, holm, krippendorff_alpha, permutation_test  # noqa: E402

# planned contrasts within a run and scene: (condition a, condition b)
CONTRASTS = [("soul_tools-on", "prompted_tools-on"), ("soul_tools-on", "soul_tools-off"),
             ("prompted_tools-on", "prompted_tools-off"), ("soul_tools-on", "world_tools-on")]
# contrasts of the follow-up conditions (rich cards, director, CoSER-style acting), added on
# 2026-10-26 after the main results: a family of their own for the Holm correction
FOLLOWUP = [("soul_tools-on_rich", "soul_tools-on"), ("prompted_tools-on_rich", "prompted_tools-on"),
            ("soul_tools-on_director", "soul_tools-on"), ("soul_tools-on_rich_director", "soul_tools-on_rich"),
            ("prompted_tools-on_rich_coser", "prompted_tools-on_rich")]
# contrasts of the tools of the characters (plan T4, state T5, scene check T6; second follow-up,
# 2026-10-26): a third family
TOOLS = [("soul_tools-on_t4_t5_t6", "soul_tools-on"), ("soul_tools-on_t4", "soul_tools-on"),
         ("soul_tools-on_t6", "soul_tools-on"), ("soul_tools-on_t4_t5_t6", "soul_tools-on_director")]
# contrasts of the consistency check T7, the choice T8 and the turn-taking T9 (third follow-up,
# 2026-10-26; all with the T5 texts of config/state_texts2.json): a fourth family
ALL = "soul_tools-on_t4_t5_t6_t7_t8_t9"
TOOLS2 = [(ALL, "soul_tools-on_t4_t5_t6_t7_t8"), (ALL, "soul_tools-on_t4_t5_t6_t7_t9"),
          ("soul_tools-on_t4_t5_t6_t7_t8", "soul_tools-on_t4_t5_t6"),
          (ALL, "soul_tools-on"), (ALL, "soul_tools-on_director")]
RATINGS = ("character", "coherence", "plausibility")
GIVE_PROPOSITION = 3  # "O-lan gives the pearls to Wang Lung." in the propositions of the pearls scene


def first_chapters(cast, chapters_path):
    """First chapter in which the text mentions every character."""
    with open(chapters_path, encoding="utf-8") as f:
        chapters = json.load(f)["chapters"]
    first = {}
    for ch in chapters:
        for p in ch["paragraphs"]:
            for _, _, cid in cast.mentions(p["text"], ch["chapter"]):
                first.setdefault(cid, ch["chapter"])
    return first


def name_patterns(cast):
    """Proper names (capitalised aliases) of every character regardless of chapter ranges."""
    out = []
    for cid, c in cast.characters.items():
        for alias, _, _ in c["aliases"]:
            if alias[0].isupper() and alias not in ("Wang",):
                out.append((re.compile(r"(?<![\w-])" + re.escape(alias) + r"(?![\w-])"), cid))
    return out


def load_judges(path_glob):
    out = collections.defaultdict(dict)
    for path in glob.glob(path_glob):
        with open(path, encoding="utf-8") as f:
            for rec in map(json.loads, f):
                if rec.get("ok"):
                    out[rec["key"]][rec["judge"]] = rec
    return out


def repeated_share(turns):
    """Share of the spoken, summarised and acted moves whose words (lower case, letters only)
    were already used earlier in the session."""
    seen, n, rep = set(), 0, 0
    for t in turns:
        if t["item"]["kind"] not in ("speak", "gist", "act"):
            continue
        k = re.sub(r"[^a-z ]", "", t["item"]["text"].lower()).strip()
        if k:
            n, rep = n + 1, rep + (k in seen)
            seen.add(k)
    return rep / n if n else 0.0


def plurality(labels):
    c = collections.Counter(x for x in labels if x is not None).most_common()
    if not c or (len(c) > 1 and c[0][1] == c[1][1]):
        return None
    return c[0][0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim-root", default="results/sim")
    ap.add_argument("--judges", default="results/judge/*.jsonl")
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="results/summary")
    ap.add_argument("--old-transfer-rule", action="store_true",
                    help="count the pearls transfer with the rule before 2026-10-27 (without state.HIS_OWN)")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    cast = Cast("config/cast.json")
    with open("config/scenes.json", encoding="utf-8") as f:
        scenes = {s["id"]: s for s in json.load(f)["scenes"]}
    names, pats = pseudonyms.load("config/pseudonyms.json")
    back = {v: k for k, v in names.items()}
    unpseudo = (dict(sorted(back.items(), key=lambda kv: -len(kv[0]))),
                re.compile("|".join(r"(?<![\w-])" + re.escape(k) + r"(?![\w-])"
                                    for k in sorted(back, key=len, reverse=True))))
    first = first_chapters(cast, os.path.join(args.data, "text/chapters.json"))
    patterns = name_patterns(cast)
    judged = load_judges(args.judges)
    judges = sorted({j for v in judged.values() for j in v})
    os.makedirs(args.out, exist_ok=True)

    # the judges on the controls
    report = {"judges": judges, "controls": {}}
    units = []
    classes = {k: by for k, by in judged.items() if k.startswith("control/") and k.split("/")[2] in "ABCD"}
    for key, by in sorted(classes.items()):
        report["controls"][key] = {j: by[j]["ending"] for j in judges if j in by}
        units.append([by[j]["ending"] if j in by else None for j in judges])
    report["control_accuracy"] = {j: sum(by.get(j, {}).get("ending") == k.split("/")[2] for k, by in classes.items())
                                  / max(1, len(classes)) for j in judges}
    # the controls of the ratings: the drop of coherence when the lines of the original scene are
    # shuffled and of character when its first two characters swap their lines
    report["rating_controls"] = {}
    for scene in scenes:
        orig = judged.get(f"control/{scene}/A", {})
        for kind, rating in (("shuffled", "coherence"), ("swapped", "character")):
            broken = judged.get(f"control/{scene}/{kind}", {})
            report["rating_controls"][f"{scene}/{kind}/{rating}"] = {
                j: [orig[j][rating], broken[j][rating]] for j in judges if j in orig and j in broken}
    # the ratings of a judge are used when it rates every broken version lower than the original
    # on the scale that the break targets (rule fixed on 2026-10-24, before the sessions were
    # judged with the second prompt; on its controls phi4 gave 5 and mistral-nemo 4 throughout)
    rating_judges = [j for j in judges if report["rating_controls"] and all(
        j in v and v[j][1] < v[j][0] for v in report["rating_controls"].values())]
    report["rating_judges"] = rating_judges

    # the sessions
    rows = []
    for path in sorted(glob.glob(os.path.join(args.sim_root, "*", "*", "*", "seed*.json"))):
        rel = os.path.relpath(path, args.sim_root).replace(os.sep, "/")
        run, scene, cond, _ = rel.split("/")
        with open(path, encoding="utf-8") as f:
            res = json.load(f)
        sc = scenes[scene]
        text = " ".join(t["item"]["text"] for t in res["turns"])
        if run.endswith("_pseudo"):
            text = pseudonyms.apply(text, unpseudo)
        late = sorted({cid for pat, cid in patterns if pat.search(text) and first.get(cid, 0) > sc["chapter"]})
        actors = collections.Counter(t["actor"] for t in res["turns"])
        intends = collections.Counter(t["intend"] for t in res["turns"])
        # the object outcome is recomputed with the rules of charsim/state.py (the state was not
        # shown to the characters, so recomputing it does not change the sessions; the state tool
        # T5 shows the state of the same rules; since 2026-10-27 the measure leaves out Wang Lung's
        # sentences about pearls he already has, state.HIS_OWN)
        strict = not args.old_transfer_rule
        state = outcome(scene, res["turns"], strict)
        row = {"run": run, "scene": scene, "condition": cond, "seed": res["seed"], "turns": len(res["turns"]),
               "anachronism": int(bool(late)), "late_characters": " ".join(late),
               "seconds": round(res.get("seconds", 0)), "repeated": repeated_share(res["turns"]),
               "away": int(away(res["turns"]))}
        if state["changed"] is not None:
            row |= {"state_changed": int(state["changed"]), "state_turn": state["turn"]}
        if (u := undone(scene, res["turns"], strict)) is not None:
            row["undone"] = int(u)  # the transfer is undone or contradicted later (descriptive)
        row |= {f"share_{c}": actors[c] / len(res["turns"]) for c in sc["cast"]}
        row |= {f"intend_{k}": intends[k] / len(res["turns"]) for k in ("speak", "act", "silent")}
        by = judged.get(rel, {})
        if by:
            units.append([by[j]["ending"] if j in by else None for j in judges])
            row["ending"] = plurality([by[j]["ending"] for j in judges if j in by])
            row["original_class"] = int(row["ending"] == sc["original_class"])
            for i in range(len(sc["propositions"])):
                row[f"p{i + 1}"] = plurality([by[j]["propositions"][i] for j in judges if j in by])
            for r in RATINGS:
                vals = [by[j][r] for j in rating_judges if j in by]
                if vals:
                    row[r] = sum(vals) / len(vals)
        rows.append(row)
    report["alpha_ending"] = krippendorff_alpha(units, "nominal")
    report["alpha_ratings"] = {r: krippendorff_alpha(
        [[judged[k][j][r] if j in judged[k] else None for j in judges] for k in judged], "interval") for r in RATINGS}

    fields = sorted({k for r in rows for k in r}, key=lambda k: (k not in ("run", "scene", "condition", "seed"), k))
    with open(os.path.join(args.out, "sessions.csv"), "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    # per condition
    groups = collections.defaultdict(list)
    for r in rows:
        groups[(r["run"], r["scene"], r["condition"])].append(r)
    conds = {}
    for (run, scene, cond), rs in sorted(groups.items()):
        sc = scenes[scene]
        c = {"n": len(rs), "anachronism": sum(r["anachronism"] for r in rs) / len(rs),
             "shares": {x: sum(r[f"share_{x}"] for r in rs) / len(rs) for x in sc["cast"]},
             "seconds": sum(r["seconds"] for r in rs) / len(rs), "moves": sum(r["turns"] for r in rs) / len(rs),
             "repeated": sum(r["repeated"] for r in rs) / len(rs), "away": sum(r["away"] for r in rs) / len(rs)}
        if all("state_changed" in r for r in rs):
            c["state_changed"] = sum(r["state_changed"] for r in rs) / len(rs)
            if moved := [r["undone"] for r in rs if "undone" in r]:
                c["undone"] = sum(moved) / len(moved)
            given = [r for r in rs if "state_changed" in r and r.get(f"p{GIVE_PROPOSITION + 1}") is not None]
            if given:  # agreement of the rule with the judges on "O-lan gives the pearls to Wang Lung"
                c["rule_judge_agreement"] = sum(r["state_changed"] == int(r[f"p{GIVE_PROPOSITION + 1}"])
                                                for r in given) / len(given)
        judged_rs = [r for r in rs if "ending" in r]
        if judged_rs:
            c["classes"] = dict(collections.Counter(r["ending"] or "split" for r in judged_rs))
            # descriptive (2026-10-26): the judges agree on an ending other than "no decision"
            c["decided"] = sum(r["ending"] in ("A", "B", "C") for r in judged_rs) / len(judged_rs)
            orig = [r["original_class"] for r in judged_rs]
            c["original_class"] = sum(orig) / len(orig)
            c["original_class_ci"] = bootstrap_ci(orig)
            c["propositions"] = [sum(bool(r.get(f"p{i + 1}")) for r in judged_rs) / len(judged_rs)
                                 for i in range(len(sc["propositions"]))]
            c["ratings"] = {x: sum(r[x] for r in judged_rs if x in r) / n for x in RATINGS
                            if (n := sum(x in r for r in judged_rs))}
        conds[f"{run}/{scene}/{cond}"] = c

    def family(contrasts):
        tests = []
        for (run, scene) in sorted({(k[0], k[1]) for k in groups}):
            for a, b in contrasts:
                ra, rb = groups.get((run, scene, a)), groups.get((run, scene, b))
                if not ra or not rb:
                    continue
                for metric in ("original_class", "state_changed", "anachronism", *RATINGS):
                    xa = [r[metric] for r in ra if metric in r]
                    xb = [r[metric] for r in rb if metric in r]
                    if xa and xb:
                        tests.append({"run": run, "scene": scene, "a": a, "b": b, "metric": metric,
                                      "mean_a": sum(xa) / len(xa), "mean_b": sum(xb) / len(xb),
                                      "p": permutation_test(xa, xb)})
        for t, p in zip(tests, holm([t["p"] for t in tests])):
            t["p_holm"] = p
        return tests

    tests, followup, tool_tests = family(CONTRASTS), family(FOLLOWUP), family(TOOLS)
    tool2_tests = family(TOOLS2)
    report |= {"conditions": conds, "tests": tests, "followup_tests": followup, "tool_tests": tool_tests,
               "tool2_tests": tool2_tests, "first_chapter": first}
    with open(os.path.join(args.out, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=1)

    print("judges:", judges, "control accuracy:", report["control_accuracy"], "alpha:", report["alpha_ending"])
    print("rating controls (original, broken):", report["rating_controls"], "rating judges:", rating_judges)
    for k, c in conds.items():
        extra = f" A={c['original_class']:.2f} decided={c['decided']:.2f} classes={c['classes']} ratings={ {x: round(v, 2) for x, v in c['ratings'].items()} }" \
            if "classes" in c else ""
        changed = f"{c['state_changed']:.2f}" if "state_changed" in c else "  - "
        changed += f" undone={c['undone']:.2f}" if "undone" in c else ""
        print(f"{k:45s} n={c['n']:2d} moves={c['moves']:.1f} repeated={c['repeated']:.2f} away={c['away']:.2f} "
              f"changed={changed} "
              f"anachron={c['anachronism']:.2f} "
              f"shares={ {x: round(v, 2) for x, v in c['shares'].items()} }{extra}")
    for name, ts in (("planned", tests), ("follow-up", followup), ("tools", tool_tests), ("tools2", tool2_tests)):
        for t in ts:
            if t["metric"] in ("original_class", "state_changed"):
                print(f"{name} {t['run']}/{t['scene']} {t['a']} vs {t['b']} {t['metric']}: "
                      f"{t['mean_a']:.2f} vs {t['mean_b']:.2f}, p = {t['p']:.4f}, Holm {t['p_holm']:.4f}")


if __name__ == "__main__":
    main()
