"""Supplemental Information of the paper as a LaTeX file, drawn from the result files.

Tables S1 to S4 list every permutation test of the four families with raw and Holm-adjusted p
values, Table S5 the exploratory tests of the share with an ending, Table S6 the personality edits,
Table S7 the transfer shares under the rule without its two exclusions, Table S8 the fidelity on the
held-out chapters, Table S9 the memorization probe and Table S10 how often an addressed character
moved next. Run: python scripts/make_supplement.py --out paper/supplement.tex"""

import argparse
import json
import os

S = "results/summary"
BASE = {"llama": "Llama 3.1 8B", "qwen": "Qwen3 8B", "qwen_pseudo": "Qwen3 8B, pseudonyms"}
SCENE = {"pearls": "Pearls", "sell_the_land": "Land"}
COND = {
    "prompted_tools-off": "Prompted", "prompted_tools-on": "Prompted + memory", "world_tools-on": "World + memory",
    "soul_tools-off": "Soul", "soul_tools-on": "Soul + memory", "soul_tools-on_director": "Soul + director",
    "soul_tools-on_rich": "Soul + rich cards", "soul_tools-on_rich_director": "Soul + rich cards + director",
    "prompted_tools-on_rich": "Prompted + rich cards", "prompted_tools-on_rich_coser": "Known plot",
    "soul_tools-on_t4": "+ plan", "soul_tools-on_t6": "+ scene check", "soul_tools-on_t4_t5_t6": "+ plan, state, scene check",
    "soul_tools-on_t4_t5_t6_t7_t8": "All but turn-taking", "soul_tools-on_t4_t5_t6_t7_t9": "All but choice",
    "soul_tools-on_t4_t5_t6_t7_t8_t9": "All character tools", "soul_tools-t1": "Soul + relations only",
    "soul_tools-t2": "Soul + events only", "soul_tools-on_t3": "Soul + memory + structure"}
METRIC = {"original_class": "Ending of the novel", "state_changed": "Transfer", "anachronism": "Anachronism"}
CHAR = {"o_lan": "O-lan", "wang_lung": "Wang Lung", "eldest_son": "Eldest son", "second_son": "Second son"}


def esc(s):
    return s.replace("&", "\\&").replace("%", "\\%").replace("_", "\\_")


def p(x):
    return "" if x is None else ("$<$0.0001" if x < 0.0001 else f"{x:.4f}")


def longtable(caption, label, cols, head, rows):
    out = [f"\\begin{{longtable}}{{{cols}}}", f"\\caption{{{caption}}}\\label{{{label}}}\\\\", "\\toprule",
           head + " \\\\", "\\midrule", "\\endfirsthead", "\\toprule", head + " \\\\", "\\midrule", "\\endhead"]
    out += [" & ".join(r) + " \\\\" for r in rows]
    out += ["\\bottomrule", "\\end{longtable}", ""]
    return "\n".join(out)


def tests_table(tests, caption, label):
    rows = [[BASE[t["run"]], SCENE[t["scene"]], esc(COND.get(t["a"], t["a"])), esc(COND.get(t["b"], t["b"])),
             METRIC[t["metric"]], f"{t['mean_a']:.2f}", f"{t['mean_b']:.2f}", p(t["p"]), p(t["p_holm"])]
            for t in tests if t.get("p") is not None]
    return longtable(caption, label, "llp{3.2cm}p{3.0cm}lrrrr",
                     "Base & Scene & Condition a & Condition b & Measure & a & b & p & Holm p", rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="paper/supplement.tex")
    args = ap.parse_args()
    s = json.load(open(f"{S}/summary.json", encoding="utf-8"))
    old = json.load(open(f"{S}/summary_transfer_rule_without_exclusions.json", encoding="utf-8"))
    parts = [r"""\documentclass[10pt]{article}
\usepackage[letterpaper,margin=2cm]{geometry}
\usepackage{booktabs,longtable,array}
\usepackage[T1]{fontenc}
\usepackage{mathptmx}
\title{Supplemental Information\\[4pt]\large Character determines incident: director-free language model agents for testing fictional characters}
\author{Lili Zhao and Hui Shi}
\date{}
\renewcommand{\thetable}{S\arabic{table}}
\setlength{\tabcolsep}{4pt}
\begin{document}
\maketitle
\footnotesize
This document lists the complete statistical results behind the article. Anachronism is the share of sessions that mention a character who enters the novel only after the chapter of the scene. No session did. Every number is drawn by the script make\_supplement.py of the repository from its result files. Shares are proportions of sessions. All tests are two-sided. The p values of the permutation tests come from 10,000 permutations, and the Holm correction runs within each family. The planned family was fixed before the simulations. The other families were added after the main results and are exploratory.
"""]
    parts.append(tests_table(s["tests"], "Planned contrasts of the main design.", "tab:s1"))
    parts.append(tests_table(s["followup_tests"], "Baselines, an exploratory family.", "tab:s2"))
    parts.append(tests_table(s["tool_tests"], "Plan, state and scene check, an exploratory family.", "tab:s3"))
    parts.append(tests_table(s["tool2_tests"], "Consistency check, choice and turn-taking, an exploratory family.",
                             "tab:s4"))
    d = json.load(open(f"{S}/decided_tests.json", encoding="utf-8"))
    rows = [[BASE[t["run"]], SCENE[t["scene"]], esc(COND[t["b"]]), f"{t['share_a']:.2f}", f"{t['share_b']:.2f}",
             str(t["n_a"]), str(t["n_b"]), p(t["p"]), p(t["p_holm"])] for t in d]
    parts.append(longtable("Share of sessions with an ending, all character tools against two conditions. Fisher exact "
                           "tests with the Holm correction over the eight comparisons, an exploratory analysis.",
                           "tab:s5", "lllrrrrrr",
                           "Base & Scene & Compared with & All tools & Other & n & n & p & Holm p", rows))
    pe = json.load(open(f"{S}/personality.json", encoding="utf-8"))
    rows = []
    for r in pe:
        cls = ", ".join(f"{k} {v:.2f}" for k, v in r["classes"].items() if v)
        rows.append([BASE[r["run"]], SCENE[r["scene"]], CHAR[r["character"]], esc(r["trait"]), r["method"], r["direction"],
                     esc(cls), f"{r['tv']:.2f}", p(r["p"]), p(r["p_holm"]),
                     "" if r["holds"] is None else ("yes" if r["holds"] else "no")])
    parts.append(longtable("Personality edits. Class shares of the edited sessions (split, no majority), total variation "
                           "distance to the unedited soul sessions with its permutation test and Holm correction over "
                           "the 80 edited conditions, and whether the prediction written before the simulations held.",
                           "tab:s6", "llp{1.5cm}p{2.0cm}llp{3.4cm}rrrl",
                           "Base & Scene & Character & Trait & Edit & Dir. & Classes & TV & p & Holm p & Held", rows))
    rows = []
    for k in sorted(s["conditions"]):
        run, scene, cond = k.split("/")
        a, b = s["conditions"][k].get("state_changed"), old["conditions"][k].get("state_changed")
        if scene != "pearls" or a is None or cond not in COND:
            continue
        rows.append([BASE[run], esc(COND[cond]), f"{a:.2f}", f"{b:.2f}"])
    parts.append(longtable("Transfer share of the pearls scene under the rule of the article and under the same rule "
                           "without its two exclusions (sentences about pearls that Wang Lung already holds or about "
                           "taking O-lan somewhere, and giving that O-lan only fears or intends).", "tab:s7", "llrr",
                           "Base & Condition & With exclusions & Without", rows))
    rows = []
    for run in ("llama", "qwen", "qwen_pseudo"):
        for c in ("c19", "c34"):
            f = json.load(open(f"results/fidelity/{run}/{c}/summary.json", encoding="utf-8"))
            for ch, cfgs in f.items():
                for cfg, v in cfgs.items():
                    rows.append([BASE[run], c[1:], CHAR[ch], cfg, str(v["samples"]), f"{v['accuracy']:.2f}",
                                 f"{v['macro_f1']:.2f}", f"{v['urgency_auc']:.2f}", f"{v['nll']:.2f}"])
    parts.append(longtable("Fidelity on the decision points of the held-out chapters: accuracy and macro F1 of the most "
                           "probable intention, area under the curve of the urgency, and negative log-likelihood per "
                           "token of the original move.", "tab:s8", "llllrrrrr",
                           "Base & Chapter & Character & Configuration & n & Accuracy & Macro F1 & AUC & NLL", rows))
    pr = json.load(open("data/derived/probe/summary.json", encoding="utf-8"))
    rows = [[r["model"], SCENE[r["scene"]], r["mc_title"], r["mc_plain"], r["ctx_title"], r["ctx_plain"],
             f"{r['verbatim']:.2f}"] for r in pr]
    parts.append(longtable("Memorization probe of the base models. Rotations of the four ending classes in which the "
                           "model chose the ending of the novel, from the set-up with title and author, with pseudonyms, "
                           "and from the text before the decision with real names and with pseudonyms, and the share of "
                           "the last 40 words of the scene recovered in order.", "tab:s9", "llrrrrr",
                           "Model & Scene & Title & Pseudonyms & Text & Text, pseudonyms & Verbatim", rows))
    an = json.load(open(f"{S}/addressee_next.json", encoding="utf-8"))
    rows = [[BASE[r["run"]], SCENE[r["scene"]], esc(COND[r["condition"]]), str(r["addressed"]), str(r["next"]),
             f"{r['share']:.2f}"] for r in an]
    parts.append(longtable("Moves addressed to other characters of the scene and the share after which an addressee "
                           "moved next.", "tab:s10", "lllrrr", "Base & Scene & Condition & Addressed & Next & Share", rows))
    parts.append("\\end{document}\n")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write("\n".join(parts))


if __name__ == "__main__":
    main()
