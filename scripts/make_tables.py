"""Tables of the paper as LaTeX fragments, drawn from the result files (no number is typed in).

tab_endings.tex   share of sessions with an ending, with the ending of the novel and with the
                  pearls handed over, per condition, base model and scene (sessions.csv)
tab_judges.tex    accuracy of every judge on the controls of the ending class and the agreement
                  of the judges on the sessions (summary.json)
tab_training.tex  data and cost of every adapter (adapters/*/c*/*/train_log.json)
Run: python scripts/make_tables.py --out paper"""

import argparse
import collections
import csv
import glob
import json
import os

COLUMNS = [("qwen", "pearls", True), ("llama", "pearls", True), ("qwen", "sell_the_land", False),
           ("llama", "sell_the_land", False)]
GROUPS = [
    ("Main design", [("prompted_tools-off", "Prompted"), ("prompted_tools-on", "Prompted + memory"),
                     ("world_tools-on", "World + memory"), ("soul_tools-off", "Soul"),
                     ("soul_tools-on", "Soul + memory")]),
    ("Memory and structure", [("soul_tools-t1", "Soul + relations only"), ("soul_tools-t2", "Soul + events only"),
                              ("soul_tools-on_t3", "Soul + memory + structure")]),
    ("Baselines", [("soul_tools-on_director", "Soul + director"), ("soul_tools-on_rich", "Soul + rich cards"),
                   ("soul_tools-on_rich_director", "Soul + rich cards + director"),
                   ("prompted_tools-on_rich", "Prompted + rich cards"),
                   ("prompted_tools-on_rich_coser", "Known plot")]),
    ("Character tools", [("soul_tools-on_t6", "+ scene check"), ("soul_tools-on_t4", "+ plan"),
                         ("soul_tools-on_t4_t5_t6", "+ plan, state, scene check"),
                         ("soul_tools-on_t4_t5_t6_t7_t8", "All but turn-taking"),
                         ("soul_tools-on_t4_t5_t6_t7_t9", "All but choice"),
                         ("soul_tools-on_t4_t5_t6_t7_t8_t9", "All character tools")]),
]
JUDGE_NAME = {"phi4:14b": "Phi-4 14B", "gemma3:12b": "Gemma 3 12B", "mistral-nemo:12b": "Mistral NeMo 12B"}


def share(rs, test):
    return f"{sum(test(r) for r in rs) / len(rs):.2f}" if rs else ""


def endings_table(sessions, out):
    rows = collections.defaultdict(list)
    with open(sessions, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows[(r["run"], r["scene"], r["condition"])].append(r)
    lines = []
    for g, (group, conds) in enumerate(GROUPS):
        if g:
            lines.append("\\midrule")
        lines.append(f"\\multicolumn{{11}}{{l}}{{\\textit{{{group}}}}} \\\\")
        for cond, name in conds:
            cells = []
            for run, scene, pearls in COLUMNS:
                rs = rows.get((run, scene, cond), [])
                cells.append(share(rs, lambda r: r["ending"] in ("A", "B", "C")))
                cells.append(share(rs, lambda r: r["ending"] == "A"))
                if pearls:
                    cells.append(share(rs, lambda r: r["state_changed"] == "1"))
            lines.append(f"{name} & " + " & ".join(cells) + " \\\\")
    with open(os.path.join(out, "tab_endings.tex"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def judges_table(summary, out):
    with open(summary, encoding="utf-8") as f:
        s = json.load(f)
    lines = []
    for j in s["judges"]:
        lines.append(f"{JUDGE_NAME.get(j, j)} & {s['control_accuracy'][j] * 8:.0f} of 8 \\\\")
    lines.append("\\midrule")
    lines.append(f"Krippendorff's $\\alpha$ of the ending class on the sessions & {s['alpha_ending']:.2f} \\\\")
    with open(os.path.join(out, "tab_judges.tex"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def training_table(adapters, out):
    name = {"world": "Story world", "soul_wang_lung": "Wang Lung", "soul_o_lan": "O-lan",
            "soul_eldest_son": "Eldest son", "soul_cast": "Supporting cast"}
    lines = []
    for chapter in ("c19", "c34"):
        for adapter in ("world", "soul_wang_lung", "soul_o_lan", "soul_eldest_son", "soul_cast"):
            logs = {run: os.path.join(adapters, run, chapter, adapter, "train_log.json") for run in ("llama", "qwen")}
            if not all(os.path.exists(p) for p in logs.values()):
                continue
            d = {run: json.load(open(p, encoding="utf-8")) for run, p in logs.items()}
            rows = d["llama"]["train_rows"] + d["llama"]["val_rows"]
            unit = "segments" if adapter == "world" else "points"
            cells = [f"{chapter[1:]}", name[adapter], f"{rows:,} {unit}"]
            for run in ("llama", "qwen"):
                cells += [f"{d[run]['seconds'] / 60:.0f}", f"{d[run]['peak_gpu_gb']:.1f}"]
            lines.append(" & ".join(cells) + " \\\\")
    with open(os.path.join(out, "tab_training.tex"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sessions", default="results/summary/sessions.csv")
    ap.add_argument("--summary", default="results/summary/summary.json")
    ap.add_argument("--adapters", default="adapters")
    ap.add_argument("--out", default="paper")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    endings_table(args.sessions, args.out)
    judges_table(args.summary, args.out)
    if glob.glob(os.path.join(args.adapters, "*", "c*", "*", "train_log.json")):
        training_table(args.adapters, args.out)


if __name__ == "__main__":
    main()
