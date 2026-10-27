"""Data figures of the paper, drawn from the result files (no number is typed in).

Fig4.png  fidelity on the held-out chapters: negative log-likelihood of the original moves and
          AUC of the urgency, per character and configuration (results/fidelity)
Fig5.png  ending classes of the main conditions, the baselines and all character tools
          (majority of the three judges, results/summary/sessions.csv)
Fig6.png  share of sessions with an ending, with the ending of the novel and with a character
          leaving the scene, for the character tools and their ablations
Run: python scripts/make_figures.py --out paper"""

import argparse
import collections
import csv
import json
import math
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "Liberation Serif", "DejaVu Serif"],
    "mathtext.fontset": "custom", "mathtext.rm": "Times New Roman", "mathtext.it": "Times New Roman:italic",
    "mathtext.bf": "Times New Roman:bold", "mathtext.fallback": "stix",
    "font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "savefig.dpi": 300, "savefig.bbox": "tight", "savefig.pad_inches": 0.03, "savefig.facecolor": "white"})
WIDTH = 6.5  # text width of the PeerJ page in inches

# Okabe-Ito colours, one per ending class, with hatches for print in grey
CLASS_STYLE = {"A": ("#0072B2", ""), "B": ("#56B4E9", "//"), "C": ("#E69F00", "\\\\"),
               "D": ("#BBBBBB", ""), "split": ("#FFFFFF", "..")}
CLASS_LABEL = {"A": "A, ending of the novel", "B": "B", "C": "C", "D": "D, no decision", "split": "No majority"}
CONFIG_STYLE = {"prompted": ("#E69F00", "s"), "world": ("#009E73", "^"), "soul": ("#0072B2", "o")}
SCENES = [("qwen", "pearls"), ("llama", "pearls"), ("qwen", "sell_the_land"), ("llama", "sell_the_land")]
BASE_NAME = {"qwen": "Qwen3 8B", "llama": "Llama 3.1 8B"}
SCENE_NAME = {"pearls": "pearls, chapter 19", "sell_the_land": "land, chapter 34"}

MAIN = [("prompted_tools-off", "Prompted"), ("prompted_tools-on", "Prompted + memory"),
        ("world_tools-on", "World + memory"), ("soul_tools-off", "Soul"), ("soul_tools-on", "Soul + memory"),
        ("soul_tools-on_director", "Soul + director"), ("prompted_tools-on_rich_coser", "Known plot"),
        ("soul_tools-on_t4_t5_t6_t7_t8_t9", "All character tools")]
TOOLS = [("soul_tools-on", "Soul + memory"), ("soul_tools-on_t6", "+ scene check"), ("soul_tools-on_t4", "+ plan"),
         ("soul_tools-on_t4_t5_t6", "+ plan, state, scene check"), ("soul_tools-on_t4_t5_t6_t7_t8", "All but turn-taking"),
         ("soul_tools-on_t4_t5_t6_t7_t9", "All but choice"), ("soul_tools-on_t4_t5_t6_t7_t8_t9", "All character tools"),
         ("soul_tools-on_director", "Soul + director")]


def load_sessions(path):
    rows = collections.defaultdict(list)
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows[(r["run"], r["scene"], r["condition"])].append(r)
    return rows


def wilson(k, n, z=1.96):
    """95% Wilson interval of a proportion."""
    if n == 0:
        return 0.0, 0.0
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return max(0.0, centre - half), min(1.0, centre + half)


def panel_label(ax, letter, x=-0.02):
    ax.text(x, 1.02, letter, transform=ax.transAxes, fontsize=9, fontweight="bold", ha="right", va="bottom")


def fig_fidelity(out):
    chars = [("c19", "o_lan", "O-lan, 19"), ("c19", "wang_lung", "Wang Lung, 19"),
             ("c34", "wang_lung", "Wang Lung, 34"), ("c34", "eldest_son", "Eldest son, 34")]
    fig, axes = plt.subplots(2, 2, figsize=(WIDTH, 3.6), sharex=True)
    for j, run in enumerate(("qwen", "llama")):
        res = {c: json.load(open(f"results/fidelity/{run}/{c}/summary.json", encoding="utf-8")) for c in ("c19", "c34")}
        for i, (metric, label) in enumerate((("nll", "NLL per token"), ("urgency_auc", "Urgency AUC"))):
            ax = axes[i, j]
            for k, cfg in enumerate(("prompted", "world", "soul")):
                colour, marker = CONFIG_STYLE[cfg]
                xs = [x + (k - 1) * 0.18 for x in range(len(chars))]
                ys = [res[c][ch][cfg][metric] for c, ch, _ in chars]
                ax.plot(xs, ys, marker, color=colour, markersize=5, markeredgecolor="black", markeredgewidth=0.4)
            if metric == "urgency_auc":
                ax.axhline(0.5, color="black", linewidth=0.5, linestyle=":")
                ax.set_ylim(0, 1)
            else:
                ax.set_ylim(bottom=0, top=3)
            ax.set_ylabel(label)
            if i == 0:
                ax.text(0.5, 1.02, BASE_NAME[run], transform=ax.transAxes, ha="center", va="bottom")
            ax.set_xticks(range(len(chars)))
            ax.set_xticklabels([c[2] for c in chars])
            ax.grid(axis="y", linewidth=0.3, color="#DDDDDD")
            ax.set_axisbelow(True)
            panel_label(ax, "ABCD"[i * 2 + j], x=-0.12)
    handles = [Line2D([], [], linestyle="", marker=CONFIG_STYLE[c][1], color=CONFIG_STYLE[c][0], markeredgecolor="black",
                      markeredgewidth=0.4, label=c.capitalize()) for c in ("prompted", "world", "soul")]
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.03))
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(os.path.join(out, "Fig4.png"))
    plt.close(fig)


def fig_endings(rows, out):
    fig, axes = plt.subplots(2, 2, figsize=(WIDTH, 4.6), sharey=True)
    for idx, (run, scene) in enumerate(SCENES):
        ax = axes[idx // 2, idx % 2]
        labels = []
        for y, (cond, name) in enumerate(MAIN):
            rs = rows.get((run, scene, cond), [])
            labels.append(name)
            left = 0.0
            for cls in ("A", "B", "C", "D", "split"):
                share = sum((r["ending"] or "split") == cls for r in rs) / len(rs) if rs else 0
                colour, hatch = CLASS_STYLE[cls]
                ax.barh(y, share, left=left, color=colour, hatch=hatch, edgecolor="black", linewidth=0.4, height=0.7)
                left += share
        ax.set_yticks(range(len(MAIN)))
        ax.set_yticklabels(labels)
        ax.set_ylim(len(MAIN) - 0.5, -0.5)
        ax.set_xlim(0, 1)
        ax.set_xlabel("Share of sessions")
        ax.text(0.5, 1.02, f"{BASE_NAME[run]}, {SCENE_NAME[scene]}", transform=ax.transAxes, ha="center", va="bottom")
        panel_label(ax, "ABCD"[idx], x=-0.02 if idx % 2 else -0.45)
    handles = [Patch(facecolor=CLASS_STYLE[c][0], hatch=CLASS_STYLE[c][1], edgecolor="black", linewidth=0.4,
                     label=CLASS_LABEL[c]) for c in ("A", "B", "C", "D", "split")]
    fig.legend(handles=handles, loc="upper center", ncol=5, frameon=False, bbox_to_anchor=(0.55, 1.03))
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(os.path.join(out, "Fig5.png"))
    plt.close(fig)


def fig_tools(rows, out):
    series = [("decided", "With an ending", "#0072B2", "o"), ("original", "Ending of the novel", "#E69F00", "s"),
              ("away", "A character leaves", "#999999", "^")]
    fig, axes = plt.subplots(2, 2, figsize=(WIDTH, 4.4), sharex=True, sharey=True)
    for idx, (run, scene) in enumerate(SCENES):
        ax = axes[idx // 2, idx % 2]
        for s, (key, _, colour, marker) in enumerate(series):
            for x, (cond, _) in enumerate(TOOLS):
                rs = rows.get((run, scene, cond), [])
                if not rs:
                    continue
                if key == "decided":
                    k = sum(r["ending"] in ("A", "B", "C") for r in rs)
                elif key == "original":
                    k = sum(r["ending"] == "A" for r in rs)
                else:
                    k = sum(int(r["away"]) for r in rs)
                lo, hi = wilson(k, len(rs))
                xx = x + (s - 1) * 0.22
                ax.plot([xx, xx], [lo, hi], color=colour, linewidth=0.8)
                ax.plot(xx, k / len(rs), marker, color=colour, markersize=4.5, markeredgecolor="black",
                        markeredgewidth=0.4)
        ax.set_ylim(0, 1)
        ax.set_xlim(-0.6, len(TOOLS) - 0.4)
        ax.axvline(len(TOOLS) - 1.5, color="black", linewidth=0.5, linestyle=":")
        ax.set_xticks(range(len(TOOLS)))
        ax.set_xticklabels([t[1] for t in TOOLS], rotation=35, ha="right")
        ax.grid(axis="y", linewidth=0.3, color="#DDDDDD")
        ax.set_axisbelow(True)
        if idx % 2 == 0:
            ax.set_ylabel("Share of sessions")
        ax.text(0.5, 1.02, f"{BASE_NAME[run]}, {SCENE_NAME[scene]}", transform=ax.transAxes, ha="center", va="bottom")
        panel_label(ax, "ABCD"[idx], x=-0.02 if idx % 2 else -0.12)
    handles = [Line2D([], [], linestyle="-", linewidth=0.8, marker=m, color=c, markeredgecolor="black",
                      markeredgewidth=0.4, label=lab) for _, lab, c, m in series]
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.03))
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(os.path.join(out, "Fig6.png"))
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sessions", default="results/summary/sessions.csv")
    ap.add_argument("--out", default="paper")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    rows = load_sessions(args.sessions)
    fig_fidelity(args.out)
    fig_endings(rows, args.out)
    fig_tools(rows, args.out)


if __name__ == "__main__":
    main()
