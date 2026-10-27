"""Diagrams of the paper, drawn with boxes, arrows and text only (no icons, no images).

Fig1.png  concept: the writer's cards and scene opening, the director-free scene with the
          character agents and their tools, and the comparison of the ending with the chapter
Fig2.png  method: offline annotation and training, online move with the character tools
Run: python scripts/make_diagrams.py --out paper"""

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "Liberation Serif", "DejaVu Serif"],
    "mathtext.fontset": "custom", "mathtext.rm": "Times New Roman", "mathtext.it": "Times New Roman:italic",
    "mathtext.fallback": "stix", "font.size": 8,
    "savefig.dpi": 300, "savefig.bbox": "tight", "savefig.pad_inches": 0.03, "savefig.facecolor": "white"})

# one fill per kind of component, used in both diagrams
FILL = {"writer": "#FFFFFF", "data": "#EEEEEE", "model": "#DCEAF5", "tool": "#FBE7C6", "eval": "#E3F1E8",
        "frame": "#FFFFFF"}


def canvas(w, h):
    fig = plt.figure(figsize=(w, h))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    ax.axis("off")
    return fig, ax


def box(ax, x, y, w, h, text, kind="model", size=8, bold=False, dashed=False, align="center", pad=0.06):
    """A box with its lower left corner at (x, y); returns the box as (x, y, w, h)."""
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.05", linewidth=0.7,
                                edgecolor="black", facecolor=FILL[kind], linestyle="--" if dashed else "-"))
    if align == "center":
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=size,
                fontweight="bold" if bold else "normal", linespacing=1.25)
    else:
        ax.text(x + pad, y + h - pad, text, ha="left", va="top", fontsize=size,
                fontweight="bold" if bold else "normal", linespacing=1.25)
    return (x, y, w, h)


def side(b, where):
    x, y, w, h = b
    return {"l": (x, y + h / 2), "r": (x + w, y + h / 2), "t": (x + w / 2, y + h), "b": (x + w / 2, y)}[where]


def arrow(ax, p, q, text=None, rad=0.0, size=7, offset=(0, 0.06), style="-|>", ls="-"):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle=style, mutation_scale=8, linewidth=0.7, color="black",
                                 connectionstyle=f"arc3,rad={rad}", linestyle=ls, shrinkA=1, shrinkB=1))
    if text:
        ax.text((p[0] + q[0]) / 2 + offset[0], (p[1] + q[1]) / 2 + offset[1], text, ha="center", va="bottom",
                fontsize=size, style="italic")


def fig_concept(out):
    W, H = 6.5, 2.25
    fig, ax = canvas(W, H)
    # writer
    box(ax, 0.02, 0.08, 1.55, 2.05, "Writer", kind="frame", bold=True, align="left")
    cards = box(ax, 0.15, 1.12, 1.3, 0.62, "Character cards\nbackground, personality,\nwhat each one wants")
    opening = box(ax, 0.15, 0.25, 1.3, 0.62, "Opening of a scene\nplace, people present,\nlines before the scene")
    # scene
    box(ax, 1.95, 0.08, 2.6, 2.05, "Director-free scene", kind="frame", bold=True, align="left")
    wl = box(ax, 2.1, 1.35, 1.1, 0.45, "Wang Lung\nown fine-tuned model")
    ol = box(ax, 3.3, 1.35, 1.1, 0.45, "O-lan\nown fine-tuned model")
    ax.text(3.25, 1.12, "most urgent moves next", ha="center", va="center", fontsize=7, style="italic")
    tools = box(ax, 2.1, 0.25, 2.3, 0.62, "Character tools\nplan, state, scene check,\nconsistency check, choice, turn-taking",
                kind="tool")
    arrow(ax, (2.4, 0.87), (2.4, 1.35))
    arrow(ax, (4.1, 0.87), (4.1, 1.35))
    # evaluation
    box(ax, 4.93, 0.08, 1.55, 2.05, "Comparison", kind="frame", bold=True, align="left")
    sim = box(ax, 5.05, 1.12, 1.3, 0.62, "Simulated ending\nclass A, B, C or D\nby three judges", kind="eval")
    nov = box(ax, 5.05, 0.25, 1.3, 0.62, "Chapter of the novel\nclass A, the ending\nof the novel", kind="data")
    arrow(ax, side(sim, "b"), side(nov, "t"), style="<|-|>")
    arrow(ax, side(cards, "r"), (2.1, 1.68))
    arrow(ax, side(opening, "r"), (2.1, 1.47), rad=-0.25)
    arrow(ax, side(ol, "r"), side(sim, "l"))
    # a wide, low diagram: a higher resolution keeps its short side above the 900 pixels that PeerJ asks for
    fig.savefig(os.path.join(out, "Fig1.png"), dpi=440)
    plt.close(fig)


def fig_method(out):
    W, H = 6.5, 4.35
    fig, ax = canvas(W, H)
    # offline
    box(ax, 0.02, 2.28, 6.46, 2.03, "Offline, for a scene in chapter $c$", kind="frame", bold=True, align="left")
    text = box(ax, 0.15, 3.3, 1.05, 0.62, "Chapters\n1 to $c-1$", kind="data")
    ann = box(ax, 1.45, 3.3, 1.4, 0.62, "Two local annotators\nquotations, beats,\nevents, relations")
    chk = box(ax, 3.1, 3.3, 1.3, 0.62, "Evidence check\nquoted span found\nin the window", kind="tool")
    mem = box(ax, 4.65, 3.48, 1.7, 0.44, "Memory stores\nrelations T1, events T2", kind="data")
    dp = box(ax, 3.1, 2.45, 1.3, 0.62, "Decision points\ninput as perceived,\ntarget move or wait", kind="data")
    world = box(ax, 0.15, 2.45, 1.4, 0.62, "Story-world adapter\nlanguage and world\nof the novel, QLoRA")
    soul = box(ax, 1.8, 2.45, 1.05, 0.62, "Soul adapter\nper main character\nQLoRA")
    arrow(ax, side(text, "r"), side(ann, "l"))
    arrow(ax, side(ann, "r"), side(chk, "l"))
    arrow(ax, side(chk, "r"), side(mem, "l"))
    arrow(ax, side(chk, "b"), side(dp, "t"))
    arrow(ax, side(text, "b"), side(world, "t"))
    arrow(ax, side(dp, "l"), side(soul, "r"))
    arrow(ax, (1.8, 2.76), (1.55, 2.76))
    ax.text(1.675, 2.36, "stacked on", ha="center", va="center", fontsize=7, style="italic")
    # online
    box(ax, 0.02, 0.04, 6.46, 2.1, "Online, one move of the scene", kind="frame", bold=True, align="left")
    inp = box(ax, 0.15, 1.05, 1.35, 0.75, "Every character present\ncard, circumstances,\nT1 and T2 results,\nstate of T5")
    urg = box(ax, 1.75, 1.18, 1.1, 0.5, "Urgency\n$1-\\Pr(\\mathrm{wait})$")
    draw = box(ax, 3.1, 1.18, 1.3, 0.5, "Draw who moves\naddressees only with T9")
    plan = box(ax, 4.65, 1.18, 1.7, 0.5, "Plan T4\nwhat it wants here and now", kind="tool")
    cand = box(ax, 4.65, 0.22, 1.7, 0.6, "Candidate moves\nwritten by its soul,\nchecked by T6 and T7", kind="tool")
    choose = box(ax, 3.1, 0.22, 1.3, 0.6, "Choice T8\none of three,\nas the character", kind="tool")
    move = box(ax, 1.75, 0.22, 1.1, 0.6, "Move\nspeak, act\nor keep silent", kind="eval")
    state = box(ax, 0.15, 0.22, 1.35, 0.6, "State machine\nplace and holder\nof the object", kind="data")
    arrow(ax, side(inp, "r"), side(urg, "l"))
    arrow(ax, side(urg, "r"), side(draw, "l"))
    arrow(ax, side(draw, "r"), side(plan, "l"))
    arrow(ax, side(plan, "b"), side(cand, "t"))
    arrow(ax, side(cand, "l"), side(choose, "r"))
    arrow(ax, side(choose, "l"), side(move, "r"))
    arrow(ax, side(move, "l"), side(state, "r"))
    arrow(ax, side(state, "t"), side(inp, "b"), text="next move", offset=(0.38, -0.08))
    arrow(ax, (2.45, 0.82), (3.6, 1.18), text="addressees", rad=-0.2, offset=(-0.05, -0.12))
    # the adapters and memories serve the online part
    arrow(ax, side(soul, "b"), (2.3, 1.68), ls="--")
    arrow(ax, (5.5, 3.48), (5.5, 1.68), ls="--")
    ax.text(5.55, 2.2, "memory results", ha="left", va="center", fontsize=7, style="italic")
    fig.savefig(os.path.join(out, "Fig2.png"))
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="paper")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    fig_concept(args.out)
    fig_method(args.out)


if __name__ == "__main__":
    main()
