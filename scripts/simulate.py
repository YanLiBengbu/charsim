"""Simulations of a test scene under a grid of conditions (resumable).

A condition is a mode (prompted, world, soul), tools on or off, optionally T3 shown to the
characters and optionally one card edit "character:trait:up|down". The follow-up conditions add
the rich cards and circumstances of config/cards_rich.json (--rich) and the director or CoSER
baseline with the texts of config/baselines.json (--baseline); the second follow-up adds the plan
tool T4 (--t4), the state tool T5 with the texts of config/state_texts.json (--t5) and the scene
check T6 (--t6); the third adds the consistency check T7 (--t7), the choice T8 (--t8), the
turn-taking T9 (--t9) and the texts of T5 in config/state_texts2.json (--state-texts). Every session is written to results/sim/<run>/<scene>/<condition>/seed<k>.json."""

import argparse
import glob
import itertools
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from charsim import pseudonyms  # noqa: E402
from charsim.agent import AgentModel  # noqa: E402
from charsim.cast import Cast  # noqa: E402
from charsim.samples import build_tools, timeline  # noqa: E402
from charsim.scene import SceneRun  # noqa: E402


def opening_items(data, scene):
    """The timeline of the scene's chapter before the paragraph of its start anchor."""
    c = scene["chapter"]
    with open(os.path.join(data, "text/chapters.json"), encoding="utf-8") as f:
        paras = next(ch for ch in json.load(f)["chapters"] if ch["chapter"] == c)["paragraphs"]
    anchor = next(p["pid"] for p in paras if scene["start_anchor"] in p["text"])
    with open(os.path.join(data, "derived/quotes_final.jsonl"), encoding="utf-8") as f:
        quotes = [q for q in map(json.loads, f) if q["chapter"] == c]
    with open(os.path.join(data, "derived/beats_final.jsonl"), encoding="utf-8") as f:
        beats = [b for b in map(json.loads, f) if b["chapter"] == c and b["grounded"] and b["valid"]]
    return [it for it in timeline(c, quotes, beats) if it["pid"] < anchor]


def seeds(spec):
    lo, _, hi = spec.partition("-")
    return list(range(int(lo), int(hi or lo) + 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--run", required=True, help="adapter set, for example llama or qwen_pseudo")
    ap.add_argument("--scene", required=True)
    ap.add_argument("--modes", nargs="+", default=["soul"], choices=["prompted", "world", "soul"])
    ap.add_argument("--tools", nargs="+", default=["on"], choices=["on", "off", "t1", "t2"])
    ap.add_argument("--t3-to-agents", action="store_true")
    ap.add_argument("--edit", default=None, help="character:trait:up|down")
    ap.add_argument("--steer", default=None, help="character:trait:coefficient (persona vector of results/steer)")
    ap.add_argument("--rich", action="store_true", help="cards and circumstances of config/cards_rich.json")
    ap.add_argument("--baseline", default=None, choices=["director", "coser"])
    ap.add_argument("--t4", action="store_true", help="plan tool: every character plans what it wants and does next")
    ap.add_argument("--t5", action="store_true", help="state tool: the place and the holder of the scene object")
    ap.add_argument("--t6", action="store_true", help="scene check: a move away from the scene is drawn again")
    ap.add_argument("--t7", action="store_true", help="consistency check: an action impossible in the state is drawn again")
    ap.add_argument("--t8", action="store_true", help="choice: the character chooses one of three moves")
    ap.add_argument("--t9", action="store_true", help="turn-taking: the characters spoken to answer")
    ap.add_argument("--state-texts", default="config/state_texts.json", help="texts of T5")
    ap.add_argument("--seeds", default="0-19")
    ap.add_argument("--max-turns", type=int, default=20)
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="results/sim")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    with open("config/scenes.json", encoding="utf-8") as f:
        scene = next(s for s in json.load(f)["scenes"] if s["id"] == args.scene)
    with open("config/cards_rich.json" if args.rich else "config/cards.json", encoding="utf-8") as f:
        cards = json.load(f)
    if args.rich:
        scene = scene | {"circumstances": cards["circumstances"][scene["id"]]}
    if args.baseline:
        with open("config/baselines.json", encoding="utf-8") as f:
            texts = json.load(f)
        scene = scene | {"premise": texts["premise"][scene["id"]], "plot": texts["plot"][scene["id"]]}
    if args.t5:
        with open(args.state_texts, encoding="utf-8") as f:
            scene = scene | {"state_texts": json.load(f)[scene["id"]]}
    cast = Cast("config/cast.json")
    c = scene["chapter"]
    adir = f"adapters/{args.run}/c{c}"
    souls = sorted(os.path.basename(p)[5:] for p in glob.glob(f"{adir}/soul_*")
                   if os.path.isfile(f"{p}/train_log.json"))
    agent = AgentModel(args.base, {"world": f"{adir}/world"} | {f"soul_{s}": f"{adir}/soul_{s}" for s in souls})
    table = pseudonyms.load("config/pseudonyms.json") if args.run.endswith("_pseudo") else None
    tools = build_tools(args.data, cast, c, 1)
    opening = opening_items(args.data, scene)
    edits, steer = {}, {}
    if args.edit:
        cid, trait, direction = args.edit.split(":")
        edits = {cid: [trait, direction]}
    if args.steer:
        cid, trait, alpha = args.steer.split(":")
        steer = {cid: {"path": f"results/steer/{args.run}/{args.scene}/{cid}__{trait.replace(' ', '_')}.pt",
                       "alpha": float(alpha)}}

    for mode, tl in itertools.product(args.modes, args.tools):
        name = f"{mode}_tools-{tl}" + ("_t3" if args.t3_to_agents else "") + ("_t4" if args.t4 else "") + \
               ("_t5" if args.t5 else "") + ("_t6" if args.t6 else "") + ("_t7" if args.t7 else "") + \
               ("_t8" if args.t8 else "") + ("_t9" if args.t9 else "") + \
               (f"_edit-{args.edit.replace(':', '-').replace(' ', '_')}" if args.edit else "") + \
               (f"_steer-{args.steer.replace(':', '-').replace(' ', '_')}" if args.steer else "") + \
               ("_rich" if args.rich else "") + (f"_{args.baseline}" if args.baseline else "")
        out_dir = os.path.join(args.out, args.run, args.scene, name)
        os.makedirs(out_dir, exist_ok=True)
        for s in seeds(args.seeds):
            path = os.path.join(out_dir, f"seed{s}.json")
            if os.path.exists(path):
                continue
            t0 = time.time()
            cfg = {"mode": mode, "tools": tl, "t3_to_agents": args.t3_to_agents, "edits": edits, "steer": steer,
                   "souls": souls, "max_turns": args.max_turns, "cards": "rich" if args.rich else "base",
                   "baseline": args.baseline, "t4": args.t4, "t5": args.t5, "t6": args.t6}
            if args.t7 or args.t8 or args.t9:
                cfg |= {"t7": args.t7, "t8": args.t8, "t9": args.t9, "state_texts": args.state_texts}
            res = SceneRun(scene, cast, cards, agent, tools, cfg, s, table, opening).run()
            res["seconds"] = time.time() - t0
            # written under another name first, so that a failed write (the disk was full for a
            # few minutes on 2026-10-24) leaves no empty file that a later run takes for done
            with open(path + ".tmp", "w", encoding="utf-8") as f:
                json.dump(res, f, ensure_ascii=False, indent=1)
            os.replace(path + ".tmp", path)
            print(f"{name} seed {s}: {len(res['turns'])} turns, {res['seconds']:.0f} s, state {res['final_state']}",
                  flush=True)


if __name__ == "__main__":
    main()
