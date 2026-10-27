"""Persona vectors of the edited traits of a scene (activation steering).

For every trait edit of the scene the decision points of the character (training data before
the scene) are encoded twice, once with the up card and once with the down card, with the
adapters that the character plays with in the simulation. The vector is the difference of the
mean hidden states of the last prompt token at the middle decoder layer, so that steering
with coefficient +1 or -1 moves that layer by as much as the card edit does on average.
Output: results/steer/<run>/<scene>/<character>__<trait>.pt"""

import argparse
import glob
import json
import os
import random
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from charsim import pseudonyms  # noqa: E402
from charsim.agent import AgentModel  # noqa: E402
from charsim.samples import SYSTEM, card_text  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--scene", required=True)
    ap.add_argument("--prompts", type=int, default=48)
    ap.add_argument("--out", default="results/steer")
    args = ap.parse_args()

    with open("config/scenes.json", encoding="utf-8") as f:
        scene = next(s for s in json.load(f)["scenes"] if s["id"] == args.scene)
    with open("config/cards.json", encoding="utf-8") as f:
        cards = json.load(f)
    c = scene["chapter"]
    adir = f"adapters/{args.run}/c{c}"
    souls = sorted(os.path.basename(p)[5:] for p in glob.glob(f"{adir}/soul_*")
                   if os.path.isfile(f"{p}/train_log.json"))
    agent = AgentModel(args.base, {"world": f"{adir}/world"} | {f"soul_{s}": f"{adir}/soul_{s}" for s in souls})
    table = pseudonyms.load("config/pseudonyms.json") if args.run.endswith("_pseudo") else None
    layer = len(agent.layers()) // 2
    suffix = "_pseudo" if table else ""
    os.makedirs(os.path.join(args.out, args.run, args.scene), exist_ok=True)

    for edit in scene["trait_edits"]:
        cid, trait = edit["character"], edit["trait"]
        own = f"data/train{suffix}/c{c}/soul_{cid}.jsonl"
        path = own if os.path.exists(own) else f"data/train{suffix}/c{c}/soul_cast.jsonl"
        with open(path, encoding="utf-8") as f:
            rows = [r for r in map(json.loads, f) if r["character"] == cid]
        rows = random.Random(f"{cid}|{trait}").sample(rows, min(args.prompts, len(rows)))
        agent.activate(["world", f"soul_{cid}"] if cid in souls else ["world", "soul_cast"] if "cast" in souls
                       else ["world"])
        means = {}
        for direction in ("up", "down"):
            system = SYSTEM.format(card=card_text(cards, c, cid, (trait, direction)))
            system = pseudonyms.apply(system, table) if table else system
            hs = [agent.last_hidden([{"role": "system", "content": system}, r["messages"][1]], layer) for r in rows]
            means[direction] = torch.stack(hs).mean(0)
        base_norm = float(torch.stack(hs).norm(dim=-1).mean())
        vec = means["up"] - means["down"]
        out = os.path.join(args.out, args.run, args.scene, f"{cid}__{trait.replace(' ', '_')}.pt")
        torch.save({"vector": vec, "layer": layer, "prompts": len(rows), "character": cid, "trait": trait,
                    "vector_norm": float(vec.norm()), "hidden_norm": base_norm, "adapters": agent.active}, out)
        print(f"{cid} {trait}: layer {layer}, |v| {vec.norm():.2f}, |h| {base_norm:.2f}, {len(rows)} prompts", flush=True)


if __name__ == "__main__":
    main()
