"""Training and evaluation data of the character agents for every test scene.

For a scene in chapter c only chapters 1 to c-1 are used for training:
  data/train/c<c>/world.txt          the text of the novel up to chapter c-1 (story-world adapter)
  data/train/c<c>/soul_<id>.jsonl    decision points of a scene character (soul adapter), split train/val
  data/eval/c<c>/soul_<id>.jsonl     decision points of the held-out chapter c (fidelity evaluation)
A scene character with fewer than --min-positives own beats before chapter c gets no soul
adapter of its own. For such a scene, data/train/c<c>/soul_cast.jsonl pools the decision points
of every character of the novel without a soul adapter, each with its own card, so that the
shared supporting-cast adapter learns the moves and the waiting while the card gives the
identity. With --pseudonyms the same files are written under data/train_pseudo and
data/eval_pseudo with the replacement names."""

import argparse
import collections
import json
import os
import random
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from charsim import pseudonyms  # noqa: E402
from charsim.cast import Cast  # noqa: E402
from charsim.samples import build_tools, card_text, decision_points, timeline  # noqa: E402


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def rename(obj, table):
    if table is None:
        return obj
    if isinstance(obj, str):
        return pseudonyms.apply(obj, table)
    if isinstance(obj, list):
        return [rename(x, table) for x in obj]
    if isinstance(obj, dict):
        return {k: rename(v, table) if k in ("messages", "content") else v for k, v in obj.items()}
    return obj


def write_jsonl(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--cast", default="config/cast.json")
    ap.add_argument("--cards", default="config/cards.json")
    ap.add_argument("--scenes", default="config/scenes.json")
    ap.add_argument("--pseudonym-table", default="config/pseudonyms.json")
    ap.add_argument("--tool-support", type=int, default=1, help="minimum extractor support of a T1/T2 item")
    ap.add_argument("--neg-rate", type=float, default=0.5, help="share of the 'wait' decision points kept")
    ap.add_argument("--min-positives", type=int, default=100)
    ap.add_argument("--max-samples", type=int, default=3000)
    ap.add_argument("--val-frac", type=float, default=0.05)
    ap.add_argument("--pseudonyms", action="store_true")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    cast = Cast(args.cast)
    with open(args.cards, encoding="utf-8") as f:
        cards = json.load(f)
    with open(args.scenes, encoding="utf-8") as f:
        scenes = json.load(f)["scenes"]
    with open(os.path.join(args.data, "text/chapters.json"), encoding="utf-8") as f:
        chapters = {c["chapter"]: c for c in json.load(f)["chapters"]}
    quotes, beats = collections.defaultdict(list), collections.defaultdict(list)
    for q in load_jsonl(os.path.join(args.data, "derived/quotes_final.jsonl")):
        quotes[q["chapter"]].append(q)
    for b in load_jsonl(os.path.join(args.data, "derived/beats_final.jsonl")):
        if b["grounded"] and b["valid"]:
            beats[b["chapter"]].append(b)
    lines = {c: timeline(c, quotes[c], beats[c]) for c in chapters}
    tools = {}

    def tools_at(c):
        if c not in tools:
            tools[c] = build_tools(args.data, cast, c, args.tool_support)
        return tools[c]

    tables = [("", None)]
    if args.pseudonyms:
        tables.append(("_pseudo", pseudonyms.load(args.pseudonym_table)))
    manifest = {"tool_support": args.tool_support, "neg_rate": args.neg_rate, "min_positives": args.min_positives,
                "max_samples": args.max_samples, "scenes": {}}
    for sc in scenes:
        c = sc["chapter"]
        lo, hi = sc["train_chapters"]
        info = manifest["scenes"][sc["id"]] = {"chapter": c, "souls": {}, "card_only": []}
        world = "\n\n\n".join("\n\n".join(p["text"] for p in chapters[k]["paragraphs"]) for k in range(lo, hi + 1))
        for cid in sc["cast"]:
            card = card_text(cards, c, cid)
            train = [s for k in range(lo, hi + 1) for s in decision_points(lines[k], tools_at(k), cast, card, cid,
                                                                          args.neg_rate)]
            evals = decision_points(lines[c], tools_at(c), cast, card, cid, 1.0)
            pos = sum(s["intend"] != "wait" for s in train)
            if pos < args.min_positives:
                info["card_only"].append({"character": cid, "positives": pos})
                train = []
            rng = random.Random(f"{sc['id']}|{cid}")
            if len(train) > args.max_samples:
                train = sorted(rng.sample(train, args.max_samples), key=lambda s: (s["chapter"], s["ref"]))
            for s in train:
                s["split"] = "val" if rng.random() < args.val_frac else "train"
            if train:
                info["souls"][cid] = {"samples": len(train), "positives": sum(s["intend"] != "wait" for s in train),
                                      "by_intend": dict(collections.Counter(s["intend"] for s in train)),
                                      "val": sum(s["split"] == "val" for s in train),
                                      "eval_samples": len(evals)}
            for suffix, table in tables:
                if train:
                    write_jsonl(os.path.join(args.data, f"train{suffix}/c{c}/soul_{cid}.jsonl"),
                                [rename(s, table) for s in train])
                write_jsonl(os.path.join(args.data, f"eval{suffix}/c{c}/soul_{cid}.jsonl"),
                            [rename(s, table) for s in evals])
        if info["card_only"]:
            pooled = []
            for cid in cast.characters:
                if cid in info["souls"]:
                    continue
                # the description of the cast without what happens later ("; dies in ...", ", later ...")
                desc = re.split(r";|, later", cast.characters[cid]["description"])[0]
                card = card_text(cards, c, cid) or f"You are {cast.name(cid)}, {desc}."
                pooled += [s for k in range(lo, hi + 1)
                           for s in decision_points(lines[k], tools_at(k), cast, card, cid, args.neg_rate)]
            rng = random.Random(f"{sc['id']}|cast")
            if len(pooled) > args.max_samples:
                pooled = sorted(rng.sample(pooled, args.max_samples), key=lambda s: (s["chapter"], s["ref"]))
            for s in pooled:
                s["split"] = "val" if rng.random() < args.val_frac else "train"
            info["souls"]["cast"] = {"samples": len(pooled), "positives": sum(s["intend"] != "wait" for s in pooled),
                                     "by_intend": dict(collections.Counter(s["intend"] for s in pooled)),
                                     "characters": dict(collections.Counter(s["character"] for s in pooled)),
                                     "val": sum(s["split"] == "val" for s in pooled)}
            for suffix, table in tables:
                write_jsonl(os.path.join(args.data, f"train{suffix}/c{c}/soul_cast.jsonl"),
                            [rename(s, table) for s in pooled])
        for suffix, table in tables:
            path = os.path.join(args.data, f"train{suffix}/c{c}/world.txt")
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                f.write(rename(world, table))
        info["world_words"] = len(world.split())
    with open(os.path.join(args.data, "train/manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=1)
    print(json.dumps(manifest, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
