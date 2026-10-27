"""Local LLM judges of the simulated scenes and of the calibration controls (resumable).

Every judge reads a transcript in which each line names its character, and answers with the
pre-registered ending class, the truth of every outcome proposition, and three ratings of the
interaction from 1 to 5 with a description of the levels 1, 3 and 5. The ending classes are
shown in an order shuffled per transcript, so that a preference for the first option cannot
favour one class. Every answer is a flat JSON object (one key per statement): with a list of
truth values, phi4 mixed the statements into the list in 304 of 488 answers of the first run
(kept in results/judge_v1). The controls of the ending class are the original scene of the
novel (class A) and the written endings of the other classes (config/control_endings.json);
the controls of the ratings are the original scene with its lines shuffled and with the lines
of its first two characters swapped (charsim.transcript.perturbed_lines). Output:
results/judge/<judge>.jsonl."""

import argparse
import glob
import json
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from charsim import pseudonyms  # noqa: E402
from charsim.cast import Cast  # noqa: E402
from charsim.jsonfix import parse_json  # noqa: E402
from charsim.llm import OllamaClient  # noqa: E402
from charsim.transcript import judge_line, original_lines, perturbed_lines  # noqa: E402

LETTERS = "ABCD"
RATINGS = ("character", "coherence", "plausibility")

PROMPT = """You are judging a scene written for a novel. The characters of the scene are {names}.

The scene:
{lines}

1. Which of these endings describes how the scene ends? Choose exactly one.
{classes}
2. Is each statement true of the scene?
{statements}
3. Rate the scene on three scales from 1 to 5:
"character": do the characters speak and act as these characters would? 1 = they act against their nature, 3 = mostly, with some lapses, 5 = every line fits the character.
"coherence": does every move follow from what came before? 1 = the moves are unconnected or out of order, 3 = mostly, with some jumps, 5 = every move follows.
"plausibility": could the scene stand in the novel? 1 = not at all, 3 = after some changes, 5 = as it is.

Return only a JSON object of the form {schema}."""


def prompt_for(key, sc, lines, cast):
    """(prompt, order of the ending classes) of a transcript."""
    order = list(LETTERS)
    random.Random(key).shuffle(order)
    n = len(sc["propositions"])
    schema = ", ".join(['"ending": <number of the ending>'] + [f'"S{i + 1}": <true or false>' for i in range(n)]
                       + [f'"{r}": <1-5>' for r in RATINGS])
    prompt = PROMPT.format(
        names=", ".join(cast.name(c) for c in sc["cast"]), lines="\n".join(lines),
        classes="\n".join(f"Ending {i + 1}: {sc['ending_classes'][x]}" for i, x in enumerate(order)),
        statements="\n".join(f"S{i + 1}: {p}" for i, p in enumerate(sc["propositions"])),
        schema="{" + schema + "}")
    return prompt, order


def truth(v):
    if isinstance(v, bool):
        return v
    if isinstance(v, str) and v.strip().lower() in ("true", "false"):
        return v.strip().lower() == "true"
    raise ValueError(f"not a truth value: {v!r}")


def parse(text, sc, order):
    """The fields of a valid answer; raises ValueError, KeyError or TypeError otherwise."""
    obj, _ = parse_json(text)
    k = int(obj["ending"]) - 1
    props = [truth(obj[f"S{i + 1}"]) for i in range(len(sc["propositions"]))]
    scores = {r: int(obj[r]) for r in RATINGS}
    if not 0 <= k < 4 or not all(1 <= v <= 5 for v in scores.values()):
        raise ValueError("out of range")
    return {"ending": order[k], "propositions": props, **scores}


def tasks(args, scenes, cast, table):
    """(key, scene, lines, pseudonym table) of every transcript to judge."""
    out = []
    with open("config/control_endings.json", encoding="utf-8") as f:
        controls = json.load(f)
    for sc in scenes.values():
        out.append((f"control/{sc['id']}/A", sc, original_lines(args.data, sc, cast), None))
        out += [(f"control/{sc['id']}/{k}", sc, v, None) for k, v in controls[sc["id"]].items()]
        out += [(f"control/{sc['id']}/{k}", sc, v, None) for k, v in perturbed_lines(args.data, sc, cast).items()]
    for pattern in args.inputs:
        for path in sorted(glob.glob(os.path.join(pattern, "seed*.json"))):
            try:
                with open(path, encoding="utf-8") as f:
                    res = json.load(f)
            except json.JSONDecodeError:
                print("not a session, skipped:", path, flush=True)
                continue
            run = os.path.relpath(path, args.sim_root).split(os.sep)[0]
            tab = table if run.endswith("_pseudo") else None
            lines = [ln for ln in (judge_line(t["item"], cast) for t in res["turns"]) if ln]
            out.append((os.path.relpath(path, args.sim_root).replace(os.sep, "/"), scenes[res["scene"]], lines, tab))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--judges", nargs="+", default=["phi4:14b", "gemma3:12b", "mistral-nemo:12b"])
    ap.add_argument("--inputs", nargs="*", default=[], help="condition directories of results/sim")
    ap.add_argument("--sim-root", default="results/sim")
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="results/judge")
    ap.add_argument("--num-gpu", type=int, default=None, help="layers on the GPU (0 = run the judge on the CPU)")
    ap.add_argument("--base-url", default=os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434"))
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    cast = Cast("config/cast.json")
    with open("config/scenes.json", encoding="utf-8") as f:
        scenes = {s["id"]: s for s in json.load(f)["scenes"]}
    table = pseudonyms.load("config/pseudonyms.json")
    todo = tasks(args, scenes, cast, table)
    os.makedirs(args.out, exist_ok=True)
    for judge in args.judges:
        path = os.path.join(args.out, f"{judge.replace(':', '_')}.jsonl")
        done = set()
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                done = {json.loads(line)["key"] for line in f}
        llm = OllamaClient(judge, base_url=args.base_url, temperature=0.0, max_tokens=300,
                           cache_path=os.path.join(args.out, f"cache__{judge.replace(':', '_')}.jsonl"),
                           options=None if args.num_gpu is None else {"num_gpu": args.num_gpu})
        with open(path, "a", encoding="utf-8") as f:
            for key, sc, lines, tab in todo:
                if key in done:
                    continue
                prompt, order = prompt_for(key, sc, lines, cast)
                if tab:
                    prompt = pseudonyms.apply(prompt, tab)
                rec = {"key": key, "scene": sc["id"], "judge": judge, "ok": False}
                try:
                    # JSON-constrained decoding can loop on one token (Ollama aborts with a 500);
                    # the answer is then decoded without the constraint and repaired when parsed
                    try:
                        text = llm.generate(prompt, seed=0)
                    except RuntimeError:
                        text = llm.generate(prompt, seed=0, json_mode=False)
                    rec |= parse(text, sc, order) | {"ok": True}
                except (ValueError, KeyError, TypeError, RuntimeError) as e:
                    rec["error"] = str(e)[:200]
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                f.flush()
        print(judge, "done", flush=True)


if __name__ == "__main__":
    main()
