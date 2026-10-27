"""Character fidelity on the held-out chapter of a scene (decision points of data/eval/c<c>).

For every scene character and every configuration (prompted: the base model with the card
and the tools; world: plus the story-world adapter; soul: plus the soul adapter of the
character) the script reports the accuracy and macro F1 of the most probable intention, the
AUC of the urgency (probability of not waiting) for separating the character's own beats from
the beats of others, and the mean negative log-likelihood per token of the original answer."""

import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from charsim.agent import INTENDS, AgentModel  # noqa: E402


def auc(scores, labels):
    pos = [s for s, y in zip(scores, labels) if y]
    neg = [s for s, y in zip(scores, labels) if not y]
    if not pos or not neg:
        return None
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def macro_f1(gold, pred):
    f = []
    for c in INTENDS:
        tp = sum(g == c and p == c for g, p in zip(gold, pred))
        fp = sum(g != c and p == c for g, p in zip(gold, pred))
        fn = sum(g == c and p != c for g, p in zip(gold, pred))
        if tp + fp + fn:
            f.append(2 * tp / (2 * tp + fp + fn))
    return sum(f) / len(f) if f else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--adapters", required=True, help="adapters/<base>/c<c>")
    ap.add_argument("--eval", required=True, help="data/eval/c<c>")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    found = {"world": os.path.join(args.adapters, "world")}
    found.update({os.path.basename(p): p for p in glob.glob(os.path.join(args.adapters, "soul_*"))
                  if os.path.isfile(os.path.join(p, "adapter_model.safetensors"))})
    agent = AgentModel(args.base, found)
    os.makedirs(args.out, exist_ok=True)
    summary = {}
    for path in sorted(glob.glob(os.path.join(args.eval, "soul_*.jsonl"))):
        cid = os.path.basename(path)[5:-6]
        with open(path, encoding="utf-8") as f:
            rows = [json.loads(line) for line in f]
        configs = {"prompted": [], "world": ["world"]}
        if f"soul_{cid}" in found:
            configs["soul"] = ["world", f"soul_{cid}"]
        for name, active in configs.items():
            agent.activate(active)
            recs = []
            for r in rows:
                msgs, answer = r["messages"][:2], r["messages"][2]["content"]
                probs = agent.intend_probs(msgs)
                nll, n_tok = agent.answer_nll(msgs, answer) if r["intend"] != "wait" else (None, 0)
                recs.append({"ref": r["ref"], "gold": r["intend"], "probs": probs, "nll": nll, "tokens": n_tok})
            gold = [x["gold"] for x in recs]
            pred = [max(x["probs"], key=x["probs"].get) for x in recs]
            own = [x for x in recs if x["nll"] is not None]
            summary.setdefault(cid, {})[name] = {
                "samples": len(recs), "accuracy": sum(g == p for g, p in zip(gold, pred)) / len(recs),
                "macro_f1": macro_f1(gold, pred),
                "urgency_auc": auc([1 - x["probs"]["wait"] for x in recs], [g != "wait" for g in gold]),
                "nll": sum(x["nll"] * x["tokens"] for x in own) / max(1, sum(x["tokens"] for x in own)),
                "nll_speech": (lambda s: sum(x["nll"] * x["tokens"] for x in s) / max(1, sum(x["tokens"] for x in s)))(
                    [x for x in own if x["gold"] == "speak"])}
            with open(os.path.join(args.out, f"{cid}__{name}.jsonl"), "w", encoding="utf-8") as f:
                for x in recs:
                    f.write(json.dumps(x) + "\n")
            print(cid, name, json.dumps(summary[cid][name]), flush=True)
    with open(os.path.join(args.out, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=1)


if __name__ == "__main__":
    main()
