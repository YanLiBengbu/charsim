"""Training samples of the character agents.

The timeline of a chapter merges the attributed quotations and the grounded beats in the
order of the text. A decision point is a beat of a character with a soul: a quotation, a
reported speech, an action or a silence. Its input is what the character knows at that
moment (the card, the T1 and T2 tool results at the chapter cutoff, and the lines of the
chapter so far that the character can perceive) and its target is what the character did.
A character who is present at another character's beat gets the target "wait", so that the
probability of not waiting can serve as the urgency of the character in the simulation."""

import hashlib
import json
import re

from .tools import NarrativeTools

# an "action" whose summary only states what a character is or has ("The uncle is jovial")
STATE = re.compile(r"^[A-Z][\w'-]*(?: [\w'-]+){0,3}? (?:is|are|was|were|has|have) ")
HISTORY = 12
PRESENT_BEFORE, PRESENT_AFTER = 5, 1
QUERY_LINES = 3
INTENDS = ("speak", "act", "silent", "wait")

SYSTEM = """{card}

You take part in a scene of a novel as this character. At every turn you decide what you do next and answer only with JSON in one of these forms:
{{"intend": "speak", "to": [names], "speech": "the words you say"}}
{{"intend": "speak", "to": [names], "gist": "what you say, reported in a short sentence"}}
{{"intend": "act", "to": [names], "action": "what you do, in a short sentence in the present tense"}}
{{"intend": "silent", "to": [names]}} when you are spoken to and say nothing
{{"intend": "wait"}} when you let the others go on"""

USER = """People here: {present}
{tools}
The scene so far:
{lines}

What do you do now?"""


def card_text(cards, cutoff, cid, edit=None):
    """The card of a character at a cutoff; edit = (trait, "up" | "down") replaces or adds one sentence."""
    entry = cards["cards"].get(str(cutoff), {}).get(cid)
    if entry is None:
        return cards["default"].get(cid, "")
    traits = dict(entry["traits"])
    if edit:
        trait, direction = edit
        traits[trait] = entry["edits"][trait][direction]
    return " ".join([entry["base"], *traits.values()])


def _overlap(a, b):
    if a["pid"] != b["pid"]:
        return 0.0
    s, e = max(a["start"], b["start"]), min(a["end"], b["end"])
    return max(0, e - s) / max(1, a["end"] - a["start"])


def timeline(chapter, quotes, beats):
    """Quotations and beats of one chapter in text order, without beats inside a quotation
    and without the second of two beats of one character on the same evidence."""
    items = []
    qspans = []
    for q in quotes:
        seg = q["segments"][0]
        speaker = q["speaker"]
        actor = speaker if isinstance(speaker, str) and speaker not in ("other", "none") else None
        items.append({"kind": "speak", "actor": actor, "chorus": speaker if isinstance(speaker, list) else None,
                      "to": [c for c in q.get("to") or [] if c not in ("other", "none")], "text": q["text"],
                      "pid": seg["pid"], "start": seg["start"], "ref": q["group"]})
        qspans += [{"pid": s["pid"], "start": s["start"], "end": s["end"]} for s in q["segments"]]
    kinds = {"action": "act", "reported_speech": "gist", "silence": "silent", "thought": "thought"}
    kept = []
    for b in sorted(beats, key=lambda b: -b["support"]):
        sp = b["span"]
        if not sp or any(_overlap(sp, q) >= 0.5 for q in qspans):
            continue
        if b["type"] == "action" and STATE.match(b["summary"] or ""):
            continue
        if any(k["character"] == b["character"] and _overlap(sp, k["span"]) >= 0.5 for k in kept):
            continue
        kept.append(b)
    for b in kept:
        items.append({"kind": kinds[b["type"]], "actor": b["character"], "chorus": None, "to": b.get("to") or [],
                      "text": b["summary"], "pid": b["span"]["pid"], "start": b["span"]["start"], "ref": b["id"],
                      "support": b["support"]})
    items.sort(key=lambda it: (it["pid"], it["start"]))
    for it in items:
        it["chapter"] = chapter
    return items


def render(it, viewer, cast):
    """One line of the scene as the viewer perceives it, or None (the thoughts of others)."""
    name = cast.name(it["actor"]) if it["actor"] else "Someone"
    if it["chorus"]:
        name = " and ".join(cast.name(c) for c in it["chorus"])
    to = f" (to {', '.join(cast.name(c) for c in it['to'])})" if it["to"] else ""
    if it["kind"] == "speak":
        return f'{name}{to}: "{it["text"]}"'
    if it["kind"] == "thought":
        return f"(your thought) {it['text']}" if it["actor"] == viewer else None
    tag = {"act": "action", "gist": "says", "silent": "silence"}[it["kind"]]
    return f"[{tag}]{to} {it['text']}"


def present_at(items, t):
    lo, hi = max(0, t - PRESENT_BEFORE), min(len(items), t + PRESENT_AFTER + 1)
    out = set()
    for it in items[lo:hi]:
        if it["actor"]:
            out.add(it["actor"])
        out.update(it["chorus"] or [])
        out.update(it["to"])
    return out


def target(it, cast):
    to = [cast.name(c) for c in it["to"]]
    if it["kind"] == "speak":
        return {"intend": "speak", "to": to, "speech": it["text"]}
    if it["kind"] == "gist":
        return {"intend": "speak", "to": to, "gist": it["text"]}
    if it["kind"] == "act":
        return {"intend": "act", "to": to, "action": it["text"]}
    return {"intend": "silent", "to": to}


def _keep(key, rate):
    h = int(hashlib.sha1(key.encode("utf-8")).hexdigest()[:8], 16)
    return h / 0xFFFFFFFF < rate


def decision_points(items, tools, cast, card, cid, neg_rate):
    """Samples of one character in one chapter timeline."""
    out = []
    for t, it in enumerate(items):
        if it["kind"] == "thought" or it["chorus"]:
            continue
        here = present_at(items, t)
        if it["actor"] == cid:
            tgt = target(it, cast)
        elif cid in here and it["actor"] and _keep(f"{cid}|{it['ref']}", neg_rate):
            tgt = {"intend": "wait"}
        else:
            continue
        lines = [ln for ln in (render(x, cid, cast) for x in items[:t]) if ln][-HISTORY:]
        others = sorted(here - {cid})
        known = tools.describe(cid, others, " ".join(lines[-QUERY_LINES:])) if tools else ""
        user = USER.format(present=", ".join(cast.name(c) for c in [cid, *others]),
                           tools=known + "\n" if known else "",
                           lines="\n".join(lines) if lines else "(the scene begins)")
        out.append({"messages": [{"role": "system", "content": SYSTEM.format(card=card)},
                                 {"role": "user", "content": user},
                                 {"role": "assistant", "content": json.dumps(tgt, ensure_ascii=False)}],
                    "character": cid, "chapter": it["chapter"], "ref": it["ref"], "intend": tgt["intend"]})
    return out


def build_tools(data_dir, cast, cutoff, min_support):
    return NarrativeTools(f"{data_dir}/tools/events.jsonl", f"{data_dir}/tools/relations.jsonl",
                          f"{data_dir}/text/chapters.json", cast, cutoff, min_support=min_support)
