"""Transcripts in which every line names its character (for the judges and the interface),
and the original scene of the novel in the same form."""

import json
import os
import random

from .samples import timeline


def judge_line(it, cast):
    name = " and ".join(cast.name(c) for c in it["chorus"]) if it.get("chorus") else \
        (cast.name(it["actor"]) if it["actor"] else "Someone")
    to = f" (to {', '.join(cast.name(c) for c in it['to'])})" if it["to"] else ""
    if it["kind"] == "speak":
        return f'{name}{to}: "{it["text"]}"'
    if it["kind"] == "thought":
        return None
    tag = {"act": "action", "gist": "says", "silent": "silence"}[it["kind"]]
    return f"{name} [{tag}]{to}: {it['text']}"


def chapter_items(data, chapter):
    """Paragraphs and the timeline of quotations and beats of a chapter."""
    with open(os.path.join(data, "text/chapters.json"), encoding="utf-8") as f:
        paras = next(ch for ch in json.load(f)["chapters"] if ch["chapter"] == chapter)["paragraphs"]
    with open(os.path.join(data, "derived/quotes_final.jsonl"), encoding="utf-8") as f:
        quotes = [q for q in map(json.loads, f) if q["chapter"] == chapter]
    with open(os.path.join(data, "derived/beats_final.jsonl"), encoding="utf-8") as f:
        beats = [b for b in map(json.loads, f) if b["chapter"] == chapter and b["grounded"] and b["valid"]]
    return paras, timeline(chapter, quotes, beats)


def scene_items(data, scene):
    """(opening, original): the timeline before the start anchor and the original scene."""
    paras, items = chapter_items(data, scene["chapter"])
    a = next(p["pid"] for p in paras if scene["start_anchor"] in p["text"])
    e = next(p["pid"] for p in paras if scene["end_anchor"] in p["text"])
    return [it for it in items if it["pid"] < a], [it for it in items if a <= it["pid"] <= e]


def original_lines(data, scene, cast):
    return [ln for ln in (judge_line(it, cast) for it in scene_items(data, scene)[1]) if ln]


def perturbed_lines(data, scene, cast):
    """Two broken versions of the original scene, the controls of the judges' ratings: the
    lines in a random order ("shuffled", coherence should drop), and the first two characters
    of the scene trading their lines ("swapped", character should drop; who is addressed is
    traded too, the words are not changed)."""
    items = [it for it in scene_items(data, scene)[1] if it["kind"] != "thought"]
    a, b = scene["cast"][:2]
    trade = {a: b, b: a}
    swapped = [it | {"actor": trade.get(it["actor"], it["actor"]),
                     "to": [trade.get(c, c) for c in it["to"] or []],
                     "chorus": [trade.get(c, c) for c in it.get("chorus") or []]} for it in items]
    shuffled = list(items)
    random.Random(scene["id"]).shuffle(shuffled)
    return {k: [ln for ln in (judge_line(it, cast) for it in v) if ln]
            for k, v in (("shuffled", shuffled), ("swapped", swapped))}
