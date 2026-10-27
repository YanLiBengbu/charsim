"""The manually attributed quotations of the chapter 26 excerpt of the previous study."""

import json
import os
import re

GOLD = os.path.join("English_lit", "code_revised", "texts", "excerpts.json")
GOLD_NAMES = {"O-lan": "o_lan", "Wang Lung": "wang_lung", "the eldest son": "eldest_son",
              "the maid": "eldest_sons_wife"}


def norm(s):
    return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()


def load(path=GOLD):
    with open(path, encoding="utf-8") as f:
        return next(e for e in json.load(f) if e["id"] == "good_earth")["gold_quotes"]


def match(rows, gold):
    """[(gold quotation, gold speaker id, quotation group of chapter 26 or None)]"""
    ch26 = [r for r in rows if r["chapter"] == 26]
    out = []
    for gq in gold:
        hit = next((r for r in ch26 if norm(gq["text"])[:40] in norm(r["text"])
                    or norm(r["text"])[:40] in norm(gq["text"])), None)
        out.append((gq, GOLD_NAMES.get(gq["speaker"]), hit))
    return out
