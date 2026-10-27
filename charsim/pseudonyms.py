"""Whole-word replacement of character and place names for the memorization control."""

import json
import re


def load(path):
    with open(path, encoding="utf-8") as f:
        names = json.load(f)["names"]
    pairs = sorted(names.items(), key=lambda kv: len(kv[0]), reverse=True)
    pattern = re.compile("|".join(r"(?<![\w-])" + re.escape(k) + r"(?![\w-])" for k, _ in pairs))
    return dict(pairs), pattern


def apply(text, table):
    names, pattern = table
    return pattern.sub(lambda m: names[m.group(0)], text)
