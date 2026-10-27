"""Cast of the novel with aliases that are valid only in a range of chapters."""

import json
import re

PRONOUN = re.compile(r"\b(he|she|He|She)\b")


class Cast:
    def __init__(self, path):
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        self.characters = {c["id"]: c for c in data["characters"]}
        self._cache = {}

    def name(self, cid):
        return self.characters[cid]["name"] if cid in self.characters else cid

    def gender(self, cid):
        return self.characters.get(cid, {}).get("gender")

    def names(self):
        out = []
        for c in self.characters.values():
            out += [a for a, lo, hi in c["aliases"] if a[0].isupper()]
        return sorted(set(out), key=len, reverse=True)

    def present_in(self, chapter):
        return [cid for cid, c in self.characters.items()
                if any(lo <= chapter <= hi for _, lo, hi in c["aliases"])]

    def _patterns(self, chapter):
        if chapter not in self._cache:
            pats = []
            for cid, c in self.characters.items():
                for alias, lo, hi in c["aliases"]:
                    if lo <= chapter <= hi:
                        pats.append((alias, cid))
            pats.sort(key=lambda x: len(x[0]), reverse=True)
            self._cache[chapter] = [(re.compile(r"(?<![\w-])" + re.escape(a) + r"(?![\w-])"), cid) for a, cid in pats]
        return self._cache[chapter]

    def mentions(self, text, chapter):
        """Non-overlapping alias mentions [(start, end, cid)], longest alias first."""
        taken, out = [], []
        for pat, cid in self._patterns(chapter):
            for m in pat.finditer(text):
                if any(m.start() < e and s < m.end() for s, e in taken):
                    continue
                taken.append((m.start(), m.end()))
                out.append((m.start(), m.end(), cid))
        return sorted(out)

    def pronouns(self, text):
        return [(m.start(), m.end(), "m" if m.group(1).lower() == "he" else "f") for m in PRONOUN.finditer(text)]
