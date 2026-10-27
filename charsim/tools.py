"""T1 relation graph and T2 event memory, queried at a chapter cutoff from one character's view.

Only items of chapters before the cutoff are visible, so a character never learns its own
future. An event is remembered by its witnesses. A relationship statement is known to its
two characters and to the characters mentioned in the paragraph that supports it. Events
are retrieved with BM25 over their summaries, objects and participant names."""

import collections
import json
import math
import re

WORD = re.compile(r"[a-z]+")
STOP = set("a an the and or but of to in on at for with from by as is was were be been he she it they "
           "his her their him them this that these those who whom which what when where how not no "
           "you your i me my we our".split())
K1, B = 1.2, 0.75


def _terms(text):
    return [w for w in WORD.findall(text.lower()) if w not in STOP and len(w) > 1]


def _load(path, cutoff, min_support):
    with open(path, encoding="utf-8") as f:
        items = [json.loads(line) for line in f]
    return [it for it in items if it["chapter"] < cutoff and it.get("support", 2) >= min_support]


class NarrativeTools:
    def __init__(self, events_path, relations_path, chapters_path, cast, cutoff, min_support=2):
        self.cast = cast
        self.cutoff = cutoff
        self.events = _load(events_path, cutoff, min_support)
        self.relations = _load(relations_path, cutoff, min_support)
        with open(chapters_path, encoding="utf-8") as f:
            paras = {p["pid"]: p["text"] for c in json.load(f)["chapters"] for p in c["paragraphs"]}
        for r in self.relations:
            pid = (r.get("span") or {}).get("pid")
            seen = {cid for _, _, cid in cast.mentions(paras.get(pid, ""), r["chapter"])}
            r["known_by"] = sorted(seen | {r["source"], r["target"]})
        self._docs = []
        for e in self.events:
            names = " ".join(cast.name(c) for c in e["participants"])
            self._docs.append(_terms(" ".join([e.get("summary") or "", " ".join(e["objects"]), names])))
        self._df = collections.Counter(t for d in self._docs for t in set(d))
        self._avg = sum(map(len, self._docs)) / max(1, len(self._docs))

    def _bm25(self, q, i):
        d = self._docs[i]
        tf = collections.Counter(d)
        n = len(self._docs)
        s = 0.0
        for t in q:
            if t not in tf:
                continue
            idf = math.log(1 + (n - self._df[t] + 0.5) / (self._df[t] + 0.5))
            s += idf * tf[t] * (K1 + 1) / (tf[t] + K1 * (1 - B + B * len(d) / self._avg))
        return s

    def recall(self, viewer, query, k=3):
        q = _terms(query)
        scored = []
        for i, e in enumerate(self.events):
            if viewer not in e["witnesses"]:
                continue
            s = self._bm25(q, i)
            if s > 0:
                scored.append((s, e["chapter"], e))
        scored.sort(key=lambda x: (-x[0], -x[1]))
        return [{"id": e["id"], "chapter": e["chapter"], "summary": e["summary"], "participants": e["participants"]}
                for _, _, e in scored[:k]]

    def relation(self, viewer, other):
        latest = {}
        for r in self.relations:
            if {r["source"], r["target"]} != {viewer, other} or viewer not in r["known_by"]:
                continue
            k = (r["source"], r["target"], r["relation"])
            if k not in latest or r["chapter"] >= latest[k]["chapter"]:
                latest[k] = r
        return sorted(({"source": s, "target": t, "relation": rel, "polarity": r["polarity"],
                        "chapter": r["chapter"], "description": r.get("description")}
                       for (s, t, rel), r in latest.items()), key=lambda x: -x["chapter"])

    def describe(self, viewer, others, query, k=3):
        lines = []
        for o in others:
            rels = self.relation(viewer, o)[:4]
            if rels:
                parts = [f"{x['relation']} ({x['description']})" if x["description"] else x["relation"] for x in rels]
                lines.append(f"- {self.cast.name(o)}: " + "; ".join(parts))
        mem = self.recall(viewer, query, k)
        out = []
        if lines:
            out.append("What you know about the people here:\n" + "\n".join(lines))
        if mem:
            out.append("What you remember:\n" + "\n".join(f"- {m['summary']}" for m in mem))
        return "\n".join(out)
