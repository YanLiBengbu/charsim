"""LLM annotation jobs over the novel and parsing of their outputs.

Every chapter is cut into windows of consecutive paragraphs (at most CORE_WORDS words, cut
preferably at section openings). Attribution windows mark their quotations with [[qN]] and
show CONTEXT_PARAS earlier paragraphs as unmarked context. Beat, event and relation items
must quote their evidence from the window; grounding.ground() checks it."""

from .grounding import ground, tokens
from .prompts import ATTRIBUTE, BEATS, EVENTS, RELATIONS, RELATIONS_PROMPT, cast_lines

CORE_WORDS = 600
CONTEXT_PARAS = 2
TEMPLATES = {"beats": BEATS, "events": EVENTS, "relations": RELATIONS_PROMPT}
BEAT_TYPES = {"action", "reported_speech", "silence", "thought"}
RELATION_TYPES = {r.strip() for r in RELATIONS.split(",")}
SPECIAL = {"other", "none"}


def windows(chapter, max_words=CORE_WORDS):
    out, cur, n = [], [], 0
    for p in chapter["paragraphs"]:
        w = len(p["text"].split())
        if cur and (n + w > max_words or (p["section_start"] and n > max_words // 3)):
            out.append(cur)
            cur, n = [], 0
        cur.append(p)
        n += w
    if cur:
        out.append(cur)
    return out


def _join(paras):
    text, spans, pos = [], [], 0
    for p in paras:
        spans.append((p["pid"], pos, pos + len(p["text"])))
        text.append(p["text"])
        pos += len(p["text"]) + 2
    return "\n\n".join(text), spans


def attribution_jobs(chapter, rows, cast):
    paras = chapter["paragraphs"]
    idx = {p["pid"]: i for i, p in enumerate(paras)}
    present = cast.present_in(chapter["chapter"])
    for k, win in enumerate(windows(chapter)):
        core = {p["pid"] for p in win}
        marked = [r for r in rows if r["segments"][0]["pid"] in core]
        if not marked:
            continue
        local = {f"q{i + 1}": r["group"] for i, r in enumerate(marked)}
        inserts = {}
        for lid, r in zip(local, marked):
            seg = r["segments"][0]
            inserts.setdefault(seg["pid"], []).append((seg["start"] - 1, lid))
        rendered = []
        for p in win:
            t = p["text"]
            for pos, lid in sorted(inserts.get(p["pid"], []), reverse=True):
                t = t[:pos] + f"[[{lid}]] " + t[pos:]
            rendered.append(t)
        start = idx[win[0]["pid"]]
        context = [p["text"] for p in paras[max(0, start - CONTEXT_PARAS):start]]
        passage = "\n\n".join(context + rendered)
        prompt = ATTRIBUTE.format(chapter=chapter["chapter"], cast=cast_lines(cast, present, chapter["chapter"]),
                                  ids=", ".join(local), passage=passage)
        yield {"window": f"{chapter['chapter']:02d}.w{k + 1:02d}", "chapter": chapter["chapter"],
               "pids": [p["pid"] for p in win], "prompt": prompt, "map": local, "valid": present}


def item_jobs(chapter, task, cast):
    present = cast.present_in(chapter["chapter"])
    for k, win in enumerate(windows(chapter)):
        passage, spans = _join(win)
        prompt = TEMPLATES[task].format(chapter=chapter["chapter"], relations=RELATIONS,
                                        cast=cast_lines(cast, present, chapter["chapter"]), passage=passage)
        yield {"window": f"{chapter['chapter']:02d}.w{k + 1:02d}", "chapter": chapter["chapter"],
               "pids": [p["pid"] for p in win], "prompt": prompt, "passage": passage, "spans": spans,
               "valid": present}


def _ids(value, valid, allow_special=False):
    vals = value if isinstance(value, list) else ([value] if value not in (None, "") else [])
    out, ok = [], True
    for v in vals:
        v = str(v).strip().lower()
        if v in valid or (allow_special and v in SPECIAL):
            out.append(v)
        else:
            ok = False
    return out, ok


def parse_attribution(obj, job):
    valid = set(job["valid"])
    out = {}
    for lid, group in job["map"].items():
        v = obj.get(lid) if isinstance(obj, dict) else None
        if not isinstance(v, dict):
            out[group] = {"speaker": None, "to": [], "ok": False}
            continue
        sp, ok1 = _ids(v.get("speaker"), valid, allow_special=True)
        to, ok2 = _ids(v.get("to"), valid)
        speaker = sp[0] if len(sp) == 1 else (sorted(sp) if sp else None)
        out[group] = {"speaker": speaker, "to": sorted(set(to)), "ok": ok1 and bool(sp)}
    return out


def _locate(g, spans):
    for pid, s, e in spans:
        if s <= g["start"] < e:
            return {"pid": pid, "start": g["start"] - s, "end": min(g["end"], e) - s}
    return None


def parse_items(obj, job, task):
    key = {"beats": "beats", "events": "events", "relations": "relations"}[task]
    items = obj.get(key) if isinstance(obj, dict) else None
    if not isinstance(items, list):
        return [], False
    valid = set(job["valid"])
    ptoks = tokens(job["passage"])
    out = []
    for i, it in enumerate(items):
        if not isinstance(it, dict):
            continue
        g = ground(it.get("evidence"), job["passage"], ptoks)
        rec = {"n": i, "evidence": it.get("evidence"), "grounded": bool(g),
               "ground_score": g["score"] if g else 0.0, "span": _locate(g, job["spans"]) if g else None}
        if task == "beats":
            ch, ok = _ids(it.get("character"), valid)
            to, ok2 = _ids(it.get("to"), valid)
            rec.update({"type": str(it.get("type", "")).strip().lower(), "character": ch[0] if ch else None,
                        "to": to, "summary": it.get("summary")})
            rec["valid"] = ok and len(ch) == 1 and rec["type"] in BEAT_TYPES
        elif task == "events":
            parts, ok = _ids(it.get("participants"), valid)
            wit, ok2 = _ids(it.get("witnesses"), valid)
            causes = [c for c in it.get("causes") or [] if isinstance(c, int) and 0 <= c < i]
            rec.update({"summary": it.get("summary"), "participants": parts, "witnesses": sorted(set(wit) | set(parts)),
                        "place": it.get("place") or "", "time": it.get("time") or "",
                        "objects": [str(o) for o in it.get("objects") or []], "causes": causes})
            rec["valid"] = bool(parts)
        else:
            src, ok = _ids(it.get("source"), valid)
            tgt, ok2 = _ids(it.get("target"), valid)
            rel = str(it.get("relation", "")).strip().lower()
            try:
                pol = max(-2, min(2, int(it.get("polarity", 0))))
            except (TypeError, ValueError):
                pol = 0
            rec.update({"source": src[0] if src else None, "target": tgt[0] if tgt else None, "relation": rel,
                        "polarity": pol, "description": it.get("description")})
            rec["valid"] = ok and ok2 and len(src) == 1 and len(tgt) == 1 and src != tgt and rel in RELATION_TYPES
        out.append(rec)
    return out, True
