"""Check that a span quoted by an LLM really occurs in the passage it annotated.

An evidence string is grounded if its tokens occur as a contiguous run in the passage, or
if the best window of the same length reaches a token Jaccard similarity of at least
MIN_JACCARD (the faithfulness measure of Bartalesi et al., PeerJ CS 2024)."""

import re

TOKEN = re.compile(r"[A-Za-z0-9]+(?:['\-][A-Za-z0-9]+)*")
MIN_JACCARD = 0.8


def tokens(text):
    return [(m.group(0).lower(), m.start(), m.end()) for m in TOKEN.finditer(text)]


def ground(evidence, passage, ptoks=None):
    """Return {"start", "end", "score", "exact"} in passage offsets, or None."""
    ev = [t for t, _, _ in tokens(evidence or "")]
    if not ev:
        return None
    pt = ptoks if ptoks is not None else tokens(passage)
    words = [t for t, _, _ in pt]
    n = len(ev)
    for i in range(len(words) - n + 1):
        if words[i:i + n] == ev:
            return {"start": pt[i][1], "end": pt[i + n - 1][2], "score": 1.0, "exact": True}
    best, best_i, sev = 0.0, None, set(ev)
    for i in range(max(1, len(words) - n + 1)):
        win = set(words[i:i + n])
        j = len(win & sev) / len(win | sev) if win | sev else 0.0
        if j > best:
            best, best_i = j, i
    if best_i is None or best < MIN_JACCARD:
        return None
    last = min(best_i + n, len(pt)) - 1
    return {"start": pt[best_i][1], "end": pt[last][2], "score": round(best, 3), "exact": False}
