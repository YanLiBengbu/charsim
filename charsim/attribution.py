"""Rule-based speaker attribution of quotation groups.

A tag after the quotation (', said the old man', '? he asked') is used only when the
quotation does not end a sentence. An introduction before the quotation must end with a
comma or a colon ('he said to her roughly, and he was rough because ...,'); it may stand in
the previous paragraph, because the edition often sets the quotation as its own paragraph.
The character mention closest to the speech verb is taken. A pronoun goes to an earlier
mention of the same gender in its own paragraph, else to the character of that gender
mentioned most often in the preceding paragraphs, else to the latest speaker of that gender. A
quotation that stands alone in its paragraph continues an A-B alternation. Every label
records its method, which serves as its confidence."""

import re

from .quotes import groups

SAY = ("said|says|say|asked|asks|ask|cried|cry|answered|answer|replied|reply|muttered|mutter|"
       "whispered|whisper|called|call|shouted|shout|screamed|began|repeated|went on|added|"
       "demanded|murmured|exclaimed|faltered|groaned|retorted|returned|sighed|sobbed|spoke|"
       "told|begged|urged|insisted|wailed|roared|laughed|gasped|stammered|pleaded|commanded|"
       "ordered|bawled|grumbled|whimpered|moaned|coaxed|mumbled|thought")
SAY_RE = re.compile(r"\b(?:" + SAY + r")\b")
POST_WINDOW = 90
PRE_WINDOW = 220
NEAR = 45
INTRO_GAP = 160
LOOKBACK_GROUPS = 8
LOOKBACK_PARAS = 6
FOCALIZER = "wang_lung"


def _candidates(text, chapter, cast):
    cands = [(s, e, cid) for s, e, cid in cast.mentions(text, chapter)]
    cands += [(s, e, ("PRON", g)) for s, e, g in cast.pronouns(text)]
    return sorted(cands)


def _closest(text, verb, chapter, cast, after_limit):
    best, dist = None, None
    for s, e, cand in _candidates(text, chapter, cast):
        if s >= verb.end():
            d = s - verb.end()
            if d > after_limit:
                continue
        else:
            d = verb.start() - e
        if d > NEAR:
            continue
        if dist is None or d < dist:
            best, dist = (s, cand), d
    return best


def _post_tag(text, end, quote_text, chapter, cast):
    last = quote_text.rstrip(" -")[-1:]
    if last not in ",?!":
        return None
    post = text[end + 1:end + 1 + POST_WINDOW]
    cut = post.find('"')
    if cut != -1:
        post = post[:cut]
    m = SAY_RE.search(post)
    if not m or m.start() > NEAR:
        return None
    head = post.lstrip(" ,")
    if last in "?!" and head[:1].isupper():
        lead = [c for c in _candidates(post, chapter, cast) if c[0] == len(post) - len(head)]
        if not lead or m.start() - lead[0][1] > 12:
            return None
    hit = _closest(post, m, chapter, cast, after_limit=NEAR)
    return (hit[0] + end + 1, hit[1]) if hit else None


def _pre_tag(pre, chapter, cast):
    """pre is the text that introduces the quotation; returns (offset in pre, candidate)."""
    pre = pre.rstrip()
    if not pre.endswith((",", ":")):
        return None
    verbs = list(SAY_RE.finditer(pre))
    if not verbs:
        return None
    m = verbs[-1]
    if len(pre) - m.end() > INTRO_GAP:
        return None
    return _closest(pre, m, chapter, cast, after_limit=6)


def attribute(chapter, quotes, cast):
    """Return {group: {"speaker": cid or None, "method": str or None}}."""
    num = chapter["chapter"]
    paras = {p["pid"]: p for p in chapter["paragraphs"]}
    order = [p["pid"] for p in chapter["paragraphs"]]
    pos_of = {pid: i for i, pid in enumerate(order)}
    by_para = {}
    for q in quotes:
        by_para.setdefault(q["pid"], []).append(q)
    out, history = {}, []

    def resolve(cand, pid, offset_text):
        if not isinstance(cand, tuple):
            return cand, "tag_name"
        gender = cand[1]
        found = [c for s, e, c in cast.mentions(offset_text, num) if cast.gender(c) == gender]
        if found:
            return found[-1], "tag_pronoun"
        i = pos_of[pid]
        counts, recent = {}, {}
        for rank, x in enumerate(reversed(order[max(0, i - LOOKBACK_PARAS):i])):
            for s, e, c in cast.mentions(paras[x]["text"], num):
                if cast.gender(c) == gender:
                    counts[c] = counts.get(c, 0) + 1
                    recent.setdefault(c, rank)
        if counts:
            return min(counts, key=lambda c: (-counts[c], recent[c])), "tag_pronoun"
        for cid in reversed(history[-LOOKBACK_GROUPS:]):
            if cid and cast.gender(cid) == gender:
                return cid, "tag_pronoun"
        if gender == "m":
            return FOCALIZER, "pronoun_default"
        return None, None

    for g, segs in groups(quotes).items():
        first, last = segs[0], segs[-1]
        label = None
        ptxt = paras[last["pid"]]["text"]
        hit = _post_tag(ptxt, last["end"], last["text"], num, cast)
        if hit:
            cid, method = resolve(hit[1], last["pid"], ptxt[:hit[0]])
            if cid:
                label = {"speaker": cid, "method": method}
        if not label:
            p = paras[first["pid"]]
            same = [q for q in by_para[first["pid"]] if q["end"] < first["start"]]
            seg_start = same[-1]["end"] + 1 if same else 0
            pre = p["text"][max(seg_start, first["start"] - 1 - PRE_WINDOW):first["start"] - 1]
            pid = first["pid"]
            if not pre.strip() and p.get("joins_previous"):
                pid = order[pos_of[first["pid"]] - 1]
                prev = paras[pid]["text"]
                tail = prev.rfind('"') + 1
                pre = prev[max(tail, len(prev) - PRE_WINDOW):]
            hit = _pre_tag(pre, num, cast)
            if hit:
                cid, method = resolve(hit[1], pid, pre[:hit[0]])
                if cid:
                    label = {"speaker": cid, "method": method}
        out[g] = label or {"speaker": None, "method": None}
        history.append(out[g]["speaker"])

    gl = list(groups(quotes).items())
    for k, (g, segs) in enumerate(gl):
        if out[g]["speaker"] or k < 2:
            continue
        p = paras[segs[0]["pid"]]
        narration = p["text"].replace("".join(s["text"] for s in segs if s["pid"] == p["pid"]), "").strip(' "')
        if len(narration) > 3 or p.get("joins_previous"):
            continue
        a, b = out[gl[k - 2][0]]["speaker"], out[gl[k - 1][0]]["speaker"]
        pa, pb, pc = pos_of[gl[k - 2][1][0]["pid"]], pos_of[gl[k - 1][1][-1]["pid"]], pos_of[segs[0]["pid"]]
        if a and b and a != b and pb - pa == 1 and pc - pb == 1:
            out[g] = {"speaker": a, "method": "alternation"}
    return out
