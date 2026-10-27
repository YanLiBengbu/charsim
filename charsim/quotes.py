"""Quotation segmentation with exact offsets.

A quotation that runs over several paragraphs has an opening mark at the start of every
paragraph and a closing mark only at its end; its segments share one group id."""


def find_quotes(chapter):
    out, open_group, n = [], None, 0
    for p in chapter["paragraphs"]:
        text = p["text"]
        if open_group is not None and not text.startswith('"'):
            open_group = None
        inside, start = False, 0
        for i, ch in enumerate(text):
            if ch != '"':
                continue
            if not inside:
                inside, start = True, i + 1
                continue
            inside = False
            n += 1
            qid = f"{chapter['chapter']:02d}.q{n:03d}"
            group = open_group if (open_group is not None and start == 1) else qid
            open_group = None
            out.append({"qid": qid, "group": group, "pid": p["pid"], "start": start, "end": i,
                        "text": text[start:i], "open_end": False})
        if inside:
            n += 1
            qid = f"{chapter['chapter']:02d}.q{n:03d}"
            group = open_group if (open_group is not None and start == 1) else qid
            out.append({"qid": qid, "group": group, "pid": p["pid"], "start": start, "end": len(text),
                        "text": text[start:], "open_end": True})
            open_group = group
        else:
            open_group = None
    return out


def groups(quotes):
    """Quotation groups in order: {group: [segments]}."""
    out = {}
    for q in quotes:
        out.setdefault(q["group"], []).append(q)
    return out
