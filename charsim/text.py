"""Chapters and paragraphs of The Good Earth from the PDF edition.

Paragraph-initial lines of the edition are indented (x0 of about 114 pt) and continuation
lines are not (x0 of about 90 pt), which gives the paragraph boundaries even across pages.
Section openings are set in capitals ("BUT ONE DAY he saw ...") and are kept as a flag.
The edition often sets a quotation as its own paragraph after an introduction that ends
with a comma ("he said to her roughly, ..."); such paragraphs are flagged joins_previous."""

import hashlib
import re

import pymupdf

INDENT_X = 105.0
HEADING_SIZE = 16.0
N_CHAPTERS = 34
END_OF_NOVEL = "looked at each other and smiled."
MARKERS = re.compile(r"\[(?:pict\d+|memoirs|servants|\d{1,2})\]|#eBookz")
CAPS_RUN = re.compile(r"^((?:[A-Z][A-Z'\-]*\b\s+){1,5}[A-Z][A-Z'\-]*\b)(?=\s+[a-z]|[,.!?]|\s+[A-Z][a-z])")


def _lines(doc):
    for pno, page in enumerate(doc):
        for block in page.get_text("dict")["blocks"]:
            for line in block.get("lines", []):
                spans = line["spans"]
                text = "".join(s["text"] for s in spans)
                if not text.strip():
                    continue
                yield {"page": pno + 1, "x0": line["bbox"][0], "size": max(s["size"] for s in spans),
                       "bold": any("Bold" in s["font"] or "-Bo" in s["font"] for s in spans), "text": text}


def _clean(text):
    text = MARKERS.sub("", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _uncap(text, names):
    """'BUT ONE DAY he saw' -> 'But one day he saw'; 'IT WAS Wang Lung's' -> 'It was Wang Lung's'."""
    m = CAPS_RUN.match(text)
    if not m or len(m.group(1).split()) < 2:
        return text, False
    run = m.group(1).lower()
    for name in names:
        run = re.sub(r"\b" + re.escape(name.lower()) + r"\b", name, run)
    run = run[0].upper() + run[1:]
    return run + text[m.end(1):], True


def parse_chapters(pdf_path, names=()):
    doc = pymupdf.open(pdf_path)
    lines = list(_lines(doc))
    heads, want = [], 1
    for i, line in enumerate(lines):
        if want <= N_CHAPTERS and line["text"].strip() == str(want) and line["size"] >= HEADING_SIZE:
            heads.append(i)
            want += 1
    if len(heads) != N_CHAPTERS:
        raise ValueError(f"found {len(heads)} chapter headings, expected {N_CHAPTERS}")

    chapters = []
    for k, start in enumerate(heads):
        stop = heads[k + 1] if k + 1 < len(heads) else len(lines)
        paras = []
        for line in lines[start + 1:stop]:
            if line["x0"] >= INDENT_X or not paras:
                paras.append({"page": line["page"], "raw": [line["text"]]})
            else:
                paras[-1]["raw"].append(line["text"])
        out = []
        for p in paras:
            text = _clean(" ".join(p["raw"]))
            if not text:
                continue
            text, section = _uncap(text, names)
            out.append({"text": text, "section_start": section or not out, "page": p["page"]})
        if k + 1 == N_CHAPTERS:
            last = next(i for i, p in enumerate(out) if END_OF_NOVEL in p["text"])
            t = out[last]["text"]
            out[last]["text"] = t[:t.index(END_OF_NOVEL) + len(END_OF_NOVEL)]
            out = out[:last + 1]
        for i, p in enumerate(out):
            p["pid"] = f"{k + 1:02d}.{i + 1:03d}"
            p["joins_previous"] = bool(i and p["text"].startswith('"') and out[i - 1]["text"].endswith((",", ":")))
        chapters.append({"chapter": k + 1, "paragraphs": out})
    return chapters


def text_sha256(chapters):
    joined = "\n\n".join(p["text"] for ch in chapters for p in ch["paragraphs"])
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()


def paragraph_index(chapters):
    return {p["pid"]: p for ch in chapters for p in ch["paragraphs"]}
