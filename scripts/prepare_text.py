"""PDF edition -> data/text/chapters.json (34 chapters, paragraphs with ids and section flags)."""

import argparse
import json
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from charsim.cast import Cast  # noqa: E402
from charsim.text import parse_chapters, text_sha256  # noqa: E402

# a PDF copy of the novel (not distributed); its chapters.json can be checked against data/checksums.json
PDF = os.path.join("data", "source", "the_good_earth.pdf")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", default=PDF)
    ap.add_argument("--cast", default="config/cast.json")
    ap.add_argument("--out", default="data/text/chapters.json")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    cast = Cast(args.cast)
    chapters = parse_chapters(args.pdf, names=cast.names() + ["Hwang"])
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"source": os.path.basename(args.pdf), "sha256": text_sha256(chapters),
                   "chapters": chapters}, f, ensure_ascii=False, indent=0)

    n_words = 0
    for ch in chapters:
        w = sum(len(p["text"].split()) for p in ch["paragraphs"])
        n_words += w
        secs = sum(p["section_start"] for p in ch["paragraphs"])
        print(f"ch{ch['chapter']:2d} paragraphs={len(ch['paragraphs']):4d} sections={secs:2d} words={w:6d} "
              f"| {ch['paragraphs'][0]['text'][:60]}")
    leftovers = [p["pid"] for ch in chapters for p in ch["paragraphs"]
                 if re.search(r"\[pict|#eBookz|<<<", p["text"])]
    print(f"total words={n_words}; paragraphs with leftover markers={leftovers}")
    print("last paragraph:", chapters[-1]["paragraphs"][-1]["text"][-120:])
    print("sha256:", text_sha256(chapters))


if __name__ == "__main__":
    main()
