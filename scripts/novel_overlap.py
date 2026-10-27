"""Find passages of the novel in files that are to be released (copyright check).

The text of The Good Earth (data/text/chapters.json) is cut into windows of N characters (after
folding quotation marks and white space). A file passage counts as copied when it contains such a
window. The script reports, per file, the copied passages of at least N characters; with
--redact it replaces each of them by [GE-OMITTED:<first 12 hex digits of its SHA-256>] and writes a
manifest of the replaced passages (hashes and lengths only, no text).
Usage: python scripts/novel_overlap.py FILE_OR_DIR ... [--n 25] [--redact --manifest PATH]"""

import argparse
import hashlib
import json
import os
import re
import sys

TEXT_EXT = (".json", ".jsonl", ".csv", ".txt", ".md", ".tex", ".py", ".sh", ".cff", ".html")
FOLD = str.maketrans({"“": '"', "”": '"', "‘": "'", "’": "'", "—": "-", "–": "-"})


def norm(s):
    return re.sub(r"\s+", " ", s.translate(FOLD))


def novel_windows(path, n):
    with open(path, encoding="utf-8") as f:
        chapters = json.load(f)["chapters"]
    text = norm(" ".join(p["text"] for ch in chapters for p in ch["paragraphs"]))
    return {text[i:i + n] for i in range(len(text) - n + 1)}


def copied_spans(text, windows, n):
    """Character spans of text (in its own coordinates) covered by windows of the novel."""
    folded = text.translate(FOLD)  # same length as text: every folded character maps to one character
    hits = [False] * len(folded)
    # white space runs are folded to one space for the lookup; keep a map back to positions
    idx, chars = [], []
    for i, ch in enumerate(folded):
        if ch.isspace():
            if chars and chars[-1] == " ":
                continue
            ch = " "
        idx.append(i)
        chars.append(ch)
    s = "".join(chars)
    for i in range(len(s) - n + 1):
        if s[i:i + n] in windows:
            for j in range(idx[i], idx[i + n - 1] + 1):
                hits[j] = True
    spans, start = [], None
    for i, h in enumerate(hits + [False]):
        if h and start is None:
            start = i
        elif not h and start is not None:
            spans.append((start, i))
            start = None
    return spans


def redact_text(text, windows, n, path, manifest):
    """text with every copied passage replaced by a placeholder; the passages go to the manifest."""
    spans = copied_spans(text, windows, n)
    out, last = [], 0
    for a, b in spans:
        h = hashlib.sha256(norm(text[a:b]).encode("utf-8")).hexdigest()
        out += [text[last:a], f"[GE-OMITTED:{h[:12]}]"]
        manifest.append({"file": path, "sha256": h, "length": b - a})
        last = b
    out.append(text[last:])
    return "".join(out)


def redact_value(v, windows, n, path, manifest):
    """The same inside the strings of a decoded JSON value, so that the file stays valid JSON."""
    if isinstance(v, str):
        return redact_text(v, windows, n, path, manifest)
    if isinstance(v, list):
        return [redact_value(x, windows, n, path, manifest) for x in v]
    if isinstance(v, dict):
        return {k: redact_value(x, windows, n, path, manifest) for k, x in v.items()}
    return v


def files(paths):
    for p in paths:
        if os.path.isdir(p):
            for root, _, names in os.walk(p):
                for name in sorted(names):
                    if name.endswith(TEXT_EXT):
                        yield os.path.join(root, name)
        elif p.endswith(TEXT_EXT):
            yield p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--novel", default="data/text/chapters.json")
    ap.add_argument("--n", type=int, default=25)
    ap.add_argument("--redact", action="store_true")
    ap.add_argument("--manifest", default=None)
    ap.add_argument("--show", type=int, default=3, help="passages shown per file in the report")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    windows = novel_windows(args.novel, args.n)
    manifest, n_files, n_spans, n_chars = [], 0, 0, 0
    for path in files(args.paths):
        with open(path, encoding="utf-8", errors="replace") as f:
            text = f.read()
        spans = copied_spans(text, windows, args.n)
        if not spans:
            continue
        n_files += 1
        n_spans += len(spans)
        n_chars += sum(b - a for a, b in spans)
        if args.show:
            print(f"{path}: {len(spans)} passages, {sum(b - a for a, b in spans)} characters")
            for a, b in spans[:args.show]:
                print("    ", repr(text[a:b][:120]))
        if args.redact:
            if path.endswith(".json"):
                obj = redact_value(json.loads(text), windows, args.n, path, manifest)
                new = json.dumps(obj, ensure_ascii=False, indent=1)
            elif path.endswith(".jsonl"):
                new = "".join(json.dumps(redact_value(json.loads(line), windows, args.n, path, manifest),
                                         ensure_ascii=False) + "\n" for line in text.splitlines() if line.strip())
            else:
                new = redact_text(text, windows, args.n, path, manifest)
            with open(path, "w", encoding="utf-8") as f:
                f.write(new)
    print(f"files with passages of the novel: {n_files}, passages: {n_spans}, characters: {n_chars}")
    if args.redact and args.manifest:
        with open(args.manifest, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=1)


if __name__ == "__main__":
    main()
