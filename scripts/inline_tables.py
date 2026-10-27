"""Write a copy of the manuscript with the table fragments of scripts/make_tables.py pasted in place
of their \\tabinput lines and without the drafting macros, as the input of make_single_tex.py.
Usage: python scripts/inline_tables.py paper/manuscript.tex paper/manuscript_tables.tex"""

import os
import re
import sys

src, out = sys.argv[1], sys.argv[2]
folder = os.path.dirname(src)
s = open(src, encoding="utf-8").read()
s = re.sub(r"\\tabinput\{([^}]+)\}", lambda m: open(os.path.join(folder, m.group(1)), encoding="utf-8").read().rstrip("\n"), s)
s = s.replace("\\makeatletter\n\\newcommand{\\tabinput}[1]{\\@@input #1 }\n\\makeatother\n", "")
s = re.sub(r"\\newcommand\{\\placeholder\}[^\n]*\n", "", s)
assert "\\tabinput" not in s and "\\placeholder" not in s
open(out, "w", encoding="utf-8").write(s)
print("tables inlined:", out)
