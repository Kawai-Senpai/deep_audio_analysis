# save as find_bad_encodings.py and run with python
from pathlib import Path

text_globs = ["*.py","*.md","*.txt","*.json","*.ipynb",".gitattributes",".gitignore"]
paths = {p for g in text_globs for p in Path(".").rglob(g)}
bad = []
for p in sorted(paths):
    try:
        p.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        bad.append(str(p))
for p in bad:
    print(p)
