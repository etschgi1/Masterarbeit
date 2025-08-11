# be_ae_check_tex.py
import re, argparse, pathlib, enchant
from collections import Counter

# --- setup dictionary (British English) ---
DICT = enchant.Dict("en_GB")

# --- quick LaTeX scrubber (good enough for spell checks) ---
LATEX_MATH_PATTERNS = [
    r"\$\$.*?\$\$", r"\$.*?\$",          # $$...$$ and $...$
    r"\\\[(.*?)\\\]", r"\\\((.*?)\\\)",  # \[...\], \(...\)
]
LATEX_ENV_NAMES = ["equation", "align", "align*", "gather", "multline", "lstlisting", "verbatim"]
LATEX_CMDS_DROPARGS = ["cite", "ref", "eqref", "autoref", "label", "url", "href"]

def strip_latex(s: str) -> str:
    # remove comments (ignore escaped \%)
    s = re.sub(r"(?<!\\)%.*?$", "", s, flags=re.M)
    # remove math
    for pat in LATEX_MATH_PATTERNS:
        s = re.sub(pat, " ", s, flags=re.S)
    # remove common math/text environments
    for env in LATEX_ENV_NAMES:
        s = re.sub(rf"\\begin{{{env}}}.*?\\end{{{env}}}", " ", s, flags=re.S)
    # remove \command[...]{...} fully for noisy commands
    for cmd in LATEX_CMDS_DROPARGS:
        s = re.sub(rf"\\{cmd}\s*(\[[^\]]*\])?\s*\{{[^}}]*\}}", " ", s)
    # drop remaining commands like \command or \command[...]
    s = re.sub(r"\\[a-zA-Z@]+(\s*\[[^\]]*\])?", " ", s)
    # drop braces
    s = s.replace("{", " ").replace("}", " ")
    # drop TeX special escapes ~ ^ _
    s = re.sub(r"[~^_]", " ", s)
    return s

# tokenization: words incl. hyphen/apostrophe (we also check hyphen parts)
WORD_RE = re.compile(r"\b[A-Za-z][A-Za-z'\-]*\b")

def words_not_in_dict(text: str):
    misses = Counter()
    for tok in WORD_RE.findall(text):
        # check full token; if hyphenated, also check pieces
        candidates = [tok] + tok.split("-")
        ok = any(DICT.check(w) or DICT.check(w.lower()) for w in candidates if w)
        if not ok:
            misses[tok] += 1
    return misses

def main():
    ap = argparse.ArgumentParser(description="Check BE/AE consistency in .tex files using PyEnchant (en_GB).")
    ap.add_argument("folder", type=pathlib.Path, help="Folder with .tex files")
    ap.add_argument("-r", "--recursive", action="store_true", help="Recurse into subfolders")
    args = ap.parse_args()

    pattern = "**/*.tex" if args.recursive else "*.tex"
    files = sorted(args.folder.glob(pattern))
    if not files:
        print("No .tex files found.")
        return

    for f in files:
        text = f.read_text(encoding="utf-8", errors="ignore")
        clean = strip_latex(text)
        misses = words_not_in_dict(clean)
        print(f"\n=== {f} ===")
        if not misses:
            print("OK: no out-of-dictionary words for en_GB.")
            continue
        for w, c in sorted(misses.items(), key=lambda x: (-x[1], x[0].lower())):
            print(f"{w}\t({c})")
        input("Press key to continue...")

if __name__ == "__main__":
    main()
