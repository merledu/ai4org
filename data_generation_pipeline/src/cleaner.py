import re


def clean_text(t: str) -> str:
    if not t:
        return ""
    t = re.sub(r"\r\n?", "\n", t)
    t = re.sub(r"\n{2,}", "\n\n", t)
    # remove non-ascii sequences to reduce tokenization surprises
    t = re.sub(r"[^\x00-\x7F]+", " ", t)

    # remove everything before TABLE OF CONTENTS if present
    t = re.sub(
        r"^[\s\S]*?TABLE OF CONTENTS", "TABLE OF CONTENTS", t, flags=re.IGNORECASE
    )

    # a safe try to remove TOC region (non-fatal)
    try:
        t = re.sub(r"TABLE OF CONTENTS[\s\S]*?GLOSSARY[\s\.]*388", "", t)
    except Exception:
        pass
    # REMOVE MINI TABLE OF CONTENTS
    t = re.sub(
        r"(?:\n|^)(?:[A-Za-z][A-Za-z0-9\s\(\)/,&\-]+?\s+\d+(?:\.\d+)+\s*){6,}",
        "\n",
        t,
        flags=re.MULTILINE,
    )

    # ---------- REMOVE ANY REMAINING MINI-TOC LINES INSIDE 1.1 ----------
    m = re.search(
        r"(1\.1\s+OVERVIEW[\s\S]*?)(\n1\.2\s+CURRENT\s+ACCOUNTS)",
        t,
        flags=re.IGNORECASE,
    )
    if m:
        section_11 = m.group(1)
        rest = t[m.end(1) :]
        # remove lines ending with x.y.z inside 1.1
        section_11 = re.sub(r"(?m)^[^\n]*\b\d+\.\d+\.\d+\s*$", "", section_11)
        t = section_11.strip() + "\n\n" + rest.strip()

    # remove PREFACE up to first SECTION / CHAPTER
    t = re.sub(
        r"PREFACE[\s\S]*?(SECTION\s+1|CHAPTER\s+1)", r"\1", t, flags=re.IGNORECASE
    )

    # remove OCR-styled spaced footers e.g., "A m a n a h"
    t = re.sub(r"(?m)^(?:\s*(?:[A-Za-z]\s+){2,}[A-Za-z].*)$", "", t)

    t = re.sub(r"(?m)^\s*$", "", t)  # remove blank lines
    t = re.sub(r"\n{3,}", "\n\n", t)
    t = re.sub(r"[ \t]{2,}", " ", t)
    return t.strip()


BAD_MARKERS = ["Human:", "You are generating", "<question>", "<answer>"]


def clean_text_leakage(text: str) -> str:
    for marker in BAD_MARKERS:
        if marker in text:
            return ""
    return text.strip()
