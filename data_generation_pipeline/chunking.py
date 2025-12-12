import re
from typing import List
from config import CHUNK_SIZE_WORDS, CHUNK_OVERLAP_WORDS
import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize


def clean_text(t: str) -> str:
    t = re.sub(r"\r\n?", "\n", t)
    t = re.sub(r"\n{2,}", "\n\n", t)   # keep paragraph breaks
    t = re.sub(r"[^\x00-\x7F]+", " ", t)   # remove non-ascii to reduce token issues


    #     # ---------- REMOVE EVERYTHING BEFORE TOC ----------
    t = re.sub(
        r"^[\s\S]*?TABLE OF CONTENTS",
        "TABLE OF CONTENTS",
        t,
        flags=re.IGNORECASE
    )

    #     # ---------- REMOVE TOC ----------
    try:
        t = re.sub(r"TABLE OF CONTENTS[\s\S]*?GLOSSARY[\s\.]*388", "", t)
    except Exception as e:
        # fail-safe: don't break the pipeline if regex crashes
        print("[WARN] TOC removal regex failed:", e)


    #     # ---------- REMOVE PREFACE BLOCK ----------
    #     # Remove PREFACE → first real section
    t = re.sub(
        r"PREFACE[\s\S]*?(SECTION\s+1|CHAPTER\s+1)",
        r"\1",
        t,
        flags=re.IGNORECASE
    )
    # ---------- REMOVE FOOTERS WITH OCR-SPACED LETTERS ----------
    # Remove spaced-out words like A m a n a h, P a g e, V e r s i o n
    t = re.sub(
        r"(?m)^(?:\s*(?:[A-Za-z]\s+){2,}[A-Za-z].*)$",
        "",
        t
    )

    t = re.sub(r"(?m)^\s*$", "", t) # Remove all white space only lines
    t = re.sub(r"\n{3,}", "\n\n", t) # multiple lines
    t = re.sub(r"[ \t]{2,}", " ", t) # final collapse of double spaces

    t = t.strip()


    return t



def chunk_text(text: str, chunk_size: int=CHUNK_SIZE_WORDS, overlap: int=CHUNK_OVERLAP_WORDS) -> List[str]:
    words = text.split()
    if len(words) <= chunk_size:
        return [" ".join(words)]
    chunks = []
    i = 0
    while i < len(words):
        chunk = words[i:i+chunk_size]
        chunks.append(" ".join(chunk))
        i += chunk_size - overlap
    return chunks

# -------------------------
# Sentence-level helper
# -------------------------
def sentences_from_text(text: str) -> List[str]:
    sents = sent_tokenize(text)
    sents = [s.strip() for s in sents if s.strip()]
    return sents
