import re
from typing import List, Tuple

Q_RE = re.compile(r'^\s*Q\s*(\d+)\s*[:.\-]\s*(.+?)(\?)?\s*$', re.IGNORECASE)
A_RE = re.compile(r'^\s*A\s*(\d+)\s*[:.\-]\s*(.+)', re.IGNORECASE)

def parse_qa_block(block: str):
    lines = [ln.strip() for ln in block.splitlines() if ln.strip()]
    pairs = []
    i = 0
    while i < len(lines)-1:
        q_match = Q_RE.match(lines[i])
        a_match = A_RE.match(lines[i+1])
        if q_match and a_match and q_match.group(1) == a_match.group(1):
            q = q_match.group(2).strip()
            a = a_match.group(2).strip().rstrip(".")
            pairs.append((q, a))
            i += 2
        else:
            i += 1
    return pairs

