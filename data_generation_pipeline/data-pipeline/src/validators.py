import re

POLICY_NUM_RE = re.compile(r'\b(?:section|sec|clause|article|policy)\s*\d+(?:\.\d+)*\b', re.IGNORECASE)
NAMED_POLICY_RE = re.compile(r'\b([A-Z][A-Za-z&\-]{2,}(?:\s+[A-Z][A-Za-z&\-]{2,}){0,6}\s+(?:Policy|policy))\b')
ABBR_POLICY_RE = re.compile(r'\b(KYC|CDD|EDD|AML|CFT|FD|ATM|STR|CTR)\b', re.IGNORECASE)

def question_has_valid_reference(q: str) -> bool:
    if POLICY_NUM_RE.search(q):
        return True
    if NAMED_POLICY_RE.search(q):
        return True
    if ABBR_POLICY_RE.search(q):
        return True
    return False

def is_vague_question(q: str) -> bool:
    ql = q.lower()
    vague_phrases = ["purpose of this policy", "what is this policy", "what is the purpose of this policy", "what does this policy say"]
    return any(p in ql for p in vague_phrases)

def valid_question(q: str) -> bool:
    if not q or len(q) < 5:
        return False
    if is_vague_question(q) and not question_has_valid_reference(q):
        return False
    return question_has_valid_reference(q)

