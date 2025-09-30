import nltk
from nltk.tokenize import sent_tokenize

nltk.download("punkt", quiet=True)

def split_into_passages(text: str, max_words: int = 200):
    """Splits raw text into passages (by sentences)."""
    sentences = sent_tokenize(text)
    passages, current, count = [], [], 0
    for sent in sentences:
        words = sent.split()
        if count + len(words) > max_words:
            passages.append(" ".join(current))
            current, count = [], 0
        current.append(sent)
        count += len(words)
    if current:
        passages.append(" ".join(current))
    return passages
