from typing import List, Tuple
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class SimpleRetriever:
    def __init__(self, passages: List[str]):
        self.passages = passages
        self.vectorizer = TfidfVectorizer().fit(passages)
        self.vectors = self.vectorizer.transform(passages)

    def retrieve(self, query: str, k: int = 3) -> List[Tuple[int, str]]:
        qv = self.vectorizer.transform([query])
        sims = cosine_similarity(qv, self.vectors)[0]
        idxs = np.argsort(-sims)[:k]
        return [(int(i), self.passages[i]) for i in idxs]
