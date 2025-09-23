# hallucination_reduction/retriever.py
import faiss
from sklearn.feature_extraction.text import TfidfVectorizer

class SimpleRetriever:
    """
    Simple FAISS retriever over TF-IDF embeddings of passages.
    """
    def __init__(self, passages):
        self.passages = passages
        self.vectorizer = TfidfVectorizer().fit(passages)
        vectors = self.vectorizer.transform(passages).astype('float32')
        self.vectors = vectors
        self.index = faiss.IndexFlatL2(vectors.shape[1])
        self.index.add(vectors.toarray())

    def retrieve(self, query, k=5):
        """
        Retrieve top-k most relevant passages for a query.
        Returns list of tuples: (passage_text, similarity_score)
        """
        qv = self.vectorizer.transform([query]).astype('float32')
        distances, indices = self.index.search(qv.toarray(), k)
        # Convert L2 distance to similarity for better interpretability
        sims = 1 / (1 + distances[0])
        return [(self.passages[i], sims[j]) for j, i in enumerate(indices[0])]


def build_rag_prompt(question, retrieved_docs):
    """
    Construct a RAG-style prompt with retrieved context.
    retrieved_docs: list of tuples (passage_text, similarity_score)
    """
    prompt = "### Context:\n"
    for i, (doc, _) in enumerate(retrieved_docs, 1):
        prompt += f"[{i}] {doc}\n"
    prompt += f"\n### Question:\n{question}\n### Answer:\n"
    return prompt
