import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from hallucination_reduction.retriever import SimpleRetriever


class TestSimpleRetriever:
    """Test suite for SimpleRetriever class."""

    @pytest.fixture
    def sample_passages(self):
        """Create sample passages for testing."""
        return [
            "Python is a programming language",
            "Java is used for enterprise applications",
            "JavaScript runs in web browsers",
            "Machine learning uses algorithms",
            "Deep learning is a subset of machine learning"
        ]

    def test_class_exists(self):
        """Test that SimpleRetriever class can be imported."""
        assert SimpleRetriever is not None

    def test_initialization(self, sample_passages):
        """Test retriever initialization."""
        retriever = SimpleRetriever(sample_passages)
        assert retriever is not None

    def test_stores_passages(self, sample_passages):
        """Test that passages are stored."""
        retriever = SimpleRetriever(sample_passages)
        assert retriever.passages == sample_passages

    def test_creates_vectorizer(self, sample_passages):
        """Test that TfidfVectorizer is created."""
        retriever = SimpleRetriever(sample_passages)
        assert retriever.vectorizer is not None

    def test_creates_vectors(self, sample_passages):
        """Test that document vectors are created."""
        retriever = SimpleRetriever(sample_passages)
        assert retriever.vectors is not None

    def test_vectors_match_passage_count(self, sample_passages):
        """Test that number of vectors matches number of passages."""
        retriever = SimpleRetriever(sample_passages)
        assert retriever.vectors.shape[0] == len(sample_passages)

    def test_retrieve_returns_list(self, sample_passages):
        """Test that retrieve returns a list."""
        retriever = SimpleRetriever(sample_passages)
        query = "What is Python?"
        result = retriever.retrieve(query)
        assert isinstance(result, list)

    def test_retrieve_returns_tuples(self, sample_passages):
        """Test that retrieve returns list of tuples."""
        retriever = SimpleRetriever(sample_passages)
        query = "What is Python?"
        result = retriever.retrieve(query)
        assert all(isinstance(item, tuple) for item in result)

    def test_retrieve_tuple_structure(self, sample_passages):
        """Test that each tuple contains (index, passage)."""
        retriever = SimpleRetriever(sample_passages)
        query = "What is Python?"
        result = retriever.retrieve(query)
        
        for idx, passage in result:
            assert isinstance(idx, int)
            assert isinstance(passage, str)

    def test_retrieve_default_k_is_three(self, sample_passages):
        """Test that default k returns 3 results."""
        retriever = SimpleRetriever(sample_passages)
        query = "programming"
        result = retriever.retrieve(query)
        assert len(result) == 3

    def test_retrieve_custom_k(self, sample_passages):
        """Test that custom k parameter works."""
        retriever = SimpleRetriever(sample_passages)
        query = "programming"
        k = 2
        result = retriever.retrieve(query, k=k)
        assert len(result) == k

    def test_retrieve_k_equals_one(self, sample_passages):
        """Test retrieving single result."""
        retriever = SimpleRetriever(sample_passages)
        query = "Python programming"
        result = retriever.retrieve(query, k=1)
        assert len(result) == 1

    def test_retrieve_k_larger_than_passages(self, sample_passages):
        """Test that k larger than passage count returns all passages."""
        retriever = SimpleRetriever(sample_passages)
        query = "programming"
        k = 100
        result = retriever.retrieve(query, k=k)
        assert len(result) == len(sample_passages)

    def test_retrieve_returns_valid_indices(self, sample_passages):
        """Test that returned indices are valid."""
        retriever = SimpleRetriever(sample_passages)
        query = "programming"
        result = retriever.retrieve(query)
        
        for idx, _ in result:
            assert 0 <= idx < len(sample_passages)

    def test_retrieve_returns_correct_passages(self, sample_passages):
        """Test that returned passages match the indices."""
        retriever = SimpleRetriever(sample_passages)
        query = "programming"
        result = retriever.retrieve(query)
        
        for idx, passage in result:
            assert passage == sample_passages[idx]

    def test_retrieve_relevant_result_for_python(self, sample_passages):
        """Test that Python query returns Python passage."""
        retriever = SimpleRetriever(sample_passages)
        query = "Python programming language"
        result = retriever.retrieve(query, k=1)
        
        idx, passage = result[0]
        assert "Python" in passage

    def test_retrieve_relevant_result_for_machine_learning(self, sample_passages):
        """Test that machine learning query returns ML passage."""
        retriever = SimpleRetriever(sample_passages)
        query = "machine learning algorithms"
        result = retriever.retrieve(query, k=1)
        
        idx, passage = result[0]
        assert "machine learning" in passage.lower()

    def test_retrieve_handles_empty_query(self, sample_passages):
        """Test that empty query is handled."""
        retriever = SimpleRetriever(sample_passages)
        query = ""
        result = retriever.retrieve(query)
        
        # Should still return k results
        assert len(result) == 3

    def test_retrieve_handles_query_with_no_matches(self, sample_passages):
        """Test query with words not in any passage."""
        retriever = SimpleRetriever(sample_passages)
        query = "quantum physics relativity"
        result = retriever.retrieve(query)
        
        # Should still return k results (lowest scores)
        assert len(result) == 3

    def test_retrieve_case_insensitive(self, sample_passages):
        """Test that retrieval works regardless of case."""
        retriever = SimpleRetriever(sample_passages)
        
        query_lower = "python programming"
        query_upper = "PYTHON PROGRAMMING"
        
        result_lower = retriever.retrieve(query_lower, k=1)
        result_upper = retriever.retrieve(query_upper, k=1)
        
        # Should retrieve same passage (case-insensitive)
        assert result_lower[0][1] == result_upper[0][1]

    def test_returns_ordered_by_relevance(self, sample_passages):
        """Test that results are ordered by relevance (descending)."""
        retriever = SimpleRetriever(sample_passages)
        query = "machine learning deep learning"
        result = retriever.retrieve(query, k=3)
        
        # First result should be most relevant
        # Check that machine learning passages come first
        top_passage = result[0][1]
        assert "learning" in top_passage.lower()

    def test_single_passage(self):
        """Test retriever with single passage."""
        passages = ["Single passage about Python"]
        retriever = SimpleRetriever(passages)
        
        query = "Python"
        result = retriever.retrieve(query, k=1)
        
        assert len(result) == 1
        assert result[0][1] == passages[0]

    def test_two_passages(self):
        """Test retriever with two passages."""
        passages = ["Python programming", "Java development"]
        retriever = SimpleRetriever(passages)
        
        query = "Python"
        result = retriever.retrieve(query, k=2)
        
        assert len(result) == 2

    def test_identical_passages(self):
        """Test retriever with identical passages."""
        passages = ["Same text", "Same text", "Same text"]
        retriever = SimpleRetriever(passages)
        
        query = "Same text"
        result = retriever.retrieve(query, k=3)
        
        assert len(result) == 3
        # All should have same text
        assert all(passage == "Same text" for _, passage in result)

    def test_retrieve_with_special_characters(self, sample_passages):
        """Test query with special characters."""
        retriever = SimpleRetriever(sample_passages)
        query = "What is Python? Can you explain!"
        result = retriever.retrieve(query)
        
        assert len(result) > 0
        assert all(isinstance(idx, int) for idx, _ in result)

    def test_retrieve_with_numbers(self):
        """Test passages and queries with numbers."""
        passages = [
            "Python 3.9 is the latest version",
            "Java version 11 is stable",
            "JavaScript ES6 features"
        ]
        retriever = SimpleRetriever(passages)
        
        query = "Python 3.9"
        result = retriever.retrieve(query, k=1)
        
        assert "Python" in result[0][1]

    def test_retrieve_preserves_index_order_information(self, sample_passages):
        """Test that indices correspond to original passage positions."""
        retriever = SimpleRetriever(sample_passages)
        query = "programming"
        result = retriever.retrieve(query, k=len(sample_passages))
        
        # Verify all indices are valid positions
        retrieved_indices = [idx for idx, _ in result]
        assert all(idx in range(len(sample_passages)) for idx in retrieved_indices)

    def test_vectorizer_fitted_on_passages(self, sample_passages):
        """Test that vectorizer is fitted on all passages."""
        retriever = SimpleRetriever(sample_passages)
        
        # Vectorizer should have vocabulary from passages
        vocab = retriever.vectorizer.get_feature_names_out()
        assert len(vocab) > 0

    def test_cosine_similarity_used(self, sample_passages):
        """Test that cosine similarity is computed."""
        retriever = SimpleRetriever(sample_passages)
        
        with patch('hallucination_reduction.retriever.cosine_similarity') as mock_cosine:
            # Return dummy similarities
            mock_cosine.return_value = np.array([[0.5, 0.3, 0.8, 0.1, 0.2]])
            
            query = "test query"
            retriever.retrieve(query, k=2)
            
            # Cosine similarity should be called
            mock_cosine.assert_called_once()

    def test_retrieve_k_zero(self, sample_passages):
        """Test that k=0 returns empty list."""
        retriever = SimpleRetriever(sample_passages)
        query = "Python"
        result = retriever.retrieve(query, k=0)
        
        assert result == []

    def test_retrieve_negative_k_returns_empty(self, sample_passages):
        """Test that negative k returns empty list."""
        retriever = SimpleRetriever(sample_passages)
        query = "Python"
        result = retriever.retrieve(query, k=-1)
        
        # argsort with negative slice returns empty
        assert result == []

    def test_multiple_queries_same_retriever(self, sample_passages):
        """Test that same retriever can handle multiple queries."""
        retriever = SimpleRetriever(sample_passages)
        
        query1 = "Python"
        query2 = "Java"
        query3 = "machine learning"
        
        result1 = retriever.retrieve(query1, k=1)
        result2 = retriever.retrieve(query2, k=1)
        result3 = retriever.retrieve(query3, k=1)
        
        assert len(result1) == 1
        assert len(result2) == 1
        assert len(result3) == 1

    def test_passages_not_modified(self, sample_passages):
        """Test that original passages are not modified."""
        original_passages = sample_passages.copy()
        retriever = SimpleRetriever(sample_passages)
        
        query = "Python"
        retriever.retrieve(query)
        
        assert retriever.passages == original_passages

    def test_long_query(self, sample_passages):
        """Test with a very long query."""
        retriever = SimpleRetriever(sample_passages)
        query = " ".join(["Python programming language"] * 100)
        result = retriever.retrieve(query, k=2)
        
        assert len(result) == 2

    def test_passages_with_unicode(self):
        """Test passages with Unicode characters."""
        passages = [
            "Python es un lenguaje de programación",
            "Java est un langage de programmation",
            "JavaScript 是一种编程语言"
        ]
        retriever = SimpleRetriever(passages)
        
        query = "programming"
        result = retriever.retrieve(query)
        
        assert len(result) > 0

    def test_retrieve_consistent_results(self, sample_passages):
        """Test that same query returns consistent results."""
        retriever = SimpleRetriever(sample_passages)
        query = "Python programming"
        
        result1 = retriever.retrieve(query, k=3)
        result2 = retriever.retrieve(query, k=3)
        
        # Should return same results
        assert result1 == result2

    def test_index_type_is_int(self, sample_passages):
        """Test that returned indices are Python int type."""
        retriever = SimpleRetriever(sample_passages)
        query = "Python"
        result = retriever.retrieve(query)
        
        for idx, _ in result:
            assert type(idx) == int  # Not np.int64 or other numpy types