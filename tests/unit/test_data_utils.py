import pytest
import os
import json
from unittest.mock import mock_open, patch, MagicMock
from hallucination_reduction.data_utils import build_corpus_and_qa, QAPair  


class TestBuildCorpusAndQA:
    """Comprehensive test suite for build_corpus_and_qa function."""

    def test_build_returns_two_items(self):
        """Ensure build_corpus_and_qa returns exactly two items."""
        with patch("builtins.open", mock_open(read_data="passage1\npassage2\n")):
            with patch("os.path.exists", return_value=False):
                passages, qa_pairs = build_corpus_and_qa()
                assert isinstance(passages, list)
                assert isinstance(qa_pairs, list)

    def test_returns_correct_types(self):
        """Verify return types are lists."""
        with patch("builtins.open", mock_open(read_data="test passage\n")):
            with patch("os.path.exists", return_value=False):
                passages, qa_pairs = build_corpus_and_qa()
                assert type(passages) == list
                assert type(qa_pairs) == list

    def test_passages_loaded_correctly(self):
        """Test that passages are read and stripped properly."""
        corpus_data = "passage one\n\npassage two\n  passage three  \n"
        with patch("builtins.open", mock_open(read_data=corpus_data)):
            with patch("os.path.exists", return_value=False):
                passages, _ = build_corpus_and_qa()
                assert len(passages) == 3
                assert passages[0] == "passage one"
                assert passages[1] == "passage two"
                assert passages[2] == "passage three"

    def test_empty_lines_filtered_out(self):
        """Ensure empty lines in corpus are filtered."""
        corpus_data = "passage1\n\n\npassage2\n"
        with patch("builtins.open", mock_open(read_data=corpus_data)):
            with patch("os.path.exists", return_value=False):
                passages, _ = build_corpus_and_qa()
                assert len(passages) == 2

    def test_qa_json_not_exists_generates_default_qa(self):
        """When qa.json doesn't exist, generate default Q&A pairs."""
        corpus_data = "passage1\npassage2\n"
        with patch("builtins.open", mock_open(read_data=corpus_data)):
            with patch("os.path.exists", return_value=False):
                _, qa_pairs = build_corpus_and_qa()
                # Should have 2 passages * 3 augmentations = 6 QA pairs
                assert len(qa_pairs) == 6
                assert all(isinstance(qa, QAPair) for qa in qa_pairs)

    def test_qa_json_exists_and_valid(self):
        """Test loading valid QA pairs from JSON file."""
        corpus_data = "passage1\n"
        qa_json_data = json.dumps([
            {
                "question": "What is X?",
                "answer": "X is Y",
                "supporting_passages": ["passage1"]
            }
        ])
        
        def mock_open_handler(filename, *args, **kwargs):
            if "corpus.txt" in filename:
                return mock_open(read_data=corpus_data)()
            elif "qa.json" in filename:
                return mock_open(read_data=qa_json_data)()
        
        with patch("builtins.open", side_effect=mock_open_handler):
            with patch("os.path.exists", return_value=True):
                _, qa_pairs = build_corpus_and_qa()
                # 1 base QA * 3 augmentations = 3
                assert len(qa_pairs) == 3
                assert qa_pairs[0].question == "What is X?"
                assert qa_pairs[0].answer == "X is Y"

    def test_qa_json_malformed_falls_back_to_default(self):
        """When qa.json is malformed, fall back to default generation."""
        corpus_data = "passage1\n"
        qa_json_data = "not valid json{{"
        
        def mock_open_handler(filename, *args, **kwargs):
            if "corpus.txt" in filename:
                return mock_open(read_data=corpus_data)()
            elif "qa.json" in filename:
                return mock_open(read_data=qa_json_data)()
        
        with patch("builtins.open", side_effect=mock_open_handler):
            with patch("os.path.exists", return_value=True):
                _, qa_pairs = build_corpus_and_qa()
                # Should fall back to default generation
                assert len(qa_pairs) == 3  # 1 passage * 3 augmentations

    def test_qa_json_not_a_list(self):
        """When qa.json contains non-list data, fall back to default."""
        corpus_data = "passage1\n"
        qa_json_data = json.dumps({"not": "a list"})
        
        def mock_open_handler(filename, *args, **kwargs):
            if "corpus.txt" in filename:
                return mock_open(read_data=corpus_data)()
            elif "qa.json" in filename:
                return mock_open(read_data=qa_json_data)()
        
        with patch("builtins.open", side_effect=mock_open_handler):
            with patch("os.path.exists", return_value=True):
                _, qa_pairs = build_corpus_and_qa()
                assert len(qa_pairs) == 3

    def test_qa_missing_required_fields_skipped(self):
        """QA pairs without question or answer should be skipped."""
        corpus_data = "passage1\n"
        qa_json_data = json.dumps([
            {"question": "Valid?", "answer": "Yes"},
            {"question": "No answer?"},  # Missing answer
            {"answer": "No question"},   # Missing question
            {}                            # Empty
        ])
        
        def mock_open_handler(filename, *args, **kwargs):
            if "corpus.txt" in filename:
                return mock_open(read_data=corpus_data)()
            elif "qa.json" in filename:
                return mock_open(read_data=qa_json_data)()
        
        with patch("builtins.open", side_effect=mock_open_handler):
            with patch("os.path.exists", return_value=True):
                _, qa_pairs = build_corpus_and_qa()
                # Only 1 valid QA * 3 augmentations = 3
                assert len(qa_pairs) == 3

    def test_supporting_passages_string_converted_to_list(self):
        """When supporting_passages is a string, convert to list."""
        corpus_data = "passage1\n"
        qa_json_data = json.dumps([
            {
                "question": "Q?",
                "answer": "A",
                "supporting_passages": "single_passage"
            }
        ])
        
        def mock_open_handler(filename, *args, **kwargs):
            if "corpus.txt" in filename:
                return mock_open(read_data=corpus_data)()
            elif "qa.json" in filename:
                return mock_open(read_data=qa_json_data)()
        
        with patch("builtins.open", side_effect=mock_open_handler):
            with patch("os.path.exists", return_value=True):
                _, qa_pairs = build_corpus_and_qa()
                assert isinstance(qa_pairs[0].supporting_passages, list)
                assert qa_pairs[0].supporting_passages == ["single_passage"]

    def test_supporting_passages_none_becomes_empty_list(self):
        """When supporting_passages is None, use empty list."""
        corpus_data = "passage1\n"
        qa_json_data = json.dumps([
            {
                "question": "Q?",
                "answer": "A",
                "supporting_passages": None
            }
        ])
        
        def mock_open_handler(filename, *args, **kwargs):
            if "corpus.txt" in filename:
                return mock_open(read_data=corpus_data)()
            elif "qa.json" in filename:
                return mock_open(read_data=qa_json_data)()
        
        with patch("builtins.open", side_effect=mock_open_handler):
            with patch("os.path.exists", return_value=True):
                _, qa_pairs = build_corpus_and_qa()
                assert qa_pairs[0].supporting_passages == []

    def test_default_qa_generation_bounded_to_200_passages(self):
        """Default QA generation should be limited to first 200 passages."""
        corpus_data = "\n".join([f"passage{i}" for i in range(250)])
        with patch("builtins.open", mock_open(read_data=corpus_data)):
            with patch("os.path.exists", return_value=False):
                _, qa_pairs = build_corpus_and_qa()
                # 200 passages * 3 augmentations = 600
                assert len(qa_pairs) == 600

    def test_default_qa_structure(self):
        """Test default QA pairs have correct structure."""
        corpus_data = "test passage content\n"
        with patch("builtins.open", mock_open(read_data=corpus_data)):
            with patch("os.path.exists", return_value=False):
                _, qa_pairs = build_corpus_and_qa()
                base_qa = qa_pairs[0]
                assert base_qa.question == "What does this passage say?"
                assert base_qa.answer == "test passage content"[:500]
                assert base_qa.supporting_passages == ["test passage content"]

    def test_augmentation_creates_three_variants(self):
        """Each base QA should produce 3 augmented variants."""
        corpus_data = "passage1\n"
        qa_json_data = json.dumps([
            {
                "question": "Can a customer do X?",
                "answer": "Yes",
                "supporting_passages": ["passage1"]
            }
        ])
        
        def mock_open_handler(filename, *args, **kwargs):
            if "corpus.txt" in filename:
                return mock_open(read_data=corpus_data)()
            elif "qa.json" in filename:
                return mock_open(read_data=qa_json_data)()
        
        with patch("builtins.open", side_effect=mock_open_handler):
            with patch("os.path.exists", return_value=True):
                _, qa_pairs = build_corpus_and_qa()
                assert len(qa_pairs) == 3
                assert qa_pairs[0].question == "Can a customer do X?"
                assert qa_pairs[1].question == "Is it possible for a customer do X?"
                assert qa_pairs[2].question == "Can a customer do X? (please advise)"

    def test_augmentation_preserves_answer_and_passages(self):
        """Augmented QA pairs should keep the same answer and supporting passages."""
        corpus_data = "passage1\n"
        qa_json_data = json.dumps([
            {
                "question": "Original?",
                "answer": "Original answer",
                "supporting_passages": ["p1", "p2"]
            }
        ])
        
        def mock_open_handler(filename, *args, **kwargs):
            if "corpus.txt" in filename:
                return mock_open(read_data=corpus_data)()
            elif "qa.json" in filename:
                return mock_open(read_data=qa_json_data)()
        
        with patch("builtins.open", side_effect=mock_open_handler):
            with patch("os.path.exists", return_value=True):
                _, qa_pairs = build_corpus_and_qa()
                for qa in qa_pairs:
                    assert qa.answer == "Original answer"
                    assert qa.supporting_passages == ["p1", "p2"]

    def test_answer_truncation_to_500_chars(self):
        """Default QA answers should be truncated to 500 characters."""
        long_passage = "x" * 1000
        corpus_data = long_passage + "\n"
        with patch("builtins.open", mock_open(read_data=corpus_data)):
            with patch("os.path.exists", return_value=False):
                _, qa_pairs = build_corpus_and_qa()
                assert len(qa_pairs[0].answer) == 500

    def test_file_encoding_utf8(self):
        """Verify files are opened with UTF-8 encoding."""
        corpus_data = "passage with émojis 🎉\n"
        with patch("builtins.open", mock_open(read_data=corpus_data)) as m:
            with patch("os.path.exists", return_value=False):
                passages, _ = build_corpus_and_qa()
                # Check that open was called with utf-8 encoding
                m.assert_any_call("data/processed/corpus.txt", "r", encoding="utf-8")

    def test_empty_corpus_file(self):
        """Handle empty corpus file gracefully."""
        with patch("builtins.open", mock_open(read_data="")):
            with patch("os.path.exists", return_value=False):
                passages, qa_pairs = build_corpus_and_qa()
                assert passages == []
                assert qa_pairs == []

    def test_multiple_base_qa_all_augmented(self):
        """Multiple base QA pairs should all be augmented."""
        corpus_data = "passage1\n"
        qa_json_data = json.dumps([
            {"question": "Q1?", "answer": "A1"},
            {"question": "Q2?", "answer": "A2"}
        ])
        
        def mock_open_handler(filename, *args, **kwargs):
            if "corpus.txt" in filename:
                return mock_open(read_data=corpus_data)()
            elif "qa.json" in filename:
                return mock_open(read_data=qa_json_data)()
        
        with patch("builtins.open", side_effect=mock_open_handler):
            with patch("os.path.exists", return_value=True):
                _, qa_pairs = build_corpus_and_qa()
                # 2 base QA * 3 augmentations = 6
                assert len(qa_pairs) == 6