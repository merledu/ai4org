from unittest.mock import MagicMock, patch

import pytest

from hallucination_reduction.evaluation import (
    evaluate_classifier,
    evaluate_old_vs_new_generator,
    exact_match,
    f1_score,
    overlap_fact_check,
)


class TestExactMatch:
    """Test suite for exact_match function."""

    def test_function_exists(self):
        """Test that exact_match can be imported."""
        assert callable(exact_match)

    def test_identical_strings_return_one(self):
        """Test that identical strings return 1."""
        result = exact_match("hello", "hello")
        assert result == 1

    def test_different_strings_return_zero(self):
        """Test that different strings return 0."""
        result = exact_match("hello", "world")
        assert result == 0

    def test_case_insensitive_matching(self):
        """Test that matching is case insensitive."""
        result = exact_match("Hello", "hello")
        assert result == 1

        result = exact_match("HELLO", "hello")
        assert result == 1

    def test_strips_whitespace(self):
        """Test that whitespace is stripped before comparison."""
        result = exact_match("  hello  ", "hello")
        assert result == 1

        result = exact_match("hello", "  hello  ")
        assert result == 1

    def test_whitespace_and_case_combined(self):
        """Test that both whitespace stripping and case insensitivity work together."""
        result = exact_match("  Hello World  ", "hello world")
        assert result == 1

    def test_empty_strings_match(self):
        """Test that empty strings match."""
        result = exact_match("", "")
        assert result == 1

    def test_empty_vs_non_empty_no_match(self):
        """Test that empty and non-empty strings don't match."""
        result = exact_match("", "hello")
        assert result == 0

        result = exact_match("hello", "")
        assert result == 0

    def test_return_type_is_int(self):
        """Test that return type is int."""
        result = exact_match("hello", "hello")
        assert isinstance(result, int)


class TestF1Score:
    """Test suite for f1_score function."""

    def test_function_exists(self):
        """Test that f1_score can be imported."""
        assert callable(f1_score)

    def test_identical_strings_return_one(self):
        """Test that identical strings return 1.0."""
        result = f1_score("hello world", "hello world")
        assert result == 1.0

    def test_no_overlap_returns_zero(self):
        """Test that completely different strings return 0.0."""
        result = f1_score("hello world", "foo bar")
        assert result == 0.0

    def test_partial_overlap(self):
        """Test F1 score with partial overlap."""
        result = f1_score("hello world", "hello foo")
        assert result == 0.5

    def test_subset_prediction(self):
        """Test F1 when prediction is subset of gold."""
        result = f1_score("hello", "hello world")
        assert abs(result - 0.6666666666666666) < 0.001

    def test_superset_prediction(self):
        """Test F1 when prediction is superset of gold."""
        result = f1_score("hello world", "hello")
        assert abs(result - 0.6666666666666666) < 0.001

    def test_empty_prediction_returns_zero(self):
        """Test that empty prediction returns 0.0."""
        result = f1_score("", "hello world")
        assert result == 0.0

    def test_empty_gold_returns_zero(self):
        """Test that empty gold returns 0.0."""
        result = f1_score("hello world", "")
        assert result == 0.0

    def test_both_empty_returns_zero(self):
        """Test that both empty returns 0.0."""
        result = f1_score("", "")
        assert result == 0.0

    def test_repeated_tokens_counted_correctly(self):
        """Test that repeated tokens are counted correctly."""
        result = f1_score("hello hello", "hello")
        assert abs(result - 0.6666666666666666) < 0.001

    def test_return_type_is_float(self):
        """Test that return type is float."""
        result = f1_score("hello", "hello")
        assert isinstance(result, float)

    def test_symmetric_with_duplicate_tokens(self):
        """Test behavior with duplicate tokens."""
        result = f1_score("the the cat", "the cat cat")
        assert abs(result - 0.6666666666666666) < 0.001


class TestOverlapFactCheck:
    """Test suite for overlap_fact_check function."""

    def test_function_exists(self):
        """Test that overlap_fact_check can be imported."""
        assert callable(overlap_fact_check)

    def test_complete_overlap_returns_one(self):
        """Test that complete overlap returns 1.0."""
        answer = "hello world"
        passages = ["hello world"]
        result = overlap_fact_check(answer, passages)
        assert result == 1.0

    def test_no_overlap_returns_zero(self):
        """Test that no overlap returns 0.0."""
        answer = "hello world"
        passages = ["foo bar"]
        result = overlap_fact_check(answer, passages)
        assert result == 0.0

    def test_partial_overlap(self):
        """Test partial overlap calculation."""
        answer = "hello world"
        passages = ["hello foo"]
        result = overlap_fact_check(answer, passages)
        assert result == 0.5

    def test_multiple_passages_averaged(self):
        """Test that multiple passages are averaged."""
        answer = "hello world"
        passages = ["hello world", "hello foo"]
        result = overlap_fact_check(answer, passages)
        assert result == 0.75

    def test_empty_passages_returns_zero(self):
        """Test that empty passages list returns 0.0."""
        answer = "hello world"
        passages = []
        result = overlap_fact_check(answer, passages)
        assert result == 0.0

    def test_empty_answer_returns_zero(self):
        """Test that empty answer returns 0.0."""
        answer = ""
        passages = ["hello world"]
        result = overlap_fact_check(answer, passages)
        assert result == 0.0

    def test_empty_passage_returns_zero(self):
        """Test that empty passage returns 0.0 for that passage."""
        answer = "hello world"
        passages = [""]
        result = overlap_fact_check(answer, passages)
        assert result == 0.0

    def test_case_insensitive_matching(self):
        """Test that matching is case insensitive."""
        answer = "Hello World"
        passages = ["hello world"]
        result = overlap_fact_check(answer, passages)
        assert result == 1.0

    def test_superset_answer(self):
        """Test when answer is superset of passage."""
        answer = "hello world foo bar"
        passages = ["hello world"]
        result = overlap_fact_check(answer, passages)
        assert result == 1.0

    def test_subset_answer(self):
        """Test when answer is subset of passage."""
        answer = "hello"
        passages = ["hello world foo"]
        result = overlap_fact_check(answer, passages)
        assert abs(result - 0.3333333333333333) < 0.001

    def test_return_type_is_float(self):
        """Test that return type is float."""
        result = overlap_fact_check("hello", ["world"])
        assert isinstance(result, float)

    def test_return_value_in_range(self):
        """Test that return value is in [0, 1] range."""
        answer = "hello world"
        passages = ["foo bar baz"]
        result = overlap_fact_check(answer, passages)
        assert 0.0 <= result <= 1.0


class TestEvaluateOldVsNewGenerator:
    """Test suite for evaluate_old_vs_new_generator function."""

    @pytest.fixture
    def mock_qa_pair(self):
        """Create a mock QA pair."""
        qa = MagicMock()
        qa.question = "What is Python?"
        qa.answer = "A programming language"
        qa.supporting_passages = ["Python is a programming language"]
        return qa

    @pytest.fixture
    def mock_generators(self):
        """Create mock generator models."""
        old_gen = MagicMock()
        new_gen = MagicMock()
        return old_gen, new_gen

    @pytest.fixture
    def mock_tokenizer(self):
        """Create a mock tokenizer."""
        return MagicMock()

    @pytest.fixture
    def mock_retriever(self):
        """Create a mock retriever."""
        retriever = MagicMock()
        retriever.retrieve.return_value = [(0, "Retrieved passage")]
        return retriever

    @pytest.fixture
    def mock_fact_disc_and_tok(self):
        """Create mock fact discriminator and tokenizer."""
        fact_disc = MagicMock()
        fact_tok = MagicMock()
        return fact_disc, fact_tok

    def test_function_exists(self):
        """Test that evaluate_old_vs_new_generator can be imported."""
        assert callable(evaluate_old_vs_new_generator)

    def test_returns_three_items(
        self,
        mock_generators,
        mock_tokenizer,
        mock_retriever,
        mock_fact_disc_and_tok,
        mock_qa_pair,
    ):
        """Test that function returns three items."""
        old_gen, new_gen = mock_generators
        fact_disc, fact_tok = mock_fact_disc_and_tok

        with patch("hallucination_reduction.evaluation.generate_answer") as mock_gen:
            mock_gen.return_value = ["Generated answer"]

            with patch(
                "hallucination_reduction.evaluation.discriminator_predict_text"
            ) as mock_pred:
                mock_pred.return_value = [{"probs": [0.3, 0.7]}]

                result = evaluate_old_vs_new_generator(
                    old_gen,
                    new_gen,
                    mock_tokenizer,
                    mock_retriever,
                    [mock_qa_pair],
                    fact_disc,
                    fact_tok,
                )

                assert len(result) == 3

    def test_returns_rows_old_summary_new_summary(
        self,
        mock_generators,
        mock_tokenizer,
        mock_retriever,
        mock_fact_disc_and_tok,
        mock_qa_pair,
    ):
        """Test that function returns rows, old_summary, new_summary."""
        old_gen, new_gen = mock_generators
        fact_disc, fact_tok = mock_fact_disc_and_tok

        with patch("hallucination_reduction.evaluation.generate_answer") as mock_gen:
            mock_gen.return_value = ["Generated answer"]

            with patch(
                "hallucination_reduction.evaluation.discriminator_predict_text"
            ) as mock_pred:
                mock_pred.return_value = [{"probs": [0.3, 0.7]}]

                rows, old_summary, new_summary = evaluate_old_vs_new_generator(
                    old_gen,
                    new_gen,
                    mock_tokenizer,
                    mock_retriever,
                    [mock_qa_pair],
                    fact_disc,
                    fact_tok,
                )

                assert isinstance(rows, list)
                assert isinstance(old_summary, dict)
                assert isinstance(new_summary, dict)

    def test_summaries_contain_required_keys(
        self,
        mock_generators,
        mock_tokenizer,
        mock_retriever,
        mock_fact_disc_and_tok,
        mock_qa_pair,
    ):
        """Test that summaries contain required metric keys."""
        old_gen, new_gen = mock_generators
        fact_disc, fact_tok = mock_fact_disc_and_tok

        with patch("hallucination_reduction.evaluation.generate_answer") as mock_gen:
            mock_gen.return_value = ["Generated answer"]

            with patch(
                "hallucination_reduction.evaluation.discriminator_predict_text"
            ) as mock_pred:
                mock_pred.return_value = [{"probs": [0.3, 0.7]}]

                _, old_summary, new_summary = evaluate_old_vs_new_generator(
                    old_gen,
                    new_gen,
                    mock_tokenizer,
                    mock_retriever,
                    [mock_qa_pair],
                    fact_disc,
                    fact_tok,
                )

                assert "exact_match_rate" in old_summary
                assert "avg_f1" in old_summary
                assert "hallucination_rate" in old_summary

                assert "exact_match_rate" in new_summary
                assert "avg_f1" in new_summary
                assert "hallucination_rate" in new_summary

    def test_rows_contain_question_gold_old_new(
        self,
        mock_generators,
        mock_tokenizer,
        mock_retriever,
        mock_fact_disc_and_tok,
        mock_qa_pair,
    ):
        """Test that rows contain question, gold, old, new keys."""
        old_gen, new_gen = mock_generators
        fact_disc, fact_tok = mock_fact_disc_and_tok

        with patch("hallucination_reduction.evaluation.generate_answer") as mock_gen:
            mock_gen.return_value = ["Generated answer"]

            with patch(
                "hallucination_reduction.evaluation.discriminator_predict_text"
            ) as mock_pred:
                mock_pred.return_value = [{"probs": [0.3, 0.7]}]

                rows, _, _ = evaluate_old_vs_new_generator(
                    old_gen,
                    new_gen,
                    mock_tokenizer,
                    mock_retriever,
                    [mock_qa_pair],
                    fact_disc,
                    fact_tok,
                )

                assert len(rows) > 0
                row = rows[0]
                assert "question" in row
                assert "gold" in row
                assert "old" in row
                assert "new" in row

    def test_retriever_called_for_each_qa(
        self,
        mock_generators,
        mock_tokenizer,
        mock_retriever,
        mock_fact_disc_and_tok,
        mock_qa_pair,
    ):
        """Test that retriever is called for each QA pair."""
        old_gen, new_gen = mock_generators
        fact_disc, fact_tok = mock_fact_disc_and_tok

        qa_pairs = [mock_qa_pair, mock_qa_pair, mock_qa_pair]

        with patch("hallucination_reduction.evaluation.generate_answer") as mock_gen:
            mock_gen.return_value = ["Generated answer"]

            with patch(
                "hallucination_reduction.evaluation.discriminator_predict_text"
            ) as mock_pred:
                mock_pred.return_value = [{"probs": [0.3, 0.7]}]

                evaluate_old_vs_new_generator(
                    old_gen,
                    new_gen,
                    mock_tokenizer,
                    mock_retriever,
                    qa_pairs,
                    fact_disc,
                    fact_tok,
                )

                assert mock_retriever.retrieve.call_count == len(qa_pairs)

    def test_generate_answer_called_twice_per_qa(
        self,
        mock_generators,
        mock_tokenizer,
        mock_retriever,
        mock_fact_disc_and_tok,
        mock_qa_pair,
    ):
        """Test that generate_answer is called twice per QA (old and new)."""
        old_gen, new_gen = mock_generators
        fact_disc, fact_tok = mock_fact_disc_and_tok

        with patch("hallucination_reduction.evaluation.generate_answer") as mock_gen:
            mock_gen.return_value = ["Generated answer"]

            with patch(
                "hallucination_reduction.evaluation.discriminator_predict_text"
            ) as mock_pred:
                mock_pred.return_value = [{"probs": [0.3, 0.7]}]

                evaluate_old_vs_new_generator(
                    old_gen,
                    new_gen,
                    mock_tokenizer,
                    mock_retriever,
                    [mock_qa_pair],
                    fact_disc,
                    fact_tok,
                )

                # Should be called twice: once for old_gen, once for new_gen
                assert mock_gen.call_count == 2

    def test_metrics_rates_between_zero_and_one(
        self,
        mock_generators,
        mock_tokenizer,
        mock_retriever,
        mock_fact_disc_and_tok,
        mock_qa_pair,
    ):
        """Test that all metric rates are between 0 and 1."""
        old_gen, new_gen = mock_generators
        fact_disc, fact_tok = mock_fact_disc_and_tok

        with patch("hallucination_reduction.evaluation.generate_answer") as mock_gen:
            mock_gen.return_value = ["Generated answer"]

            with patch(
                "hallucination_reduction.evaluation.discriminator_predict_text"
            ) as mock_pred:
                mock_pred.return_value = [{"probs": [0.3, 0.7]}]

                _, old_summary, new_summary = evaluate_old_vs_new_generator(
                    old_gen,
                    new_gen,
                    mock_tokenizer,
                    mock_retriever,
                    [mock_qa_pair],
                    fact_disc,
                    fact_tok,
                )

                for summary in [old_summary, new_summary]:
                    assert 0.0 <= summary["exact_match_rate"] <= 1.0
                    assert 0.0 <= summary["avg_f1"] <= 1.0
                    assert 0.0 <= summary["hallucination_rate"] <= 1.0


class TestEvaluateClassifier:
    """Test suite for evaluate_classifier function (from evaluation)."""

    @pytest.fixture
    def mock_classifier(self):
        """Create a mock classifier."""
        return MagicMock()

    @pytest.fixture
    def mock_tokenizer(self):
        """Create a mock tokenizer."""
        return MagicMock()

    def test_function_exists(self):
        """Test that evaluate_classifier can be imported."""
        assert callable(evaluate_classifier)
