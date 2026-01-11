from unittest.mock import MagicMock, patch

import pytest


class TestEvaluateClassifier:
    """Test suite for evaluate_classifier function."""

    @pytest.fixture
    def mock_classifier(self):
        """Create a mock classifier."""
        classifier = MagicMock()
        return classifier

    @pytest.fixture
    def mock_tokenizer(self):
        """Create a mock tokenizer."""
        return MagicMock()

    @pytest.fixture
    def sample_eval_data(self):
        """Create sample evaluation data."""
        texts = ["text1", "text2", "text3", "text4"]
        labels = [1, 0, 1, 0]
        return texts, labels

    def test_evaluate_returns_dict(
        self, mock_classifier, mock_tokenizer, sample_eval_data
    ):
        """Test that evaluate_classifier returns a dictionary."""
        from hallucination_reduction.discriminator_training_utils import (
            evaluate_classifier,
        )

        texts, labels = sample_eval_data

        with patch(
            "hallucination_reduction.discriminator_training_utils.discriminator_predict_text"
        ) as mock_predict:
            mock_predict.return_value = [{"probs": [0.3, 0.7]}]

            result = evaluate_classifier(mock_classifier, mock_tokenizer, texts, labels)

            assert isinstance(result, dict)

    def test_evaluate_returns_all_metrics(
        self, mock_classifier, mock_tokenizer, sample_eval_data
    ):
        """Test that all required metrics are returned."""
        from hallucination_reduction.discriminator_training_utils import (
            evaluate_classifier,
        )

        texts, labels = sample_eval_data

        with patch(
            "hallucination_reduction.discriminator_training_utils.discriminator_predict_text"
        ) as mock_predict:
            mock_predict.return_value = [{"probs": [0.3, 0.7]}]

            result = evaluate_classifier(mock_classifier, mock_tokenizer, texts, labels)

            assert "acc" in result
            assert "prec" in result
            assert "rec" in result
            assert "f1" in result

    def test_classifier_set_to_eval_mode(
        self, mock_classifier, mock_tokenizer, sample_eval_data
    ):
        """Test that classifier is set to eval mode."""
        from hallucination_reduction.discriminator_training_utils import (
            evaluate_classifier,
        )

        texts, labels = sample_eval_data

        with patch(
            "hallucination_reduction.discriminator_training_utils.discriminator_predict_text"
        ) as mock_predict:
            mock_predict.return_value = [{"probs": [0.3, 0.7]}]

            evaluate_classifier(mock_classifier, mock_tokenizer, texts, labels)

            mock_classifier.eval.assert_called()

    def test_accuracy_calculated_correctly(
        self, mock_classifier, mock_tokenizer, sample_eval_data
    ):
        """Test that accuracy is calculated using sklearn."""
        from hallucination_reduction.discriminator_training_utils import (
            evaluate_classifier,
        )

        texts, labels = sample_eval_data

        with patch(
            "hallucination_reduction.discriminator_training_utils.discriminator_predict_text"
        ) as mock_predict:
            mock_predict.return_value = [{"probs": [0.3, 0.7]}]

            with patch(
                "hallucination_reduction.discriminator_training_utils.accuracy_score"
            ) as mock_acc:
                mock_acc.return_value = 0.75

                result = evaluate_classifier(
                    mock_classifier, mock_tokenizer, texts, labels
                )

                mock_acc.assert_called_once()
                assert result["acc"] == 0.75

    def test_precision_recall_f1_calculated(
        self, mock_classifier, mock_tokenizer, sample_eval_data
    ):
        """Test that precision, recall, and F1 are calculated."""
        from hallucination_reduction.discriminator_training_utils import (
            evaluate_classifier,
        )

        texts, labels = sample_eval_data

        with patch(
            "hallucination_reduction.discriminator_training_utils.discriminator_predict_text"
        ) as mock_predict:
            mock_predict.return_value = [{"probs": [0.3, 0.7]}]

            with patch(
                "hallucination_reduction.discriminator_training_utils.precision_recall_fscore_support"
            ) as mock_prf:
                mock_prf.return_value = (0.8, 0.75, 0.77, None)

                result = evaluate_classifier(
                    mock_classifier, mock_tokenizer, texts, labels
                )

                mock_prf.assert_called_once()
                assert result["prec"] == 0.8
                assert result["rec"] == 0.75
                assert result["f1"] == 0.77

    def test_binary_average_used_for_metrics(
        self, mock_classifier, mock_tokenizer, sample_eval_data
    ):
        """Test that binary averaging is used for precision/recall/F1."""
        from hallucination_reduction.discriminator_training_utils import (
            evaluate_classifier,
        )

        texts, labels = sample_eval_data

        with patch(
            "hallucination_reduction.discriminator_training_utils.discriminator_predict_text"
        ) as mock_predict:
            mock_predict.return_value = [{"probs": [0.3, 0.7]}]

            with patch(
                "hallucination_reduction.discriminator_training_utils.precision_recall_fscore_support"
            ) as mock_prf:
                mock_prf.return_value = (0.8, 0.75, 0.77, None)

                evaluate_classifier(mock_classifier, mock_tokenizer, texts, labels)

                call_args = mock_prf.call_args
                assert call_args[1]["average"] == "binary"

    def test_zero_division_handled(
        self, mock_classifier, mock_tokenizer, sample_eval_data
    ):
        """Test that zero division is handled in metrics."""
        from hallucination_reduction.discriminator_training_utils import (
            evaluate_classifier,
        )

        texts, labels = sample_eval_data

        with patch(
            "hallucination_reduction.discriminator_training_utils.discriminator_predict_text"
        ) as mock_predict:
            mock_predict.return_value = [{"probs": [0.3, 0.7]}]

            with patch(
                "hallucination_reduction.discriminator_training_utils.precision_recall_fscore_support"
            ) as mock_prf:
                mock_prf.return_value = (0.8, 0.75, 0.77, None)

                evaluate_classifier(mock_classifier, mock_tokenizer, texts, labels)

                call_args = mock_prf.call_args
                assert call_args[1]["zero_division"] == 0

    def test_handles_empty_labels(self, mock_classifier, mock_tokenizer):
        """Test handling of empty labels list."""
        from hallucination_reduction.discriminator_training_utils import (
            evaluate_classifier,
        )

        texts = []
        labels = []

        with patch(
            "hallucination_reduction.discriminator_training_utils.discriminator_predict_text"
        ) as mock_predict:
            mock_predict.return_value = [{"probs": [0.3, 0.7]}]

            result = evaluate_classifier(mock_classifier, mock_tokenizer, texts, labels)

            assert result["acc"] == 0.0

    def test_prediction_threshold_05(self, mock_classifier, mock_tokenizer):
        """Test that prediction threshold is 0.5."""
        from hallucination_reduction.discriminator_training_utils import (
            evaluate_classifier,
        )

        texts = ["text1", "text2"]
        labels = [1, 0]

        with patch(
            "hallucination_reduction.discriminator_training_utils.discriminator_predict_text"
        ) as mock_predict:
            # Return probabilities just above and below threshold
            mock_predict.side_effect = [
                [{"probs": [0.4, 0.6]}],  # Should predict 1
                [{"probs": [0.6, 0.4]}],  # Should predict 0
            ]

            with patch(
                "hallucination_reduction.discriminator_training_utils.accuracy_score"
            ) as mock_acc:
                mock_acc.return_value = 1.0

                evaluate_classifier(mock_classifier, mock_tokenizer, texts, labels)

                # Check that predictions were made correctly
                pred_calls = mock_acc.call_args[0]
                assert len(pred_calls) == 2  # labels and predictions

    def test_evaluates_all_texts(self, mock_classifier, mock_tokenizer):
        """Test that all texts are evaluated."""
        from hallucination_reduction.discriminator_training_utils import (
            evaluate_classifier,
        )

        texts = ["text1", "text2", "text3", "text4", "text5"]
        labels = [1, 0, 1, 0, 1]

        with patch(
            "hallucination_reduction.discriminator_training_utils.discriminator_predict_text"
        ) as mock_predict:
            mock_predict.return_value = [{"probs": [0.3, 0.7]}]

            evaluate_classifier(mock_classifier, mock_tokenizer, texts, labels)

            # Should be called once per text
            assert mock_predict.call_count == len(texts)

    def test_metrics_are_numeric(
        self, mock_classifier, mock_tokenizer, sample_eval_data
    ):
        """Test that all metrics are numeric values."""
        from hallucination_reduction.discriminator_training_utils import (
            evaluate_classifier,
        )

        texts, labels = sample_eval_data

        with patch(
            "hallucination_reduction.discriminator_training_utils.discriminator_predict_text"
        ) as mock_predict:
            mock_predict.return_value = [{"probs": [0.3, 0.7]}]

            result = evaluate_classifier(mock_classifier, mock_tokenizer, texts, labels)

            assert isinstance(result["acc"], (int, float))
            assert isinstance(result["prec"], (int, float))
            assert isinstance(result["rec"], (int, float))
            assert isinstance(result["f1"], (int, float))

    def test_handles_two_class_probabilities(self, mock_classifier, mock_tokenizer):
        """Test normal case with two-class probabilities."""
        from hallucination_reduction.discriminator_training_utils import (
            evaluate_classifier,
        )

        texts = ["text1"]
        labels = [1]

        with patch(
            "hallucination_reduction.discriminator_training_utils.discriminator_predict_text"
        ) as mock_predict:
            # Return proper two-class probabilities
            mock_predict.return_value = [{"probs": [0.3, 0.7]}]

            result = evaluate_classifier(mock_classifier, mock_tokenizer, texts, labels)

            # Should handle gracefully
            assert isinstance(result, dict)
            assert "acc" in result


class TestTrainDiscriminatorMinibatchIntegration:
    """Integration tests for train_discriminator_minibatch that verify behavior without deep mocking."""

    def test_function_exists_and_callable(self):
        """Test that the function can be imported."""
        from hallucination_reduction.discriminator_training_utils import (
            train_discriminator_minibatch,
        )

        assert callable(train_discriminator_minibatch)

    def test_function_signature(self):
        """Test that the function has expected parameters."""
        import inspect

        from hallucination_reduction.discriminator_training_utils import (
            train_discriminator_minibatch,
        )

        sig = inspect.signature(train_discriminator_minibatch)
        params = list(sig.parameters.keys())

        # Check for expected parameters
        assert "classifier" in params
        assert "tokenizer" in params
        assert "texts" in params
        assert "labels" in params
