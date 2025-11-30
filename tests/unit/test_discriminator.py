import pytest
import torch
import numpy as np
from unittest.mock import Mock, MagicMock, patch, call
from transformers import AutoTokenizer, AutoModelForSequenceClassification


class TestLoadDiscriminator:
    """Test suite for load_discriminator function."""

    def test_function_exists(self):
        """Test that load_discriminator can be imported."""
        from hallucination_reduction.discriminator import load_discriminator
        assert callable(load_discriminator)

    @patch('hallucination_reduction.discriminator.AutoModelForSequenceClassification.from_pretrained')
    @patch('hallucination_reduction.discriminator.AutoTokenizer.from_pretrained')
    def test_returns_tokenizer_and_model(self, mock_tokenizer, mock_model):
        """Test that function returns tokenizer and model."""
        from hallucination_reduction.discriminator import load_discriminator
        
        mock_tok_instance = MagicMock()
        mock_model_instance = MagicMock()
        mock_model_instance.to.return_value = mock_model_instance
        
        mock_tokenizer.return_value = mock_tok_instance
        mock_model.return_value = mock_model_instance
        
        tokenizer, model = load_discriminator()
        
        assert tokenizer is not None
        assert model is not None

    @patch('hallucination_reduction.discriminator.AutoModelForSequenceClassification.from_pretrained')
    @patch('hallucination_reduction.discriminator.AutoTokenizer.from_pretrained')
    def test_uses_default_model_name(self, mock_tokenizer, mock_model):
        """Test that default model name is used."""
        from hallucination_reduction.discriminator import load_discriminator, DISC_MODEL
        
        mock_tok_instance = MagicMock()
        mock_model_instance = MagicMock()
        mock_model_instance.to.return_value = mock_model_instance
        
        mock_tokenizer.return_value = mock_tok_instance
        mock_model.return_value = mock_model_instance
        
        load_discriminator()
        
        mock_tokenizer.assert_called_once_with(DISC_MODEL)
        mock_model.assert_called_once()

    @patch('hallucination_reduction.discriminator.AutoModelForSequenceClassification.from_pretrained')
    @patch('hallucination_reduction.discriminator.AutoTokenizer.from_pretrained')
    def test_uses_custom_model_name(self, mock_tokenizer, mock_model):
        """Test that custom model name is used when provided."""
        from hallucination_reduction.discriminator import load_discriminator
        
        custom_model = "bert-base-uncased"
        mock_tok_instance = MagicMock()
        mock_model_instance = MagicMock()
        mock_model_instance.to.return_value = mock_model_instance
        
        mock_tokenizer.return_value = mock_tok_instance
        mock_model.return_value = mock_model_instance
        
        load_discriminator(model_name=custom_model)
        
        mock_tokenizer.assert_called_once_with(custom_model)

    @patch('hallucination_reduction.discriminator.AutoModelForSequenceClassification.from_pretrained')
    @patch('hallucination_reduction.discriminator.AutoTokenizer.from_pretrained')
    def test_model_moved_to_device(self, mock_tokenizer, mock_model):
        """Test that model is moved to the specified device."""
        from hallucination_reduction.discriminator import load_discriminator
        
        mock_tok_instance = MagicMock()
        mock_model_instance = MagicMock()
        mock_model_instance.to.return_value = mock_model_instance
        
        mock_tokenizer.return_value = mock_tok_instance
        mock_model.return_value = mock_model_instance
        
        custom_device = "cuda:0"
        load_discriminator(device=custom_device)
        
        mock_model_instance.to.assert_called_once_with(custom_device)

    @patch('hallucination_reduction.discriminator.AutoModelForSequenceClassification.from_pretrained')
    @patch('hallucination_reduction.discriminator.AutoTokenizer.from_pretrained')
    def test_model_set_to_eval_mode(self, mock_tokenizer, mock_model):
        """Test that model is set to eval mode."""
        from hallucination_reduction.discriminator import load_discriminator
        
        mock_tok_instance = MagicMock()
        mock_model_instance = MagicMock()
        mock_model_instance.to.return_value = mock_model_instance
        
        mock_tokenizer.return_value = mock_tok_instance
        mock_model.return_value = mock_model_instance
        
        _, model = load_discriminator()
        
        mock_model_instance.eval.assert_called_once()

    @patch('hallucination_reduction.discriminator.AutoModelForSequenceClassification.from_pretrained')
    @patch('hallucination_reduction.discriminator.AutoTokenizer.from_pretrained')
    def test_num_labels_parameter(self, mock_tokenizer, mock_model):
        """Test that num_labels parameter is passed correctly."""
        from hallucination_reduction.discriminator import load_discriminator
        
        mock_tok_instance = MagicMock()
        mock_model_instance = MagicMock()
        mock_model_instance.to.return_value = mock_model_instance
        
        mock_tokenizer.return_value = mock_tok_instance
        mock_model.return_value = mock_model_instance
        
        custom_labels = 3
        load_discriminator(num_labels=custom_labels)
        
        call_args = mock_model.call_args
        assert call_args[1]['num_labels'] == custom_labels

    @patch('hallucination_reduction.discriminator.AutoModelForSequenceClassification.from_pretrained')
    @patch('hallucination_reduction.discriminator.AutoTokenizer.from_pretrained')
    def test_default_num_labels_is_two(self, mock_tokenizer, mock_model):
        """Test that default num_labels is 2."""
        from hallucination_reduction.discriminator import load_discriminator
        
        mock_tok_instance = MagicMock()
        mock_model_instance = MagicMock()
        mock_model_instance.to.return_value = mock_model_instance
        
        mock_tokenizer.return_value = mock_tok_instance
        mock_model.return_value = mock_model_instance
        
        load_discriminator()
        
        call_args = mock_model.call_args
        assert call_args[1]['num_labels'] == 2


class TestDiscriminatorPredictText:
    """Test suite for discriminator_predict_text function."""

    @pytest.fixture
    def mock_classifier(self):
        """Create a mock classifier."""
        classifier = MagicMock()
        
        def mock_forward(**kwargs):
            # Determine batch size from input
            batch_size = kwargs.get('input_ids', torch.tensor([[1]])).shape[0]
            mock_output = MagicMock()
            mock_output.logits = torch.randn(batch_size, 2)  # Dynamic batch_size, num_classes=2
            return mock_output
        
        classifier.side_effect = mock_forward
        classifier.eval = MagicMock()
        return classifier

    @pytest.fixture
    def mock_tokenizer(self):
        """Create a mock tokenizer."""
        tokenizer = MagicMock()
        mock_encoding = {
            'input_ids': torch.randint(0, 1000, (2, 10)),
            'attention_mask': torch.ones(2, 10)
        }
        mock_encoding_obj = MagicMock()
        mock_encoding_obj.to.return_value = mock_encoding
        tokenizer.return_value = mock_encoding_obj
        return tokenizer

    def test_function_exists(self):
        """Test that discriminator_predict_text can be imported."""
        from hallucination_reduction.discriminator import discriminator_predict_text
        assert callable(discriminator_predict_text)

    def test_returns_list(self, mock_classifier, mock_tokenizer):
        """Test that function returns a list."""
        from hallucination_reduction.discriminator import discriminator_predict_text
        
        texts = ["text1", "text2"]
        result = discriminator_predict_text(mock_classifier, mock_tokenizer, texts)
        
        assert isinstance(result, list)


    def test_result_contains_pred_and_probs(self, mock_classifier, mock_tokenizer):
        """Test that each result contains 'pred' and 'probs' keys."""
        from hallucination_reduction.discriminator import discriminator_predict_text
        
        texts = ["text1"]
        result = discriminator_predict_text(mock_classifier, mock_tokenizer, texts)
        
        assert "pred" in result[0]
        assert "probs" in result[0]

    def test_pred_is_integer(self, mock_classifier, mock_tokenizer):
        """Test that prediction is an integer."""
        from hallucination_reduction.discriminator import discriminator_predict_text
        
        texts = ["text1"]
        result = discriminator_predict_text(mock_classifier, mock_tokenizer, texts)
        
        assert isinstance(result[0]["pred"], int)

    def test_probs_is_list(self, mock_classifier, mock_tokenizer):
        """Test that probabilities is a list."""
        from hallucination_reduction.discriminator import discriminator_predict_text
        
        texts = ["text1"]
        result = discriminator_predict_text(mock_classifier, mock_tokenizer, texts)
        
        assert isinstance(result[0]["probs"], list)

    def test_classifier_set_to_eval_mode(self, mock_classifier, mock_tokenizer):
        """Test that classifier is set to eval mode."""
        from hallucination_reduction.discriminator import discriminator_predict_text
        
        texts = ["text1"]
        discriminator_predict_text(mock_classifier, mock_tokenizer, texts)
        
        mock_classifier.eval.assert_called()

    def test_tokenizer_called_with_texts(self, mock_classifier, mock_tokenizer):
        """Test that tokenizer is called with input texts."""
        from hallucination_reduction.discriminator import discriminator_predict_text
        
        texts = ["text1", "text2"]
        discriminator_predict_text(mock_classifier, mock_tokenizer, texts)
        
        # Tokenizer should be called at least once
        assert mock_tokenizer.call_count >= 1

    def test_tokenizer_uses_padding_and_truncation(self, mock_classifier, mock_tokenizer):
        """Test that tokenizer uses padding and truncation."""
        from hallucination_reduction.discriminator import discriminator_predict_text
        
        texts = ["text1"]
        discriminator_predict_text(mock_classifier, mock_tokenizer, texts)
        
        call_kwargs = mock_tokenizer.call_args[1]
        assert call_kwargs.get('padding') is True
        assert call_kwargs.get('truncation') is True

    def test_handles_empty_text_list(self, mock_classifier, mock_tokenizer):
        """Test that function handles empty text list."""
        from hallucination_reduction.discriminator import discriminator_predict_text
        
        texts = []
        result = discriminator_predict_text(mock_classifier, mock_tokenizer, texts)
        
        assert result == []

    def test_batch_size_parameter(self, mock_classifier, mock_tokenizer):
        """Test that batch_size parameter affects processing."""
        from hallucination_reduction.discriminator import discriminator_predict_text
        
        texts = ["text1", "text2", "text3", "text4"]
        batch_size = 2
        
        result = discriminator_predict_text(
            mock_classifier, 
            mock_tokenizer, 
            texts, 
            batch_size=batch_size
        )
        
        # Should return results for all texts
        assert len(result) == len(texts)

    def test_uses_no_grad_context(self, mock_classifier, mock_tokenizer):
        """Test that predictions are made with torch.no_grad()."""
        from hallucination_reduction.discriminator import discriminator_predict_text
        
        texts = ["text1"]
        
        with patch('torch.no_grad') as mock_no_grad:
            mock_no_grad.return_value.__enter__ = Mock()
            mock_no_grad.return_value.__exit__ = Mock()
            
            discriminator_predict_text(mock_classifier, mock_tokenizer, texts)
            
            # no_grad should be called
            mock_no_grad.assert_called()


class TestSimpleTextDataset:
    """Test suite for SimpleTextDataset class."""

    @pytest.fixture
    def mock_tokenizer(self):
        """Create a mock tokenizer."""
        tokenizer = MagicMock()
        tokenizer.return_value = {
            'input_ids': torch.tensor([[1, 2, 3]]),
            'attention_mask': torch.tensor([[1, 1, 1]])
        }
        return tokenizer

    @pytest.fixture
    def sample_data(self):
        """Create sample data."""
        texts = ["text1", "text2", "text3"]
        labels = [0, 1, 0]
        return texts, labels

    def test_class_exists(self):
        """Test that SimpleTextDataset class can be imported."""
        from hallucination_reduction.discriminator import SimpleTextDataset
        assert SimpleTextDataset is not None

    def test_initialization(self, mock_tokenizer, sample_data):
        """Test dataset initialization."""
        from hallucination_reduction.discriminator import SimpleTextDataset
        
        texts, labels = sample_data
        dataset = SimpleTextDataset(texts, labels, mock_tokenizer)
        
        assert dataset is not None
        assert dataset.texts == texts
        assert dataset.labels == labels

    def test_len_returns_correct_length(self, mock_tokenizer, sample_data):
        """Test that __len__ returns correct length."""
        from hallucination_reduction.discriminator import SimpleTextDataset
        
        texts, labels = sample_data
        dataset = SimpleTextDataset(texts, labels, mock_tokenizer)
        
        assert len(dataset) == len(texts)

    def test_getitem_returns_dict(self, mock_tokenizer, sample_data):
        """Test that __getitem__ returns a dictionary."""
        from hallucination_reduction.discriminator import SimpleTextDataset
        
        texts, labels = sample_data
        dataset = SimpleTextDataset(texts, labels, mock_tokenizer)
        
        item = dataset[0]
        assert isinstance(item, dict)

    def test_getitem_contains_labels(self, mock_tokenizer, sample_data):
        """Test that returned item contains 'labels' key."""
        from hallucination_reduction.discriminator import SimpleTextDataset
        
        texts, labels = sample_data
        dataset = SimpleTextDataset(texts, labels, mock_tokenizer)
        
        item = dataset[0]
        assert 'labels' in item

    def test_labels_are_tensors(self, mock_tokenizer, sample_data):
        """Test that labels are PyTorch tensors."""
        from hallucination_reduction.discriminator import SimpleTextDataset
        
        texts, labels = sample_data
        dataset = SimpleTextDataset(texts, labels, mock_tokenizer)
        
        item = dataset[0]
        assert isinstance(item['labels'], torch.Tensor)

    def test_labels_have_correct_dtype(self, mock_tokenizer, sample_data):
        """Test that labels have dtype long."""
        from hallucination_reduction.discriminator import SimpleTextDataset
        
        texts, labels = sample_data
        dataset = SimpleTextDataset(texts, labels, mock_tokenizer)
        
        item = dataset[0]
        assert item['labels'].dtype == torch.long

    def test_correct_label_returned(self, mock_tokenizer, sample_data):
        """Test that correct label is returned for each index."""
        from hallucination_reduction.discriminator import SimpleTextDataset
        
        texts, labels = sample_data
        dataset = SimpleTextDataset(texts, labels, mock_tokenizer)
        
        for idx in range(len(labels)):
            item = dataset[idx]
            assert item['labels'].item() == labels[idx]

    def test_tokenizer_called_with_correct_text(self, mock_tokenizer, sample_data):
        """Test that tokenizer is called with correct text."""
        from hallucination_reduction.discriminator import SimpleTextDataset
        
        texts, labels = sample_data
        dataset = SimpleTextDataset(texts, labels, mock_tokenizer)
        
        _ = dataset[1]
        
        # Check that tokenizer was called with the second text
        call_args = mock_tokenizer.call_args[0]
        assert call_args[0] == texts[1]

    def test_tokenizer_uses_truncation(self, mock_tokenizer, sample_data):
        """Test that tokenizer uses truncation."""
        from hallucination_reduction.discriminator import SimpleTextDataset
        
        texts, labels = sample_data
        dataset = SimpleTextDataset(texts, labels, mock_tokenizer)
        
        _ = dataset[0]
        
        call_kwargs = mock_tokenizer.call_args[1]
        assert call_kwargs.get('truncation') is True

    def test_tokenizer_uses_padding(self, mock_tokenizer, sample_data):
        """Test that tokenizer uses padding."""
        from hallucination_reduction.discriminator import SimpleTextDataset
        
        texts, labels = sample_data
        dataset = SimpleTextDataset(texts, labels, mock_tokenizer)
        
        _ = dataset[0]
        
        call_kwargs = mock_tokenizer.call_args[1]
        assert call_kwargs.get('padding') == "max_length"

    def test_custom_max_length(self, mock_tokenizer, sample_data):
        """Test that custom max_length is used."""
        from hallucination_reduction.discriminator import SimpleTextDataset
        
        texts, labels = sample_data
        custom_max_length = 128
        dataset = SimpleTextDataset(texts, labels, mock_tokenizer, max_length=custom_max_length)
        
        _ = dataset[0]
        
        call_kwargs = mock_tokenizer.call_args[1]
        assert call_kwargs.get('max_length') == custom_max_length

    def test_default_max_length_is_256(self, mock_tokenizer, sample_data):
        """Test that default max_length is 256."""
        from hallucination_reduction.discriminator import SimpleTextDataset
        
        texts, labels = sample_data
        dataset = SimpleTextDataset(texts, labels, mock_tokenizer)
        
        assert dataset.max_length == 256

    def test_all_indices_accessible(self, mock_tokenizer, sample_data):
        """Test that all indices can be accessed."""
        from hallucination_reduction.discriminator import SimpleTextDataset
        
        texts, labels = sample_data
        dataset = SimpleTextDataset(texts, labels, mock_tokenizer)
        
        for idx in range(len(dataset)):
            item = dataset[idx]
            assert item is not None

    def test_dataset_is_iterable(self, mock_tokenizer, sample_data):
        """Test that dataset can be used with DataLoader."""
        from hallucination_reduction.discriminator import SimpleTextDataset
        from torch.utils.data import DataLoader
        
        texts, labels = sample_data
        dataset = SimpleTextDataset(texts, labels, mock_tokenizer)
        
        # Should be able to create a DataLoader
        dataloader = DataLoader(dataset, batch_size=2)
        assert dataloader is not None