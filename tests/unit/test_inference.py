import unittest
from unittest.mock import MagicMock, Mock, mock_open, patch

import numpy as np
import torch

from hallucination_reduction.inference import (
    build_embeddings,
    find_best_model,
    generate_answer,
    load_corpus,
    load_model,
    retrieve_relevant_chunks,
)


class TestLoadCorpus(unittest.TestCase):

    @patch(
        "builtins.open",
        new_callable=mock_open,
        read_data="Document 1\nDocument 2\nDocument 3\n",
    )
    @patch("os.path.exists")
    def test_load_corpus_success(self, mock_exists, mock_file):
        """Test successfully loading corpus from file"""
        mock_exists.return_value = True

        docs = load_corpus("test_corpus.txt")

        self.assertEqual(len(docs), 3)
        self.assertEqual(docs[0], "Document 1")
        self.assertEqual(docs[1], "Document 2")
        self.assertEqual(docs[2], "Document 3")
        mock_file.assert_called_once_with("test_corpus.txt", "r", encoding="utf-8")

    @patch(
        "builtins.open", new_callable=mock_open, read_data="Doc 1\n\n\nDoc 2\n  \nDoc 3"
    )
    @patch("os.path.exists")
    def test_load_corpus_filters_empty_lines(self, mock_exists, mock_file):
        """Test that empty lines and whitespace are filtered"""
        mock_exists.return_value = True

        docs = load_corpus("test_corpus.txt")

        self.assertEqual(len(docs), 3)
        self.assertNotIn("", docs)
        self.assertNotIn("  ", docs)

    @patch("os.path.exists")
    def test_load_corpus_file_not_found(self, mock_exists):
        """Test error when corpus file doesn't exist"""
        mock_exists.return_value = False

        with self.assertRaises(FileNotFoundError) as context:
            load_corpus("nonexistent.txt")

        self.assertIn("Corpus file not found", str(context.exception))


class TestBuildEmbeddings(unittest.TestCase):

    @patch("hallucination_reduction.inference.SentenceTransformer")
    def test_build_embeddings(self, mock_sentence_transformer_class):
        """Test building embeddings from documents"""
        mock_embedder = Mock()
        mock_embeddings = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
        mock_embedder.encode.return_value = mock_embeddings
        mock_sentence_transformer_class.return_value = mock_embedder

        docs = ["Document 1", "Document 2"]
        embedder, embeddings = build_embeddings(
            docs, embed_model_name="test-model", device="cpu"
        )

        mock_sentence_transformer_class.assert_called_once_with(
            "test-model", device="cpu"
        )
        mock_embedder.encode.assert_called_once_with(
            docs, convert_to_numpy=True, show_progress_bar=True
        )
        self.assertEqual(embedder, mock_embedder)
        np.testing.assert_array_equal(embeddings, mock_embeddings)


class TestRetrieveRelevantChunks(unittest.TestCase):

    def test_retrieve_relevant_chunks(self):
        """Test retrieving top-k relevant chunks"""
        mock_embedder = Mock()
        query_embedding = np.array([[1.0, 0.0]])
        mock_embedder.encode.return_value = query_embedding

        # Corpus embeddings - doc 2 is most similar to [1.0, 0.0]
        corpus_embeddings = np.array(
            [
                [0.0, 1.0],  # Not similar (perpendicular)
                [0.5, 0.5],  # Somewhat similar
                [1.0, 0.0],  # Most similar (identical)
            ]
        )

        docs = ["Doc 0", "Doc 1", "Doc 2"]

        top_docs = retrieve_relevant_chunks(
            "test query", mock_embedder, corpus_embeddings, docs, top_k=2
        )

        self.assertEqual(len(top_docs), 2)
        self.assertEqual(top_docs[0], "Doc 2")  # Most similar
        self.assertEqual(top_docs[1], "Doc 1")  # Second most similar
        mock_embedder.encode.assert_called_once_with(
            ["test query"], convert_to_numpy=True
        )

    def test_retrieve_single_chunk(self):
        """Test retrieving single most relevant chunk"""
        mock_embedder = Mock()
        mock_embedder.encode.return_value = np.array([[1.0, 0.0]])

        corpus_embeddings = np.array(
            [
                [0.0, 1.0],
                [1.0, 0.0],
                [0.5, 0.5],
            ]
        )

        docs = ["Doc A", "Doc B", "Doc C"]

        top_docs = retrieve_relevant_chunks(
            "query", mock_embedder, corpus_embeddings, docs, top_k=1
        )

        self.assertEqual(len(top_docs), 1)
        self.assertEqual(top_docs[0], "Doc B")


class TestFindBestModel(unittest.TestCase):

    @patch("os.path.exists")
    @patch("os.listdir")
    def test_find_best_model_final(self, mock_listdir, mock_exists):
        """Test finding generator_final.pt when it exists"""
        mock_exists.return_value = True
        mock_listdir.return_value = [
            "generator_final.pt",
            "generator_epoch_1.pt",
            "generator_epoch_2.pt",
        ]

        result = find_best_model("test_dir")

        self.assertEqual(result, "test_dir/generator_final.pt")

    @patch("os.path.exists")
    @patch("os.listdir")
    def test_find_best_model_best_checkpoint(self, mock_listdir, mock_exists):
        """Test finding best checkpoint when no final exists"""
        mock_exists.return_value = True
        mock_listdir.return_value = [
            "generator_best.pt",
            "generator_epoch_1.pt",
            "generator_epoch_2.pt",
        ]

        result = find_best_model("test_dir")

        self.assertEqual(result, "test_dir/generator_best.pt")

    @patch("os.path.exists")
    @patch("os.listdir")
    def test_find_best_model_latest_epoch(self, mock_listdir, mock_exists):
        """Test finding latest epoch when no final/best exists"""
        mock_exists.return_value = True
        mock_listdir.return_value = [
            "generator_epoch_1.pt",
            "generator_epoch_5.pt",
            "generator_epoch_3.pt",
        ]

        result = find_best_model("test_dir")

        self.assertEqual(result, "test_dir/generator_epoch_5.pt")

    @patch("os.path.exists")
    @patch("os.listdir")
    def test_find_best_model_fallback(self, mock_listdir, mock_exists):
        """Test falling back to first available checkpoint"""
        mock_exists.return_value = True
        mock_listdir.return_value = ["generator_custom.pt", "other_file.txt"]

        result = find_best_model("test_dir")

        self.assertEqual(result, "test_dir/generator_custom.pt")

    @patch("os.path.exists")
    def test_find_best_model_no_directory(self, mock_exists):
        """Test when weights directory doesn't exist"""
        mock_exists.return_value = False

        result = find_best_model("nonexistent_dir")

        self.assertIsNone(result)

    @patch("os.path.exists")
    @patch("os.listdir")
    def test_find_best_model_no_checkpoints(self, mock_listdir, mock_exists):
        """Test when no generator checkpoints exist"""
        mock_exists.return_value = True
        mock_listdir.return_value = ["other_file.txt", "data.json"]

        result = find_best_model("test_dir")

        self.assertIsNone(result)


class TestLoadModel(unittest.TestCase):

    @patch("hallucination_reduction.inference.find_best_model")
    @patch("hallucination_reduction.inference.AutoModelForCausalLM")
    @patch("hallucination_reduction.inference.AutoTokenizer")
    @patch("torch.load")
    @patch("os.path.exists")
    @patch("torch.cuda.is_available")
    def test_load_model_with_finetuned_weights(
        self,
        mock_cuda,
        mock_exists,
        mock_torch_load,
        mock_tokenizer_class,
        mock_model_class,
        mock_find_best,
    ):
        """Test loading model with fine-tuned weights"""
        mock_cuda.return_value = False
        mock_exists.return_value = True
        mock_find_best.return_value = "test_weights.pt"

        # Mock tokenizer
        mock_tokenizer = Mock()
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer

        # Mock model
        mock_model = MagicMock()
        # Use lambda to return a new iterator each time
        mock_model.parameters.side_effect = lambda: iter(
            [torch.nn.Parameter(torch.tensor([1.0]))]
        )
        mock_model.state_dict.return_value = {"layer.weight": torch.randn(10, 10)}
        mock_model_class.from_pretrained.return_value = mock_model

        # Mock loaded weights
        mock_torch_load.return_value = {"layer.weight": torch.randn(10, 10)}

        model, tokenizer, device = load_model("test-model", "test_dir")

        mock_tokenizer_class.from_pretrained.assert_called_once_with("test-model")
        mock_model_class.from_pretrained.assert_called_once()
        mock_torch_load.assert_called_once_with("test_weights.pt", map_location="cpu")
        mock_model.eval.assert_called_once()
        self.assertEqual(model, mock_model)
        self.assertEqual(tokenizer, mock_tokenizer)

    @patch("hallucination_reduction.inference.find_best_model")
    @patch("hallucination_reduction.inference.AutoModelForCausalLM")
    @patch("hallucination_reduction.inference.AutoTokenizer")
    @patch("torch.cuda.is_available")
    def test_load_model_without_finetuned_weights(
        self, mock_cuda, mock_tokenizer_class, mock_model_class, mock_find_best
    ):
        """Test loading base model when no fine-tuned weights exist"""
        mock_cuda.return_value = False
        mock_find_best.return_value = None

        # Mock tokenizer
        mock_tokenizer = Mock()
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer

        # Mock model
        mock_model = MagicMock()
        mock_model.parameters.side_effect = lambda: iter(
            [torch.nn.Parameter(torch.tensor([1.0]))]
        )
        mock_model_class.from_pretrained.return_value = mock_model

        model, tokenizer, device = load_model("test-model", "test_dir")

        mock_model.eval.assert_called_once()
        self.assertEqual(model, mock_model)
        self.assertEqual(tokenizer, mock_tokenizer)

    @patch("hallucination_reduction.inference.find_best_model")
    @patch("hallucination_reduction.inference.AutoModelForCausalLM")
    @patch("hallucination_reduction.inference.AutoTokenizer")
    @patch("torch.cuda.is_available")
    def test_load_model_cuda_available(
        self, mock_cuda, mock_tokenizer_class, mock_model_class, mock_find_best
    ):
        """Test that bfloat16 is used when CUDA is available"""
        mock_cuda.return_value = True
        mock_find_best.return_value = None

        mock_tokenizer = Mock()
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer

        mock_model = MagicMock()
        mock_model.parameters.side_effect = lambda: iter(
            [torch.nn.Parameter(torch.tensor([1.0]))]
        )
        mock_model_class.from_pretrained.return_value = mock_model

        load_model("test-model", "test_dir")

        # Check that bfloat16 was used
        call_kwargs = mock_model_class.from_pretrained.call_args[1]
        self.assertEqual(call_kwargs["torch_dtype"], torch.bfloat16)


class TestGenerateAnswer(unittest.TestCase):

    @patch("hallucination_reduction.inference.Thread")
    @patch("hallucination_reduction.inference.TextIteratorStreamer")
    @patch("builtins.print")
    def test_generate_answer_basic(
        self, mock_print, mock_streamer_class, mock_thread_class
    ):
        """Test basic answer generation"""
        # Mock model and tokenizer
        mock_model = MagicMock()
        mock_tokenizer = Mock()

        # Mock tokenizer return
        mock_inputs = MagicMock()
        mock_inputs.__getitem__.side_effect = lambda key: torch.tensor([[1, 2, 3]])
        mock_inputs.to.return_value = mock_inputs
        mock_tokenizer.return_value = mock_inputs
        mock_tokenizer.eos_token_id = 0

        # Mock streamer
        mock_streamer = MagicMock()
        mock_streamer.__iter__.return_value = iter(["Hello", " world", "!"])
        mock_streamer_class.return_value = mock_streamer

        # Mock thread
        mock_thread = Mock()
        mock_thread_class.return_value = mock_thread

        device = torch.device("cpu")
        question = "What is AI?"
        retrieved_docs = ["AI is artificial intelligence."]

        result = generate_answer(
            mock_model, mock_tokenizer, device, question, retrieved_docs
        )

        self.assertEqual(result, "Hello world!")
        mock_thread.start.assert_called_once()
        mock_thread.join.assert_called_once()
        mock_streamer_class.assert_called_once_with(
            mock_tokenizer, skip_prompt=True, skip_special_tokens=True
        )

    @patch("hallucination_reduction.inference.Thread")
    @patch("hallucination_reduction.inference.TextIteratorStreamer")
    @patch("builtins.print")
    def test_generate_answer_with_multiple_docs(
        self, mock_print, mock_streamer_class, mock_thread_class
    ):
        """Test answer generation with multiple retrieved documents"""
        mock_model = MagicMock()
        mock_tokenizer = Mock()

        mock_inputs = MagicMock()
        mock_inputs.__getitem__.side_effect = lambda key: torch.tensor([[1, 2, 3]])
        mock_inputs.to.return_value = mock_inputs
        mock_tokenizer.return_value = mock_inputs
        mock_tokenizer.eos_token_id = 0

        mock_streamer = MagicMock()
        mock_streamer.__iter__.return_value = iter(["Answer"])
        mock_streamer_class.return_value = mock_streamer

        mock_thread = Mock()
        mock_thread_class.return_value = mock_thread

        device = torch.device("cpu")
        question = "What is machine learning?"
        retrieved_docs = [
            "ML is a subset of AI.",
            "It involves training models.",
            "It uses data and algorithms.",
        ]

        result = generate_answer(
            mock_model, mock_tokenizer, device, question, retrieved_docs
        )

        self.assertEqual(result, "Answer")
        # Verify tokenizer was called with a prompt containing all docs
        called_prompt = mock_tokenizer.call_args[0][0]
        for doc in retrieved_docs:
            self.assertIn(doc, called_prompt)
        self.assertIn(question, called_prompt)

    @patch("hallucination_reduction.inference.Thread")
    @patch("hallucination_reduction.inference.TextIteratorStreamer")
    @patch("builtins.print")
    def test_generate_answer_custom_max_tokens(
        self, mock_print, mock_streamer_class, mock_thread_class
    ):
        """Test answer generation with custom max_new_tokens"""
        mock_model = MagicMock()
        mock_tokenizer = Mock()

        mock_inputs = MagicMock()
        mock_inputs.__getitem__.side_effect = lambda key: torch.tensor([[1, 2, 3]])
        mock_inputs.to.return_value = mock_inputs
        mock_tokenizer.return_value = mock_inputs
        mock_tokenizer.eos_token_id = 0

        mock_streamer = MagicMock()
        mock_streamer.__iter__.return_value = iter(["Test"])
        mock_streamer_class.return_value = mock_streamer

        mock_thread = Mock()
        mock_thread_class.return_value = mock_thread

        device = torch.device("cpu")

        generate_answer(
            mock_model, mock_tokenizer, device, "question", ["doc"], max_new_tokens=512
        )

        # Verify Thread was called with max_new_tokens=512
        thread_kwargs = mock_thread_class.call_args[1]["kwargs"]
        self.assertEqual(thread_kwargs["max_new_tokens"], 512)


class TestIntegration(unittest.TestCase):
    """Integration tests for the inference module"""

    @patch("hallucination_reduction.inference.SentenceTransformer")
    def test_embeddings_and_retrieval_flow(self, mock_st_class):
        """Test the flow from building embeddings to retrieval"""
        mock_embedder = Mock()

        # Create embeddings that make doc 1 most similar to query
        corpus_embs = np.array([[0.1, 0.1], [0.9, 0.9], [0.2, 0.2]])
        query_emb = np.array([[0.85, 0.85]])

        mock_embedder.encode.side_effect = [corpus_embs, query_emb]
        mock_st_class.return_value = mock_embedder

        docs = ["Doc 0", "Doc 1", "Doc 2"]

        # Build embeddings
        embedder, embeddings = build_embeddings(docs)

        # Retrieve
        top_docs = retrieve_relevant_chunks(
            "test query", embedder, embeddings, docs, top_k=1
        )

        self.assertEqual(len(top_docs), 1)
        self.assertEqual(top_docs[0], "Doc 1")
