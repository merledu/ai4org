import pytest
import torch
from unittest.mock import MagicMock, patch, mock_open
from hallucination_reduction.main import unwrap, main


class TestUnwrap:
    """Test suite for unwrap function."""

    def test_function_exists(self):
        """Test that unwrap can be imported."""
        assert callable(unwrap)

    def test_unwrap_regular_model(self):
        """Test unwrapping a regular model returns itself."""
        model = MagicMock()
        # Model without 'module' attribute
        del model.module
        
        result = unwrap(model)
        assert result == model

    def test_unwrap_dataparallel_model(self):
        """Test unwrapping DataParallel model returns module."""
        model = MagicMock()
        inner_model = MagicMock()
        model.module = inner_model
        
        result = unwrap(model)
        assert result == inner_model

    def test_unwrap_preserves_model_without_module(self):
        """Test that model without module attribute is returned as-is."""
        class SimpleModel:
            def __init__(self):
                self.layers = []
        
        model = SimpleModel()
        result = unwrap(model)
        assert result is model


class TestMainFunction:
    """Test suite for main function."""

    def test_function_exists(self):
        """Test that main function can be imported."""
        assert callable(main)

    @patch('hallucination_reduction.main.torch.save')
    @patch('hallucination_reduction.main.reinforcement_learning_loop')
    @patch('hallucination_reduction.main.evaluate_old_vs_new_generator')
    @patch('hallucination_reduction.main.sft_finetune_generator')
    @patch('hallucination_reduction.main.evaluate_classifier')
    @patch('hallucination_reduction.main.train_discriminator_minibatch')
    @patch('hallucination_reduction.main.load_discriminator')
    @patch('hallucination_reduction.main.load_generator')
    @patch('hallucination_reduction.main.SimpleRetriever')
    @patch('hallucination_reduction.main.build_corpus_and_qa')
    @patch('torch.cuda.is_available')
    def test_main_runs_without_error(self, mock_cuda, mock_build, mock_retriever,
                                     mock_load_gen, mock_load_disc, mock_train_disc,
                                     mock_eval_class, mock_sft, mock_eval_gen, mock_rl,
                                     mock_save):
        """Test that main function runs without errors."""
        # Setup mocks
        mock_cuda.return_value = False
        
        # Mock data
        mock_qa = MagicMock()
        mock_qa.question = "Q?"
        mock_qa.answer = "A"
        mock_qa.supporting_passages = ["P"]
        mock_build.return_value = (["passage"], [mock_qa])
        
        # Mock retriever
        mock_retriever_instance = MagicMock()
        mock_retriever.return_value = mock_retriever_instance
        
        # Mock generator
        mock_gen = MagicMock()
        mock_tok = MagicMock()
        mock_load_gen.return_value = (mock_tok, mock_gen)
        
        # Mock discriminator
        mock_disc = MagicMock()
        mock_disc_tok = MagicMock()
        mock_load_disc.return_value = (mock_disc_tok, mock_disc)
        
        # Mock training
        mock_train_disc.return_value = mock_disc
        mock_eval_class.return_value = {"acc": 0.9, "prec": 0.9, "rec": 0.9, "f1": 0.9}
        mock_eval_gen.return_value = (
            [{"question": "Q?", "gold": "A", "old": "O", "new": "N"}],
            {"exact_match_rate": 0.5, "avg_f1": 0.6, "hallucination_rate": 0.3},
            {"exact_match_rate": 0.7, "avg_f1": 0.8, "hallucination_rate": 0.1}
        )
        mock_sft.return_value = mock_gen
        mock_rl.return_value = []
        
        # Run main
        with patch('builtins.print'):
            main()
        
        # Verify key functions were called
        assert mock_build.called
        assert mock_load_gen.called
        assert mock_load_disc.called

    @patch('hallucination_reduction.main.torch.save')
    @patch('hallucination_reduction.main.reinforcement_learning_loop')
    @patch('hallucination_reduction.main.evaluate_old_vs_new_generator')
    @patch('hallucination_reduction.main.sft_finetune_generator')
    @patch('hallucination_reduction.main.evaluate_classifier')
    @patch('hallucination_reduction.main.train_discriminator_minibatch')
    @patch('hallucination_reduction.main.load_discriminator')
    @patch('hallucination_reduction.main.load_generator')
    @patch('hallucination_reduction.main.SimpleRetriever')
    @patch('hallucination_reduction.main.build_corpus_and_qa')
    @patch('torch.cuda.is_available')
    def test_cuda_device_selected_when_available(self, mock_cuda, mock_build, mock_retriever,
                                                 mock_load_gen, mock_load_disc, mock_train_disc,
                                                 mock_eval_class, mock_sft, mock_eval_gen, mock_rl,
                                                 mock_save):
        """Test that CUDA device is selected when available."""
        mock_cuda.return_value = True
        
        # Setup minimal mocks
        mock_qa = MagicMock()
        mock_qa.question = "Q?"
        mock_qa.answer = "A"
        mock_qa.supporting_passages = ["P"]
        mock_build.return_value = (["passage"], [mock_qa])
        mock_retriever.return_value = MagicMock()
        
        mock_gen = MagicMock()
        mock_tok = MagicMock()
        mock_load_gen.return_value = (mock_tok, mock_gen)
        
        mock_disc = MagicMock()
        mock_disc_tok = MagicMock()
        mock_load_disc.return_value = (mock_disc_tok, mock_disc)
        
        mock_train_disc.return_value = mock_disc
        mock_eval_class.return_value = {"acc": 0.9, "prec": 0.9, "rec": 0.9, "f1": 0.9}
        mock_eval_gen.return_value = (
            [{"question": "Q?", "gold": "A", "old": "O", "new": "N"}],
            {"exact_match_rate": 0.5, "avg_f1": 0.6, "hallucination_rate": 0.3},
            {"exact_match_rate": 0.7, "avg_f1": 0.8, "hallucination_rate": 0.1}
        )
        mock_sft.return_value = mock_gen
        mock_rl.return_value = []
        
        with patch('builtins.print') as mock_print:
            main()
            
            # Check that CUDA device message was printed
            print_calls = [str(call) for call in mock_print.call_args_list]
            cuda_mentioned = any('cuda' in str(call).lower() for call in print_calls)
            assert cuda_mentioned

    @patch('hallucination_reduction.main.torch.save')
    @patch('hallucination_reduction.main.reinforcement_learning_loop')
    @patch('hallucination_reduction.main.evaluate_old_vs_new_generator')
    @patch('hallucination_reduction.main.sft_finetune_generator')
    @patch('hallucination_reduction.main.evaluate_classifier')
    @patch('hallucination_reduction.main.train_discriminator_minibatch')
    @patch('hallucination_reduction.main.load_discriminator')
    @patch('hallucination_reduction.main.load_generator')
    @patch('hallucination_reduction.main.SimpleRetriever')
    @patch('hallucination_reduction.main.build_corpus_and_qa')
    @patch('torch.cuda.is_available')
    def test_loads_corpus_and_qa(self, mock_cuda, mock_build, mock_retriever,
                                 mock_load_gen, mock_load_disc, mock_train_disc,
                                 mock_eval_class, mock_sft, mock_eval_gen, mock_rl,
                                 mock_save):
        """Test that corpus and QA pairs are loaded."""
        mock_cuda.return_value = False
        
        mock_qa = MagicMock()
        mock_qa.question = "Q?"
        mock_qa.answer = "A"
        mock_qa.supporting_passages = ["P"]
        mock_build.return_value = (["passage"], [mock_qa])
        mock_retriever.return_value = MagicMock()
        
        mock_gen = MagicMock()
        mock_tok = MagicMock()
        mock_load_gen.return_value = (mock_tok, mock_gen)
        
        mock_disc = MagicMock()
        mock_disc_tok = MagicMock()
        mock_load_disc.return_value = (mock_disc_tok, mock_disc)
        
        mock_train_disc.return_value = mock_disc
        mock_eval_class.return_value = {"acc": 0.9, "prec": 0.9, "rec": 0.9, "f1": 0.9}
        mock_eval_gen.return_value = (
            [{"question": "Q?", "gold": "A", "old": "O", "new": "N"}],
            {"exact_match_rate": 0.5, "avg_f1": 0.6, "hallucination_rate": 0.3},
            {"exact_match_rate": 0.7, "avg_f1": 0.8, "hallucination_rate": 0.1}
        )
        mock_sft.return_value = mock_gen
        mock_rl.return_value = []
        
        with patch('builtins.print'):
            main()
        
        mock_build.assert_called_once()

    @patch('hallucination_reduction.main.torch.save')
    @patch('hallucination_reduction.main.reinforcement_learning_loop')
    @patch('hallucination_reduction.main.evaluate_old_vs_new_generator')
    @patch('hallucination_reduction.main.sft_finetune_generator')
    @patch('hallucination_reduction.main.evaluate_classifier')
    @patch('hallucination_reduction.main.train_discriminator_minibatch')
    @patch('hallucination_reduction.main.load_discriminator')
    @patch('hallucination_reduction.main.load_generator')
    @patch('hallucination_reduction.main.SimpleRetriever')
    @patch('hallucination_reduction.main.build_corpus_and_qa')
    @patch('torch.cuda.is_available')
    def test_trains_three_discriminators(self, mock_cuda, mock_build, mock_retriever,
                                         mock_load_gen, mock_load_disc, mock_train_disc,
                                         mock_eval_class, mock_sft, mock_eval_gen, mock_rl,
                                         mock_save):
        """Test that three discriminators are trained (fact, style, safety)."""
        mock_cuda.return_value = False
        
        mock_qa = MagicMock()
        mock_qa.question = "Q?"
        mock_qa.answer = "A"
        mock_qa.supporting_passages = ["P"]
        mock_build.return_value = (["passage"], [mock_qa])
        mock_retriever.return_value = MagicMock()
        
        mock_gen = MagicMock()
        mock_tok = MagicMock()
        mock_load_gen.return_value = (mock_tok, mock_gen)
        
        mock_disc = MagicMock()
        mock_disc_tok = MagicMock()
        mock_load_disc.return_value = (mock_disc_tok, mock_disc)
        
        mock_train_disc.return_value = mock_disc
        mock_eval_class.return_value = {"acc": 0.9, "prec": 0.9, "rec": 0.9, "f1": 0.9}
        mock_eval_gen.return_value = (
            [{"question": "Q?", "gold": "A", "old": "O", "new": "N"}],
            {"exact_match_rate": 0.5, "avg_f1": 0.6, "hallucination_rate": 0.3},
            {"exact_match_rate": 0.7, "avg_f1": 0.8, "hallucination_rate": 0.1}
        )
        mock_sft.return_value = mock_gen
        mock_rl.return_value = []
        
        with patch('builtins.print'):
            main()
        
        # Should be called 3 times (fact, style, safety)
        assert mock_train_disc.call_count == 3

    @patch('hallucination_reduction.main.torch.save')
    @patch('hallucination_reduction.main.reinforcement_learning_loop')
    @patch('hallucination_reduction.main.evaluate_old_vs_new_generator')
    @patch('hallucination_reduction.main.sft_finetune_generator')
    @patch('hallucination_reduction.main.evaluate_classifier')
    @patch('hallucination_reduction.main.train_discriminator_minibatch')
    @patch('hallucination_reduction.main.load_discriminator')
    @patch('hallucination_reduction.main.load_generator')
    @patch('hallucination_reduction.main.SimpleRetriever')
    @patch('hallucination_reduction.main.build_corpus_and_qa')
    @patch('torch.cuda.is_available')
    def test_runs_sft_finetuning(self, mock_cuda, mock_build, mock_retriever,
                                 mock_load_gen, mock_load_disc, mock_train_disc,
                                 mock_eval_class, mock_sft, mock_eval_gen, mock_rl,
                                 mock_save):
        """Test that SFT fine-tuning is executed."""
        mock_cuda.return_value = False
        
        mock_qa = MagicMock()
        mock_qa.question = "Q?"
        mock_qa.answer = "A"
        mock_qa.supporting_passages = ["P"]
        mock_build.return_value = (["passage"], [mock_qa])
        mock_retriever.return_value = MagicMock()
        
        mock_gen = MagicMock()
        mock_tok = MagicMock()
        mock_load_gen.return_value = (mock_tok, mock_gen)
        
        mock_disc = MagicMock()
        mock_disc_tok = MagicMock()
        mock_load_disc.return_value = (mock_disc_tok, mock_disc)
        
        mock_train_disc.return_value = mock_disc
        mock_eval_class.return_value = {"acc": 0.9, "prec": 0.9, "rec": 0.9, "f1": 0.9}
        mock_eval_gen.return_value = (
            [{"question": "Q?", "gold": "A", "old": "O", "new": "N"}],
            {"exact_match_rate": 0.5, "avg_f1": 0.6, "hallucination_rate": 0.3},
            {"exact_match_rate": 0.7, "avg_f1": 0.8, "hallucination_rate": 0.1}
        )
        mock_sft.return_value = mock_gen
        mock_rl.return_value = []
        
        with patch('builtins.print'):
            main()
        
        mock_sft.assert_called_once()

    @patch('hallucination_reduction.main.torch.save')
    @patch('hallucination_reduction.main.reinforcement_learning_loop')
    @patch('hallucination_reduction.main.evaluate_old_vs_new_generator')
    @patch('hallucination_reduction.main.sft_finetune_generator')
    @patch('hallucination_reduction.main.evaluate_classifier')
    @patch('hallucination_reduction.main.train_discriminator_minibatch')
    @patch('hallucination_reduction.main.load_discriminator')
    @patch('hallucination_reduction.main.load_generator')
    @patch('hallucination_reduction.main.SimpleRetriever')
    @patch('hallucination_reduction.main.build_corpus_and_qa')
    @patch('torch.cuda.is_available')
    def test_runs_reinforcement_learning(self, mock_cuda, mock_build, mock_retriever,
                                         mock_load_gen, mock_load_disc, mock_train_disc,
                                         mock_eval_class, mock_sft, mock_eval_gen, mock_rl,
                                         mock_save):
        """Test that reinforcement learning loop is executed."""
        mock_cuda.return_value = False
        
        mock_qa = MagicMock()
        mock_qa.question = "Q?"
        mock_qa.answer = "A"
        mock_qa.supporting_passages = ["P"]
        mock_build.return_value = (["passage"], [mock_qa])
        mock_retriever.return_value = MagicMock()
        
        mock_gen = MagicMock()
        mock_tok = MagicMock()
        mock_load_gen.return_value = (mock_tok, mock_gen)
        
        mock_disc = MagicMock()
        mock_disc_tok = MagicMock()
        mock_load_disc.return_value = (mock_disc_tok, mock_disc)
        
        mock_train_disc.return_value = mock_disc
        mock_eval_class.return_value = {"acc": 0.9, "prec": 0.9, "rec": 0.9, "f1": 0.9}
        mock_eval_gen.return_value = (
            [{"question": "Q?", "gold": "A", "old": "O", "new": "N"}],
            {"exact_match_rate": 0.5, "avg_f1": 0.6, "hallucination_rate": 0.3},
            {"exact_match_rate": 0.7, "avg_f1": 0.8, "hallucination_rate": 0.1}
        )
        mock_sft.return_value = mock_gen
        mock_rl.return_value = []
        
        with patch('builtins.print'):
            main()
        
        mock_rl.assert_called_once()

    @patch('hallucination_reduction.main.torch.save')
    @patch('hallucination_reduction.main.reinforcement_learning_loop')
    @patch('hallucination_reduction.main.evaluate_old_vs_new_generator')
    @patch('hallucination_reduction.main.sft_finetune_generator')
    @patch('hallucination_reduction.main.evaluate_classifier')
    @patch('hallucination_reduction.main.train_discriminator_minibatch')
    @patch('hallucination_reduction.main.load_discriminator')
    @patch('hallucination_reduction.main.load_generator')
    @patch('hallucination_reduction.main.SimpleRetriever')
    @patch('hallucination_reduction.main.build_corpus_and_qa')
    @patch('torch.cuda.is_available')
    def test_saves_models(self, mock_cuda, mock_build, mock_retriever,
                         mock_load_gen, mock_load_disc, mock_train_disc,
                         mock_eval_class, mock_sft, mock_eval_gen, mock_rl,
                         mock_save):
        """Test that models are saved."""
        mock_cuda.return_value = False
        
        mock_qa = MagicMock()
        mock_qa.question = "Q?"
        mock_qa.answer = "A"
        mock_qa.supporting_passages = ["P"]
        mock_build.return_value = (["passage"], [mock_qa])
        mock_retriever.return_value = MagicMock()
        
        mock_gen = MagicMock()
        mock_tok = MagicMock()
        mock_load_gen.return_value = (mock_tok, mock_gen)
        
        mock_disc = MagicMock()
        mock_disc_tok = MagicMock()
        mock_load_disc.return_value = (mock_disc_tok, mock_disc)
        
        mock_train_disc.return_value = mock_disc
        mock_eval_class.return_value = {"acc": 0.9, "prec": 0.9, "rec": 0.9, "f1": 0.9}
        mock_eval_gen.return_value = (
            [{"question": "Q?", "gold": "A", "old": "O", "new": "N"}],
            {"exact_match_rate": 0.5, "avg_f1": 0.6, "hallucination_rate": 0.3},
            {"exact_match_rate": 0.7, "avg_f1": 0.8, "hallucination_rate": 0.1}
        )
        mock_sft.return_value = mock_gen
        mock_rl.return_value = []
        
        with patch('builtins.print'):
            main()
        
        # Should save 4 models: generator, fact_disc, style_disc, safety_disc
        assert mock_save.call_count == 4

    @patch('hallucination_reduction.main.torch.save')
    @patch('hallucination_reduction.main.reinforcement_learning_loop')
    @patch('hallucination_reduction.main.evaluate_old_vs_new_generator')
    @patch('hallucination_reduction.main.sft_finetune_generator')
    @patch('hallucination_reduction.main.evaluate_classifier')
    @patch('hallucination_reduction.main.train_discriminator_minibatch')
    @patch('hallucination_reduction.main.load_discriminator')
    @patch('hallucination_reduction.main.load_generator')
    @patch('hallucination_reduction.main.SimpleRetriever')
    @patch('hallucination_reduction.main.build_corpus_and_qa')
    @patch('torch.cuda.is_available')
    def test_evaluates_before_and_after_training(self, mock_cuda, mock_build, mock_retriever,
                                                 mock_load_gen, mock_load_disc, mock_train_disc,
                                                 mock_eval_class, mock_sft, mock_eval_gen, mock_rl,
                                                 mock_save):
        """Test that evaluation happens before and after training."""
        mock_cuda.return_value = False
        
        mock_qa = MagicMock()
        mock_qa.question = "Q?"
        mock_qa.answer = "A"
        mock_qa.supporting_passages = ["P"]
        mock_build.return_value = (["passage"], [mock_qa])
        mock_retriever.return_value = MagicMock()
        
        mock_gen = MagicMock()
        mock_tok = MagicMock()
        mock_load_gen.return_value = (mock_tok, mock_gen)
        
        mock_disc = MagicMock()
        mock_disc_tok = MagicMock()
        mock_load_disc.return_value = (mock_disc_tok, mock_disc)
        
        mock_train_disc.return_value = mock_disc
        mock_eval_class.return_value = {"acc": 0.9, "prec": 0.9, "rec": 0.9, "f1": 0.9}
        mock_eval_gen.return_value = (
            [{"question": "Q?", "gold": "A", "old": "O", "new": "N"}],
            {"exact_match_rate": 0.5, "avg_f1": 0.6, "hallucination_rate": 0.3},
            {"exact_match_rate": 0.7, "avg_f1": 0.8, "hallucination_rate": 0.1}
        )
        mock_sft.return_value = mock_gen
        mock_rl.return_value = []
        
        with patch('builtins.print'):
            main()
        
        # Should evaluate: baseline, after SFT, final
        assert mock_eval_gen.call_count >= 3

    @patch('hallucination_reduction.main.torch.save')
    @patch('hallucination_reduction.main.reinforcement_learning_loop')
    @patch('hallucination_reduction.main.evaluate_old_vs_new_generator')
    @patch('hallucination_reduction.main.sft_finetune_generator')
    @patch('hallucination_reduction.main.evaluate_classifier')
    @patch('hallucination_reduction.main.train_discriminator_minibatch')
    @patch('hallucination_reduction.main.load_discriminator')
    @patch('hallucination_reduction.main.load_generator')
    @patch('hallucination_reduction.main.SimpleRetriever')
    @patch('hallucination_reduction.main.build_corpus_and_qa')
    @patch('torch.cuda.is_available')
    def test_creates_baseline_generator_copy(self, mock_cuda, mock_build, mock_retriever,
                                             mock_load_gen, mock_load_disc, mock_train_disc,
                                             mock_eval_class, mock_sft, mock_eval_gen, mock_rl,
                                             mock_save):
        """Test that a baseline copy of generator is created."""
        mock_cuda.return_value = False
        
        mock_qa = MagicMock()
        mock_qa.question = "Q?"
        mock_qa.answer = "A"
        mock_qa.supporting_passages = ["P"]
        mock_build.return_value = (["passage"], [mock_qa])
        mock_retriever.return_value = MagicMock()
        
        mock_gen = MagicMock()
        mock_tok = MagicMock()
        mock_load_gen.return_value = (mock_tok, mock_gen)
        
        mock_disc = MagicMock()
        mock_disc_tok = MagicMock()
        mock_load_disc.return_value = (mock_disc_tok, mock_disc)
        
        mock_train_disc.return_value = mock_disc
        mock_eval_class.return_value = {"acc": 0.9, "prec": 0.9, "rec": 0.9, "f1": 0.9}
        mock_eval_gen.return_value = (
            [{"question": "Q?", "gold": "A", "old": "O", "new": "N"}],
            {"exact_match_rate": 0.5, "avg_f1": 0.6, "hallucination_rate": 0.3},
            {"exact_match_rate": 0.7, "avg_f1": 0.8, "hallucination_rate": 0.1}
        )
        mock_sft.return_value = mock_gen
        mock_rl.return_value = []
        
        with patch('builtins.print'):
            with patch('hallucination_reduction.main.copy.deepcopy') as mock_deepcopy:
                mock_deepcopy.return_value = MagicMock()
                main()
                
                # Should create a deep copy for baseline
                mock_deepcopy.assert_called_once()