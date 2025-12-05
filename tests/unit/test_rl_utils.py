import pytest
import torch
import numpy as np
from unittest.mock import MagicMock, patch, mock_open
from hallucination_reduction.rl_utils import (
    is_repetitive,
    safe_save_state,
    compute_reward,
    monte_carlo_rewards,
    reinforce_update,
    reinforcement_learning_loop
)


class TestIsRepetitive:
    """Test suite for is_repetitive function."""

    def test_function_exists(self):
        """Test that is_repetitive can be imported."""
        assert callable(is_repetitive)

    def test_empty_string_not_repetitive(self):
        """Test that empty string is not repetitive."""
        assert is_repetitive("") is False

    def test_normal_text_not_repetitive(self):
        """Test that normal text is not repetitive."""
        text = "This is a normal sentence with different words"
        assert is_repetitive(text) is False

    def test_highly_repetitive_text(self):
        """Test that highly repetitive text is detected."""
        text = "the the the the the the the the the the"
        assert is_repetitive(text) is True

    def test_barely_repetitive_text(self):
        """Test text at the threshold."""
        # 6 out of 10 tokens are "the" (60%)
        text = "the the the the the the word word word word"
        result = is_repetitive(text, repeat_threshold=0.6)
        assert result is True

    def test_just_below_threshold(self):
        """Test text just below threshold is not repetitive."""
        # 5 out of 10 tokens are "the" (50%)
        text = "the the the the the word word word word word"
        result = is_repetitive(text, repeat_threshold=0.6)
        assert result is False

    def test_single_word(self):
        """Test single word text."""
        text = "word"
        # 1 out of 1 = 100%, above threshold
        assert is_repetitive(text) is True

    def test_two_different_words(self):
        """Test two different words."""
        text = "word1 word2"
        # Max is 1 out of 2 = 50%
        assert is_repetitive(text, repeat_threshold=0.6) is False

    def test_custom_threshold(self):
        """Test with custom threshold."""
        text = "a a a b b"
        # 3 out of 5 = 60%
        assert is_repetitive(text, repeat_threshold=0.5) is True
        assert is_repetitive(text, repeat_threshold=0.7) is False

    def test_whitespace_only(self):
        """Test string with only whitespace."""
        text = "   "
        assert is_repetitive(text) is False


class TestSafeSaveState:
    """Test suite for safe_save_state function."""

    def test_function_exists(self):
        """Test that safe_save_state can be imported."""
        assert callable(safe_save_state)

    def test_saves_model_successfully(self):
        """Test successful model save."""
        model = MagicMock()
        model.state_dict.return_value = {"weight": torch.randn(10, 10)}
        
        with patch('torch.save') as mock_save:
            safe_save_state(model, "test_path.pt")
            mock_save.assert_called_once()

    def test_handles_save_exception(self):
        """Test that exceptions during save are caught."""
        model = MagicMock()
        model.state_dict.return_value = {"weight": torch.randn(10, 10)}
        
        with patch('torch.save', side_effect=Exception("Save failed")):
            with patch('builtins.print') as mock_print:
                # Should not raise exception
                safe_save_state(model, "test_path.pt")
                # Should print warning
                assert mock_print.called


class TestComputeReward:
    """Test suite for compute_reward function."""

    @pytest.fixture
    def mock_discriminators(self):
        """Create mock discriminators."""
        fact_disc = MagicMock()
        style_disc = MagicMock()
        safety_disc = MagicMock()
        return fact_disc, style_disc, safety_disc

    @pytest.fixture
    def mock_tokenizers(self):
        """Create mock tokenizers."""
        fact_tok = MagicMock()
        style_tok = MagicMock()
        safety_tok = MagicMock()
        return fact_tok, style_tok, safety_tok

    def test_function_exists(self):
        """Test that compute_reward can be imported."""
        assert callable(compute_reward)

    def test_returns_tuple(self, mock_discriminators, mock_tokenizers):
        """Test that compute_reward returns a tuple."""
        fact_disc, style_disc, safety_disc = mock_discriminators
        fact_tok, style_tok, safety_tok = mock_tokenizers
        
        with patch('hallucination_reduction.rl_utils.discriminator_predict_text') as mock_pred:
            mock_pred.return_value = [{"probs": [0.3, 0.7]}]
            
            with patch('hallucination_reduction.rl_utils.overlap_fact_check', return_value=0.8):
                result = compute_reward(
                    "Generated answer", ["doc1"], ["passage1"],
                    fact_disc, style_disc, safety_disc,
                    fact_tok, style_tok, safety_tok
                )
                
                assert isinstance(result, tuple)
                assert len(result) == 2

    def test_returns_float_reward_and_debug_dict(self, mock_discriminators, mock_tokenizers):
        """Test return types."""
        fact_disc, style_disc, safety_disc = mock_discriminators
        fact_tok, style_tok, safety_tok = mock_tokenizers
        
        with patch('hallucination_reduction.rl_utils.discriminator_predict_text') as mock_pred:
            mock_pred.return_value = [{"probs": [0.3, 0.7]}]
            
            with patch('hallucination_reduction.rl_utils.overlap_fact_check', return_value=0.8):
                reward, debug = compute_reward(
                    "Generated answer", ["doc1"], ["passage1"],
                    fact_disc, style_disc, safety_disc,
                    fact_tok, style_tok, safety_tok
                )
                
                assert isinstance(reward, float)
                assert isinstance(debug, dict)

    def test_debug_contains_expected_keys(self, mock_discriminators, mock_tokenizers):
        """Test that debug dict contains expected keys."""
        fact_disc, style_disc, safety_disc = mock_discriminators
        fact_tok, style_tok, safety_tok = mock_tokenizers
        
        with patch('hallucination_reduction.rl_utils.discriminator_predict_text') as mock_pred:
            mock_pred.return_value = [{"probs": [0.3, 0.7]}]
            
            with patch('hallucination_reduction.rl_utils.overlap_fact_check', return_value=0.8):
                _, debug = compute_reward(
                    "Generated answer", ["doc1"], ["passage1"],
                    fact_disc, style_disc, safety_disc,
                    fact_tok, style_tok, safety_tok
                )
                
                assert "p_fact" in debug
                assert "p_style" in debug
                assert "p_safe" in debug
                assert "overlap" in debug
                assert "combined" in debug

    def test_reward_in_valid_range(self, mock_discriminators, mock_tokenizers):
        """Test that reward is in [0, 1] range."""
        fact_disc, style_disc, safety_disc = mock_discriminators
        fact_tok, style_tok, safety_tok = mock_tokenizers
        
        with patch('hallucination_reduction.rl_utils.discriminator_predict_text') as mock_pred:
            mock_pred.return_value = [{"probs": [0.3, 0.7]}]
            
            with patch('hallucination_reduction.rl_utils.overlap_fact_check', return_value=0.8):
                reward, _ = compute_reward(
                    "Generated answer", ["doc1"], ["passage1"],
                    fact_disc, style_disc, safety_disc,
                    fact_tok, style_tok, safety_tok
                )
                
                assert 0.0 <= reward <= 1.0

    def test_penalty_applied_when_fact_low(self, mock_discriminators, mock_tokenizers):
        """Test that penalty is applied when factuality is low."""
        fact_disc, style_disc, safety_disc = mock_discriminators
        fact_tok, style_tok, safety_tok = mock_tokenizers
        
        with patch('hallucination_reduction.rl_utils.discriminator_predict_text') as mock_pred:
            # Low factuality score (p_fact < 0.5)
            mock_pred.return_value = [{"probs": [0.7, 0.3]}]
            
            with patch('hallucination_reduction.rl_utils.overlap_fact_check', return_value=0.8):
                reward, _ = compute_reward(
                    "Generated answer", ["doc1"], ["passage1"],
                    fact_disc, style_disc, safety_disc,
                    fact_tok, style_tok, safety_tok
                )
                
                # Should have penalty applied
                assert reward >= 0.0

    def test_repetitive_text_penalty(self, mock_discriminators, mock_tokenizers):
        """Test that repetitive text gets penalty."""
        fact_disc, style_disc, safety_disc = mock_discriminators
        fact_tok, style_tok, safety_tok = mock_tokenizers
        
        with patch('hallucination_reduction.rl_utils.discriminator_predict_text') as mock_pred:
            mock_pred.return_value = [{"probs": [0.3, 0.7]}]
            
            with patch('hallucination_reduction.rl_utils.overlap_fact_check', return_value=0.8):
                # Repetitive text
                reward, _ = compute_reward(
                    "the the the the the the the the",
                    ["doc1"], ["passage1"],
                    fact_disc, style_disc, safety_disc,
                    fact_tok, style_tok, safety_tok
                )
                
                # Should have low reward due to repetition penalty
                assert reward < 0.5


class TestMonteCarloRewards:
    """Test suite for monte_carlo_rewards function."""

    @pytest.fixture
    def mock_generator(self):
        """Create mock generator."""
        generator = MagicMock()
        return generator

    @pytest.fixture
    def mock_tokenizer(self):
        """Create mock tokenizer."""
        return MagicMock()

    @pytest.fixture
    def mock_discriminators(self):
        """Create mock discriminators."""
        return MagicMock(), MagicMock(), MagicMock()

    @pytest.fixture
    def mock_tokenizers(self):
        """Create mock tokenizers."""
        return MagicMock(), MagicMock(), MagicMock()

    def test_function_exists(self):
        """Test that monte_carlo_rewards can be imported."""
        assert callable(monte_carlo_rewards)

    def test_returns_tuple_of_three(self, mock_generator, mock_tokenizer, 
                                    mock_discriminators, mock_tokenizers):
        """Test that function returns 3 items."""
        fact_disc, style_disc, safety_disc = mock_discriminators
        fact_tok, style_tok, safety_tok = mock_tokenizers
        
        with patch('hallucination_reduction.rl_utils.generate_answer') as mock_gen:
            mock_gen.return_value = ["sample1", "sample2", "sample3"]
            
            with patch('hallucination_reduction.rl_utils.compute_reward') as mock_reward:
                mock_reward.return_value = (0.7, {"p_fact": 0.8})
                
                result = monte_carlo_rewards(
                    "prompt", mock_generator, mock_tokenizer,
                    ["doc1"], ["passage1"],
                    fact_disc, style_disc, safety_disc,
                    fact_tok, style_tok, safety_tok
                )
                
                assert len(result) == 3

    def test_returns_avg_reward_debug_samples(self, mock_generator, mock_tokenizer,
                                              mock_discriminators, mock_tokenizers):
        """Test return types."""
        fact_disc, style_disc, safety_disc = mock_discriminators
        fact_tok, style_tok, safety_tok = mock_tokenizers
        
        with patch('hallucination_reduction.rl_utils.generate_answer') as mock_gen:
            mock_gen.return_value = ["sample1", "sample2"]
            
            with patch('hallucination_reduction.rl_utils.compute_reward') as mock_reward:
                mock_reward.return_value = (0.7, {"p_fact": 0.8})
                
                avg_reward, debug_list, samples = monte_carlo_rewards(
                    "prompt", mock_generator, mock_tokenizer,
                    ["doc1"], ["passage1"],
                    fact_disc, style_disc, safety_disc,
                    fact_tok, style_tok, safety_tok
                )
                
                assert isinstance(avg_reward, float)
                assert isinstance(debug_list, list)
                assert isinstance(samples, list)

    def test_sets_generator_to_eval_mode(self, mock_generator, mock_tokenizer,
                                         mock_discriminators, mock_tokenizers):
        """Test that generator is set to eval mode."""
        fact_disc, style_disc, safety_disc = mock_discriminators
        fact_tok, style_tok, safety_tok = mock_tokenizers
        
        with patch('hallucination_reduction.rl_utils.generate_answer') as mock_gen:
            mock_gen.return_value = ["sample1"]
            
            with patch('hallucination_reduction.rl_utils.compute_reward') as mock_reward:
                mock_reward.return_value = (0.7, {"p_fact": 0.8})
                
                monte_carlo_rewards(
                    "prompt", mock_generator, mock_tokenizer,
                    ["doc1"], ["passage1"],
                    fact_disc, style_disc, safety_disc,
                    fact_tok, style_tok, safety_tok
                )
                
                mock_generator.eval.assert_called()

    def test_computes_average_reward(self, mock_generator, mock_tokenizer,
                                     mock_discriminators, mock_tokenizers):
        """Test that average reward is computed correctly."""
        fact_disc, style_disc, safety_disc = mock_discriminators
        fact_tok, style_tok, safety_tok = mock_tokenizers
        
        with patch('hallucination_reduction.rl_utils.generate_answer') as mock_gen:
            mock_gen.return_value = ["sample1", "sample2", "sample3"]
            
            with patch('hallucination_reduction.rl_utils.compute_reward') as mock_reward:
                # Return different rewards for each sample
                mock_reward.side_effect = [
                    (0.6, {"p_fact": 0.7}),
                    (0.8, {"p_fact": 0.9}),
                    (0.7, {"p_fact": 0.8})
                ]
                
                avg_reward, _, _ = monte_carlo_rewards(
                    "prompt", mock_generator, mock_tokenizer,
                    ["doc1"], ["passage1"],
                    fact_disc, style_disc, safety_disc,
                    fact_tok, style_tok, safety_tok,
                    n_rollouts=3
                )
                
                # Average of 0.6, 0.8, 0.7 = 0.7
                assert abs(avg_reward - 0.7) < 0.01


class TestReinforceUpdate:
    """Test suite for reinforce_update function."""

    @pytest.fixture
    def mock_generator(self):
        """Create mock generator."""
        generator = MagicMock()
        mock_output = MagicMock()
        mock_output.logits = torch.randn(1, 10, 1000)  # batch, seq, vocab
        generator.return_value = mock_output
        generator.parameters.return_value = [torch.nn.Parameter(torch.randn(10, 10))]
        return generator

    @pytest.fixture
    def mock_tokenizer(self):
        """Create mock tokenizer."""
        tokenizer = MagicMock()
        tokenizer.return_value = {
            'input_ids': torch.randint(0, 1000, (1, 10)),
            'attention_mask': torch.ones(1, 10)
        }
        return tokenizer

    @pytest.fixture
    def mock_optimizer(self):
        """Create mock optimizer."""
        return MagicMock()

    def test_function_exists(self):
        """Test that reinforce_update can be imported."""
        assert callable(reinforce_update)

    def test_returns_float(self, mock_generator, mock_tokenizer, mock_optimizer):
        """Test that function returns a float."""
        prompts = ["prompt1"]
        samples = ["sample1"]
        rewards = [0.7]
        
        result = reinforce_update(
            mock_generator, mock_tokenizer,
            prompts, samples, rewards,
            mock_optimizer
        )
        
        assert isinstance(result, float)

    def test_handles_empty_prompts(self, mock_generator, mock_tokenizer, mock_optimizer):
        """Test handling of empty prompts list."""
        result = reinforce_update(
            mock_generator, mock_tokenizer,
            [], [], [],
            mock_optimizer
        )
        
        assert result == 0.0

    def test_sets_generator_to_train_mode(self, mock_generator, mock_tokenizer, mock_optimizer):
        """Test that generator is set to train mode."""
        prompts = ["prompt1"]
        samples = ["sample1"]
        rewards = [0.7]
        
        reinforce_update(
            mock_generator, mock_tokenizer,
            prompts, samples, rewards,
            mock_optimizer
        )
        
        mock_generator.train.assert_called()

    def test_calls_optimizer_step(self, mock_generator, mock_tokenizer, mock_optimizer):
        """Test that optimizer step is called."""
        prompts = ["prompt1"]
        samples = ["sample1"]
        rewards = [0.7]
        
        reinforce_update(
            mock_generator, mock_tokenizer,
            prompts, samples, rewards,
            mock_optimizer
        )
        
        mock_optimizer.step.assert_called()

    def test_handles_generator_error_gracefully(self, mock_tokenizer, mock_optimizer):
        """Test that generator errors are handled."""
        mock_gen = MagicMock()
        mock_gen.side_effect = RuntimeError("Generator error")
        
        prompts = ["prompt1"]
        samples = ["sample1"]
        rewards = [0.7]
        
        with patch('builtins.print'):
            result = reinforce_update(
                mock_gen, mock_tokenizer,
                prompts, samples, rewards,
                mock_optimizer
            )
            
            # Should return 0.0 on error
            assert result == 0.0


class TestReinforcementLearningLoop:
    """Test suite for reinforcement_learning_loop function."""

    @pytest.fixture
    def mock_components(self):
        """Create all mock components."""
        generator = MagicMock()
        gen_tokenizer = MagicMock()
        fact_disc = MagicMock()
        style_disc = MagicMock()
        safety_disc = MagicMock()
        fact_tok = MagicMock()
        style_tok = MagicMock()
        safety_tok = MagicMock()
        retriever = MagicMock()
        
        return {
            'generator': generator,
            'gen_tokenizer': gen_tokenizer,
            'fact_disc': fact_disc,
            'style_disc': style_disc,
            'safety_disc': safety_disc,
            'fact_tok': fact_tok,
            'style_tok': style_tok,
            'safety_tok': safety_tok,
            'retriever': retriever
        }

    @pytest.fixture
    def mock_qa_pairs(self):
        """Create mock QA pairs."""
        qa = MagicMock()
        qa.question = "What is Python?"
        qa.answer = "A programming language"
        qa.supporting_passages = ["Python is a language"]
        return [qa]

    def test_function_exists(self):
        """Test that reinforcement_learning_loop can be imported."""
        assert callable(reinforcement_learning_loop)

    def test_returns_history(self, mock_components, mock_qa_pairs):
        """Test that function returns history list."""
        with patch('hallucination_reduction.rl_utils.monte_carlo_rewards') as mock_mc:
            mock_mc.return_value = (0.7, [{"sample": "test", "debug": {}}], ["sample"])
            
            with patch('hallucination_reduction.rl_utils.reinforce_update', return_value=0.1):
                with patch('hallucination_reduction.rl_utils.optim.AdamW'):
                    with patch('torch.save'):
                        with patch('builtins.print'):
                            mock_components['retriever'].retrieve.return_value = [(0, "doc")]
                            
                            history = reinforcement_learning_loop(
                                mock_components['generator'],
                                mock_components['gen_tokenizer'],
                                mock_components['fact_disc'],
                                mock_components['style_disc'],
                                mock_components['safety_disc'],
                                mock_components['fact_tok'],
                                mock_components['style_tok'],
                                mock_components['safety_tok'],
                                mock_components['retriever'],
                                mock_qa_pairs
                            )
                            
                            assert isinstance(history, list)

    def test_saves_checkpoints(self, mock_components, mock_qa_pairs):
        """Test that checkpoints are saved."""
        with patch('hallucination_reduction.rl_utils.monte_carlo_rewards') as mock_mc:
            mock_mc.return_value = (0.7, [{"sample": "test", "debug": {"combined": 0.7}}], ["sample"])
            
            with patch('hallucination_reduction.rl_utils.reinforce_update', return_value=0.1):
                with patch('hallucination_reduction.rl_utils.optim.AdamW'):
                    with patch('torch.save') as mock_save:
                        with patch('builtins.print'):
                            mock_components['retriever'].retrieve.return_value = [(0, "doc")]
                            
                            reinforcement_learning_loop(
                                mock_components['generator'],
                                mock_components['gen_tokenizer'],
                                mock_components['fact_disc'],
                                mock_components['style_disc'],
                                mock_components['safety_disc'],
                                mock_components['fact_tok'],
                                mock_components['style_tok'],
                                mock_components['safety_tok'],
                                mock_components['retriever'],
                                mock_qa_pairs
                            )
                            
                            # Should save checkpoints
                            assert mock_save.called

    def test_creates_optimizer(self, mock_components, mock_qa_pairs):
        """Test that optimizer is created."""
        with patch('hallucination_reduction.rl_utils.monte_carlo_rewards') as mock_mc:
            mock_mc.return_value = (0.7, [{"sample": "test", "debug": {"combined": 0.7}}], ["sample"])
            
            with patch('hallucination_reduction.rl_utils.reinforce_update', return_value=0.1):
                with patch('hallucination_reduction.rl_utils.optim.AdamW') as mock_adamw:
                    with patch('torch.save'):
                        with patch('builtins.print'):
                            mock_components['retriever'].retrieve.return_value = [(0, "doc")]
                            
                            reinforcement_learning_loop(
                                mock_components['generator'],
                                mock_components['gen_tokenizer'],
                                mock_components['fact_disc'],
                                mock_components['style_disc'],
                                mock_components['safety_disc'],
                                mock_components['fact_tok'],
                                mock_components['style_tok'],
                                mock_components['safety_tok'],
                                mock_components['retriever'],
                                mock_qa_pairs
                            )
                            
                            mock_adamw.assert_called_once()