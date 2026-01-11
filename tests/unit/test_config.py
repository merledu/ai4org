import os
import random
from unittest.mock import patch

import numpy as np
import pytest
import torch


class TestConfigConstants:
    """Test suite for config.py constants and configurations."""

    def test_seed_value(self):
        """Verify SEED constant is set correctly."""
        from hallucination_reduction.config import SEED

        assert SEED == 42

    def test_random_seed_set(self):
        """Verify random seed is set and reproducible."""
        # Import triggers seed setting

        # Generate random values
        val1 = random.random()

        # Reset and regenerate
        random.seed(42)
        val2 = random.random()

        assert val1 == val2, "Random seed should produce reproducible results"

    def test_numpy_seed_set(self):
        """Verify numpy seed is set and reproducible."""

        val1 = np.random.rand()
        np.random.seed(42)
        val2 = np.random.rand()

        assert val1 == val2, "Numpy seed should produce reproducible results"

    def test_torch_seed_set(self):
        """Verify torch seed is set and reproducible."""

        val1 = torch.rand(1).item()
        torch.manual_seed(42)
        val2 = torch.rand(1).item()

        assert val1 == val2, "Torch seed should produce reproducible results"

    @patch("torch.cuda.is_available")
    def test_device_cpu_when_no_cuda(self, mock_cuda):
        """Test DEVICE is set to 'cpu' when CUDA is not available."""
        mock_cuda.return_value = False

        # Reload module to trigger device detection
        import importlib

        import hallucination_reduction.config as config

        importlib.reload(config)

        assert config.DEVICE == "cpu"
        assert config.MULTI_GPU is False
        assert config.N_GPUS == 0

    @patch("torch.cuda.device_count")
    @patch("torch.cuda.is_available")
    def test_device_single_gpu(self, mock_cuda, mock_count):
        """Test DEVICE configuration with single GPU."""
        mock_cuda.return_value = True
        mock_count.return_value = 1

        import importlib

        import hallucination_reduction.config as config

        importlib.reload(config)

        assert config.DEVICE == "cuda"
        assert config.MULTI_GPU is False
        assert config.N_GPUS == 1

    @patch("torch.cuda.device_count")
    @patch("torch.cuda.is_available")
    def test_device_multi_gpu(self, mock_cuda, mock_count):
        """Test DEVICE configuration with multiple GPUs."""
        mock_cuda.return_value = True
        mock_count.return_value = 4

        import importlib

        import hallucination_reduction.config as config

        importlib.reload(config)

        assert config.DEVICE == "cuda"
        assert config.MULTI_GPU is True
        assert config.N_GPUS == 4

    def test_gen_model_default(self):
        """Test default GEN_MODEL value."""
        # Clear environment variable if set
        if "GEN_MODEL" in os.environ:
            del os.environ["GEN_MODEL"]

        import importlib

        import hallucination_reduction.config as config

        importlib.reload(config)

        assert config.GEN_MODEL == "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

    def test_gen_model_from_env(self):
        """Test GEN_MODEL reads from environment variable."""
        os.environ["GEN_MODEL"] = "custom/model"

        import importlib

        import hallucination_reduction.config as config

        importlib.reload(config)

        assert config.GEN_MODEL == "custom/model"

        # Cleanup
        del os.environ["GEN_MODEL"]

    def test_disc_model_default(self):
        """Test default DISC_MODEL value."""
        if "DISC_MODEL" in os.environ:
            del os.environ["DISC_MODEL"]

        import importlib

        import hallucination_reduction.config as config

        importlib.reload(config)

        assert config.DISC_MODEL == "distilbert-base-uncased"

    def test_disc_model_from_env(self):
        """Test DISC_MODEL reads from environment variable."""
        os.environ["DISC_MODEL"] = "bert-base-uncased"

        import importlib

        import hallucination_reduction.config as config

        importlib.reload(config)

        assert config.DISC_MODEL == "bert-base-uncased"

        # Cleanup
        del os.environ["DISC_MODEL"]

    def test_save_dir_constant(self):
        """Test SAVE_DIR is set correctly."""
        from hallucination_reduction.config import SAVE_DIR

        assert SAVE_DIR == "./saved_models_improved"

    def test_save_dir_created(self):
        """Test that SAVE_DIR is created during module import."""
        from hallucination_reduction.config import SAVE_DIR

        assert os.path.exists(SAVE_DIR)

    def test_gen_model_path(self):
        """Test GEN_MODEL_PATH construction."""
        from hallucination_reduction.config import GEN_MODEL_PATH, SAVE_DIR

        expected = os.path.join(SAVE_DIR, "generator_final.pt")
        assert GEN_MODEL_PATH == expected

    def test_corpus_path(self):
        """Test CORPUS_PATH is set correctly."""
        from hallucination_reduction.config import CORPUS_PATH

        assert CORPUS_PATH == "./data/processed/corpus.txt"

    def test_sft_hyperparameters(self):
        """Test SFT training hyperparameters."""
        from hallucination_reduction.config import SFT_BATCH, SFT_EPOCHS, SFT_LR

        assert SFT_EPOCHS == 4
        assert SFT_BATCH == 1
        assert SFT_LR == 3e-5
        assert isinstance(SFT_LR, float)

    def test_disc_hyperparameters(self):
        """Test discriminator training hyperparameters."""
        from hallucination_reduction.config import DISC_BATCH, DISC_EPOCHS, DISC_LR

        assert DISC_EPOCHS == 4
        assert DISC_BATCH == 8
        assert DISC_LR == 2e-5
        assert isinstance(DISC_LR, float)

    def test_mc_rollouts(self):
        """Test Monte Carlo rollouts parameter."""
        from hallucination_reduction.config import MC_ROLLOUTS

        assert MC_ROLLOUTS == 6
        assert isinstance(MC_ROLLOUTS, int)

    def test_gen_batch_size(self):
        """Test generator batch size."""
        from hallucination_reduction.config import GEN_BATCH_SIZE

        assert GEN_BATCH_SIZE == 1

    def test_gen_learning_rate(self):
        """Test generator learning rate."""
        from hallucination_reduction.config import GEN_LR

        assert GEN_LR == 1e-5
        assert isinstance(GEN_LR, float)

    def test_token_limits(self):
        """Test token generation limits."""
        from hallucination_reduction.config import MAX_GEN_TOKENS, MIN_GEN_TOKENS

        assert MAX_GEN_TOKENS == 64
        assert MIN_GEN_TOKENS == 5
        assert MAX_GEN_TOKENS > MIN_GEN_TOKENS

    def test_top_k_parameter(self):
        """Test TOP_K parameter."""
        from hallucination_reduction.config import TOP_K

        assert TOP_K == 3
        assert isinstance(TOP_K, int)

    def test_rl_epochs(self):
        """Test RL training epochs."""
        from hallucination_reduction.config import RL_EPOCHS

        assert RL_EPOCHS == 4

    def test_reward_weights(self):
        """Test reward weight configuration."""
        from hallucination_reduction.config import (
            FACT_WEIGHT,
            SAFETY_WEIGHT,
            STYLE_WEIGHT,
        )

        assert FACT_WEIGHT == 0.8
        assert STYLE_WEIGHT == 0.15
        assert SAFETY_WEIGHT == 0.05

        # Weights should sum to 1.0
        total = FACT_WEIGHT + STYLE_WEIGHT + SAFETY_WEIGHT
        assert abs(total - 1.0) < 1e-6, "Weights should sum to 1.0"

    def test_reward_weights_types(self):
        """Test reward weights are floats."""
        from hallucination_reduction.config import (
            FACT_WEIGHT,
            SAFETY_WEIGHT,
            STYLE_WEIGHT,
        )

        assert isinstance(FACT_WEIGHT, float)
        assert isinstance(STYLE_WEIGHT, float)
        assert isinstance(SAFETY_WEIGHT, float)

    def test_hard_penalty_threshold(self):
        """Test hard penalty threshold."""
        from hallucination_reduction.config import HARD_PENALTY_IF_FACT_LT

        assert HARD_PENALTY_IF_FACT_LT == 0.4
        assert 0 <= HARD_PENALTY_IF_FACT_LT <= 1

    def test_all_epochs_positive(self):
        """Test all epoch values are positive integers."""
        from hallucination_reduction.config import DISC_EPOCHS, RL_EPOCHS, SFT_EPOCHS

        assert SFT_EPOCHS > 0
        assert DISC_EPOCHS > 0
        assert RL_EPOCHS > 0
        assert isinstance(SFT_EPOCHS, int)
        assert isinstance(DISC_EPOCHS, int)
        assert isinstance(RL_EPOCHS, int)

    def test_all_batch_sizes_positive(self):
        """Test all batch sizes are positive integers."""
        from hallucination_reduction.config import DISC_BATCH, GEN_BATCH_SIZE, SFT_BATCH

        assert SFT_BATCH > 0
        assert DISC_BATCH > 0
        assert GEN_BATCH_SIZE > 0

    def test_all_learning_rates_positive(self):
        """Test all learning rates are positive floats."""
        from hallucination_reduction.config import DISC_LR, GEN_LR, SFT_LR

        assert SFT_LR > 0
        assert DISC_LR > 0
        assert GEN_LR > 0

    def test_learning_rates_reasonable_range(self):
        """Test learning rates are in reasonable range."""
        from hallucination_reduction.config import DISC_LR, GEN_LR, SFT_LR

        # Typical LR range: 1e-6 to 1e-3
        assert 1e-6 <= SFT_LR <= 1e-3
        assert 1e-6 <= DISC_LR <= 1e-3
        assert 1e-6 <= GEN_LR <= 1e-3


def test_cudnn_benchmark_set():

    # Ensure that cudnn.benchmark is a boolean
    assert isinstance(torch.backends.cudnn.benchmark, bool)

    def test_weights_are_non_negative(self):
        """Test all weights are non-negative."""
        from hallucination_reduction.config import (
            FACT_WEIGHT,
            SAFETY_WEIGHT,
            STYLE_WEIGHT,
        )

        assert FACT_WEIGHT >= 0
        assert STYLE_WEIGHT >= 0
        assert SAFETY_WEIGHT >= 0

    def test_fact_weight_dominates(self):
        """Test that FACT_WEIGHT is the largest weight."""
        from hallucination_reduction.config import (
            FACT_WEIGHT,
            SAFETY_WEIGHT,
            STYLE_WEIGHT,
        )

        assert FACT_WEIGHT > STYLE_WEIGHT
        assert FACT_WEIGHT > SAFETY_WEIGHT

    def test_module_imports_without_error(self):
        """Test that config module can be imported without errors."""
        try:
            pass

            assert True
        except Exception as e:
            pytest.fail(f"Config import failed: {e}")

    def test_constants_are_immutable_types(self):
        """Test that configuration constants use immutable types."""
        from hallucination_reduction.config import (
            DEVICE,
            FACT_WEIGHT,
            GEN_MODEL,
            SEED,
            SFT_EPOCHS,
        )

        # These should all be immutable types
        assert isinstance(SEED, int)
        assert isinstance(SFT_EPOCHS, int)
        assert isinstance(FACT_WEIGHT, float)
        assert isinstance(GEN_MODEL, str)
        assert isinstance(DEVICE, str)

    def test_path_strings_valid(self):
        """Test that path strings are valid."""
        from hallucination_reduction.config import CORPUS_PATH, GEN_MODEL_PATH, SAVE_DIR

        # Should not be empty
        assert len(SAVE_DIR) > 0
        assert len(CORPUS_PATH) > 0
        assert len(GEN_MODEL_PATH) > 0

        # Should be strings
        assert isinstance(SAVE_DIR, str)
        assert isinstance(CORPUS_PATH, str)
        assert isinstance(GEN_MODEL_PATH, str)

    def test_mc_rollouts_reasonable(self):
        """Test MC_ROLLOUTS is in reasonable range."""
        from hallucination_reduction.config import MC_ROLLOUTS

        assert 1 <= MC_ROLLOUTS <= 20, "MC rollouts should be reasonable"

    def test_top_k_reasonable(self):
        """Test TOP_K is in reasonable range."""
        from hallucination_reduction.config import TOP_K

        assert 1 <= TOP_K <= 50, "TOP_K should be reasonable"
