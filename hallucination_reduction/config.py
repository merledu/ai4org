import os
import random
import numpy as np
import torch

# -------------------------
# Reproducibility
# -------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# -------------------------
# Device configuration
# -------------------------
if torch.cuda.is_available():
    n_gpus = torch.cuda.device_count()
    if n_gpus > 1:
        DEVICE = "cuda"  # DataParallel / ROCm uses 'cuda' name
        print(f"✅ {n_gpus} GPUs detected. Using all via DataParallel (generator).")
    else:
        DEVICE = "cuda"
        print("✅ Single GPU detected. Using cuda:0")
else:
    DEVICE = "cpu"
    print("⚠️ No GPU detected. Using CPU.")

# -------------------------
# Model configuration
# -------------------------
GEN_MODEL = os.environ.get("GEN_MODEL", "gpt2")
DISC_MODEL = os.environ.get("DISC_MODEL", "distilbert-base-uncased")

# -------------------------
# Hyperparameters
# -------------------------
SFT_EPOCHS = 8
SFT_BATCH = 1       # keep small for safety; increase to fill GPU
SFT_LR = 3e-5

DISC_EPOCHS = 10
DISC_BATCH = 8      # larger batch to keep GPU utilization up
DISC_LR = 2e-5

MC_ROLLOUTS = 6
GEN_BATCH_SIZE = 1
GEN_LR = 1e-5
MAX_GEN_TOKENS = 64
MIN_GEN_TOKENS = 5
TOP_K = 3
RL_EPOCHS = 10

# -------------------------
# Reward weights
# -------------------------
FACT_WEIGHT = 0.8
STYLE_WEIGHT = 0.15
SAFETY_WEIGHT = 0.05
HARD_PENALTY_IF_FACT_LT = 0.4  # subtract if p_fact < 0.5

# -------------------------
# Output / Save directory
# -------------------------
SAVE_DIR = "./saved_models_improved"
os.makedirs(SAVE_DIR, exist_ok=True)

# Enable backend tuning (ROCm/PyTorch will use appropriate backend)
try:
    torch.backends.cudnn.benchmark = True
except Exception:
    pass
