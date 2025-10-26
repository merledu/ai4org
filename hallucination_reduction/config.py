import os
import torch
import random
import numpy as np

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
    N_GPUS = torch.cuda.device_count()
    if N_GPUS > 1:
        DEVICE = "cuda"
        MULTI_GPU = True
        print(f"✅ {N_GPUS} GPUs detected. Using all via device_map='auto'.")
    else:
        DEVICE = "cuda"
        MULTI_GPU = False
        print("✅ Single GPU detected. Using cuda:0")
else:
    DEVICE = "cpu"
    MULTI_GPU = False
    N_GPUS = 0
    print("⚠️ No GPU detected. Using CPU.")

# -------------------------
# Model configuration
# -------------------------
GEN_MODEL = os.environ.get("GEN_MODEL", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")
DISC_MODEL = os.environ.get("DISC_MODEL", "distilbert-base-uncased")
SAVE_DIR = "./saved_models_improved"
os.makedirs(SAVE_DIR, exist_ok=True)
GEN_MODEL_PATH = os.path.join(SAVE_DIR, "generator_final.pt")

# -------------------------
# Corpus
# -------------------------
CORPUS_PATH = "./data/processed/corpus.txt"

# -------------------------
# Hyperparameters
# -------------------------
SFT_EPOCHS = 4
SFT_BATCH = 1
SFT_LR = 3e-5

DISC_EPOCHS = 4
DISC_BATCH = 8
DISC_LR = 2e-5

MC_ROLLOUTS = 6
GEN_BATCH_SIZE = 1
GEN_LR = 1e-5
MAX_GEN_TOKENS = 64
MIN_GEN_TOKENS = 5
TOP_K = 3
RL_EPOCHS = 4

# -------------------------
# Reward weights
# -------------------------
FACT_WEIGHT = 0.8
STYLE_WEIGHT = 0.15
SAFETY_WEIGHT = 0.05
HARD_PENALTY_IF_FACT_LT = 0.4  # subtract if p_fact < 0.5


#LORA configuration

LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = ["q_proj", "v_proj"]  # or appropriate for your model
LORA_TASK_TYPE = "CAUSAL_LM"




# -------------------------
# Backend tuning
# -------------------------
try:
    torch.backends.cudnn.benchmark = True
except Exception:
    pass
