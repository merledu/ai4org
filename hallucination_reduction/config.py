import os
import torch

SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

GEN_MODEL = os.environ.get("GEN_MODEL", "gpt2")
DISC_MODEL = os.environ.get("DISC_MODEL", "distilbert-base-uncased")

SFT_EPOCHS = 4
SFT_BATCH = 2
SFT_LR = 3e-5

DISC_EPOCHS = 10
DISC_BATCH = 8
DISC_LR = 2e-5

MC_ROLLOUTS = 6
GEN_BATCH_SIZE = 2
GEN_LR = 1e-5
MAX_GEN_TOKENS = 64
MIN_GEN_TOKENS = 5
TOP_K = 3
RL_EPOCHS = 6

FACT_WEIGHT = 0.8
STYLE_WEIGHT = 0.15
SAFETY_WEIGHT = 0.05
HARD_PENALTY_IF_FACT_LT = 0.4

SAVE_DIR = "./saved_models_improved"
os.makedirs(SAVE_DIR, exist_ok=True)
