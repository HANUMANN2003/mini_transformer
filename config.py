"""
config.py
----------
All hyperparameters for the mini transformer LLM live here so you
can tweak the model size / training setup from a single place.
"""

import torch

# ----------------------- Data -----------------------
DATA_PATH = "data/input.txt"     # plain text training corpus
TRAIN_SPLIT = 0.9                # 90% train / 10% val

# ----------------------- Model -----------------------
BLOCK_SIZE = 128        # context length (max tokens the model attends over)
N_EMBED = 256           # embedding dimension (d_model)
N_HEAD = 8               # number of attention heads
N_LAYER = 6              # number of transformer blocks
DROPOUT = 0.1

# ----------------------- Training -----------------------
BATCH_SIZE = 64
MAX_ITERS = 5000
EVAL_INTERVAL = 250
EVAL_ITERS = 100
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 0.01
GRAD_CLIP = 1.0

# ----------------------- Generation -----------------------
MAX_NEW_TOKENS = 500
TEMPERATURE = 0.6
TOP_K = 40

# ----------------------- Misc -----------------------
SEED = 1337
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CHECKPOINT_PATH = "checkpoints/model.pt"