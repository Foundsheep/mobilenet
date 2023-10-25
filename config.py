import torch

ROOT = "./"
LEARNING_RATE = 1e-3
BATCH_SIZE = 32
EPOCHS = 10
WIDTH_MULTIPLIER = 1
RESOLUTION_MULTIPLIER = 1
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"