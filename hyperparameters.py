import torch

# Hyperparameters
NUM_ENVS = 4
BATCH_SIZE = 128
LEARNING_RATE = 5E-5
TRAIN_STEPS = 10000000000000
# MAX_STEPS = 200
MIN_REPLAY_MEMORY_SIZE = 50000
MAX_REPLAY_MEMORY_SIZE = int(1E6)
UPDATE_TARGET_NETWORK = 10000 // NUM_ENVS
RETURN_DISCOUNT = 0.99

MAX_EPSILON = 1
MIN_EPSILON = 0.1
EPSILON_DECAY = int(1E6)

PRINT_LOGS_STEPS = 1000
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SAVE_PATH = "./models/atari_model.pth"
SAVE_INTERVAL = 10000