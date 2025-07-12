# config.py

import warnings
import torch

# Ignore specific warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# Configuration Constants
DATASET_FILENAME = './data/Loan_default.csv'
SAMPLE_SIZE = 10000
NUM_CLIENTS = 12
NUM_MIDDLE_SERVERS = 3
COMMUNICATION_ROUNDS = 10
LOCAL_EPOCHS = 30
LEARNING_RATE = 0.001
NUM_SAMPLED_CLIENTS_PER_ROUND = 10
BATCH_SIZE = 32
TEST_SPLIT_RATIO = 0.2
RANDOM_STATE = 42
TARGET_COLUMN = 'Default'

print(f"PyTorch Version: {torch.__version__}")
print("\n--- Configuration ---")
print(f"Total Clients: {NUM_CLIENTS}, Middle Servers: {NUM_MIDDLE_SERVERS}")
print(f"Sampled Clients per Round: {NUM_SAMPLED_CLIENTS_PER_ROUND}")
print(f"Rounds: {COMMUNICATION_ROUNDS}, Local Epochs: {LOCAL_EPOCHS}")
print(f"Learning Rate: {LEARNING_RATE} (Constant)")
print(f"Target Column: '{TARGET_COLUMN}'")
print(f"Dataset Sample Size: {SAMPLE_SIZE}")
