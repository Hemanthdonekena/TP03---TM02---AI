import os

MODEL_DIR = os.getenv("MODEL_DIR", "data")
N_CLUSTERS = int(os.getenv("N_CLUSTERS", "5"))
TOP_K = int(os.getenv("TOP_K", "5"))
RANDOM_STATE = 42
