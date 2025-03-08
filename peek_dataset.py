import numpy as np
import tiktoken

PEEK_DATASET_PATH = "natural_reasoning/natural_reasoning_train_000001.npy"
enc = tiktoken.get_encoding("gpt2")
arr = np.load(PEEK_DATASET_PATH)
print("Array shape:", arr.shape)
print("\nFirst few tokens decoded:", enc.decode(arr[:1000]))
