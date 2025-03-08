"""
Natural Reasoning dataset (for srs pretraining)
https://huggingface.co/datasets/facebook/natural_reasoning
Downloads and tokenizes the data and saves data shards to disk.
Run simply as:
$ python natural_reasoning.py
Will save shards to the local directory "natural_reasoning".
"""

import os
import multiprocessing as mp
import numpy as np
import tiktoken
from datasets import load_dataset
from tqdm import tqdm

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

DATASET_PATH = os.environ.get("DATASET_PATH", "natural_reasoning")
HF_DATASET_NAME = os.environ.get("HF_DATASET_NAME", "facebook/natural_reasoning")
SHARD_SIZE = int(float(os.environ.get("SHARD_SIZE", "1e8")))


def init_directories(local_dir):
    """Initialize the data cache directory."""
    data_cache_dir = os.path.join(os.path.dirname(__file__), local_dir)
    os.makedirs(data_cache_dir, exist_ok=True)
    return data_cache_dir


def init_tokenizer():
    """Initialize the tokenizer."""
    enc = tiktoken.get_encoding("gpt2")
    eot = enc._special_tokens["<|endoftext|>"]
    return enc, eot


def tokenize(doc):
    """Tokenize a single document and return numpy array of uint16 tokens."""
    enc, eot = init_tokenizer()

    # Combine the relevant fields into a single text
    # the facebook dataset follow the format: 'question', 'reference_answer', and 'responses'
    text_parts = [
        doc["question"],
        "Answer:",
        doc["reference_answer"],
        "Model Response:",
        doc["responses"][0]["response"],  # Taking the first response
    ]
    combined_text = "\n".join(text_parts)

    tokens = [eot]  # the special <|endoftext|> token delimits all documents
    tokens.extend(enc.encode_ordinary(combined_text))
    tokens_np = np.array(tokens)
    assert (0 <= tokens_np).all() and (
        tokens_np < 2**16
    ).all(), "token dictionary too large for uint16"
    tokens_np_uint16 = tokens_np.astype(np.uint16)
    return tokens_np_uint16


def write_datafile(filename, tokens_np):
    """Write tokens to a file."""
    np.save(filename, tokens_np)


def process_dataset(local_dir=DATASET_PATH, shard_size=SHARD_SIZE):
    """Main function to process the dataset."""
    data_cache_dir = init_directories(local_dir)

    # Load the dataset
    fw = load_dataset(HF_DATASET_NAME, split="train")

    # Initialize multiprocessing
    nprocs = max(1, os.cpu_count() // 2)

    with mp.Pool(nprocs) as pool:
        shard_index = 0
        # preallocate buffer to hold current shard
        all_tokens_np = np.empty((shard_size,), dtype=np.uint16)
        token_count = 0
        progress_bar = None

        for tokens in pool.imap(tokenize, fw, chunksize=16):
            # is there enough space in the current shard for the new tokens?
            if token_count + len(tokens) < shard_size:
                # simply append tokens to current shard
                all_tokens_np[token_count : token_count + len(tokens)] = tokens
                token_count += len(tokens)
                # update progress bar
                if progress_bar is None:
                    progress_bar = tqdm(
                        total=shard_size, unit="tokens", desc=f"Shard {shard_index}"
                    )
                progress_bar.update(len(tokens))
            else:
                # write the current shard and start a new one
                split = "val" if shard_index == 0 else "train"
                filename = os.path.join(
                    data_cache_dir, f"{DATASET_PATH}_{split}_{shard_index:06d}"
                )
                # split the document into whatever fits in this shard; the remainder goes to next one
                remainder = shard_size - token_count
                progress_bar.update(remainder)
                all_tokens_np[token_count : token_count + remainder] = tokens[
                    :remainder
                ]
                write_datafile(filename, all_tokens_np)
                shard_index += 1
                progress_bar = None
                # populate the next shard with the leftovers of the current doc
                all_tokens_np[0 : len(tokens) - remainder] = tokens[remainder:]
                token_count = len(tokens) - remainder

        # write any remaining tokens as the last shard
        if token_count != 0:
            split = "val" if shard_index == 0 else "train"
            filename = os.path.join(
                data_cache_dir, f"{DATASET_PATH}_{split}_{shard_index:06d}"
            )
            write_datafile(filename, all_tokens_np[:token_count])


if __name__ == "__main__":
    process_dataset()
