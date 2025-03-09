import os
import torch
import torch.nn as nn
from torch.nn import functional as F
import tiktoken
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()

# Model path will default to the latest checkpoint in the log directory
TESTING_MODEL_PATH = os.environ.get("TESTING_MODEL_PATH", "log/")

BLOCK_SIZE: int = int(os.environ.get("BLOCK_SIZE", "1024"))  # max sequence length
VOCAB_SIZE: int = int(os.environ.get("VOCAB_SIZE", "50304"))  # GPT-2 vocab size
N_LAYER: int = int(os.environ.get("N_LAYER", "12"))  # number of layers
N_HEAD: int = int(os.environ.get("N_HEAD", "12"))  # number of heads
N_EMBD: int = int(os.environ.get("N_EMBD", "768"))  # embedding dimension
BIAS: bool = os.environ.get("BIAS", "False") == "True"


# Minimal model architecture needed for inference
@dataclass
class GPTConfig:
    block_size: int = BLOCK_SIZE  # matches your training config
    vocab_size: int = VOCAB_SIZE  # updated to match checkpoint
    n_layer: int = N_LAYER
    n_head: int = N_HEAD
    n_embd: int = N_EMBD
    bias: bool = BIAS


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.n_head = config.n_head
        self.n_embd = config.n_embd

    def forward(self, x):
        B, T, C = x.size()
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)
        return y


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU(approximate="tanh")
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.vocab_size, config.n_embd),
                wpe=nn.Embedding(config.block_size, config.n_embd),
                h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
                ln_f=nn.LayerNorm(config.n_embd),
            )
        )
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight

    def forward(self, idx, targets=None):
        B, T = idx.size()
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(pos)
        x = tok_emb + pos_emb
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        return logits


def load_model(checkpoint_path: str) -> GPT:
    """Load the trained model from a checkpoint."""
    # Setup device
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    print(f"Using device: {device}")

    # Load checkpoint with weights_only=False (for trusted checkpoints)
    print("Loading checkpoint...")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
    # If we're given a directory, use the latest checkpoint
    if os.path.isdir(checkpoint_path):
        # Look for the `model_<step>.pt` file with the highest step number
        checkpoint_files = [
            f for f in os.listdir(checkpoint_path) if f.startswith("model_")
        ]
        if not checkpoint_files:
            raise FileNotFoundError(
                f"No checkpoint files found in directory: {checkpoint_path}"
            )
        checkpoint_path = os.path.join(checkpoint_path, sorted(checkpoint_files)[-1])
    print(f"Loaded checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, weights_only=False)

    # Create and load model
    print("Initializing model...")
    model = GPT(GPTConfig())
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    model.eval()
    print("Model loaded successfully!")
    return model


def generate_text(model, prompt, max_tokens=100, temperature=0.8):
    """Generate text from a prompt."""
    enc = tiktoken.get_encoding("gpt2")
    # Allow the endoftext token in encoding
    encoded = enc.encode(prompt, allowed_special={"<|endoftext|>"})
    tokens = torch.tensor(encoded).unsqueeze(0)

    device = next(model.parameters()).device
    tokens = tokens.to(device)

    # Get the endoftext token id once
    endoftext_token = enc.encode("<|endoftext|>", allowed_special={"<|endoftext|>"})[0]

    with torch.no_grad():
        for _ in range(max_tokens):
            logits = model(tokens)
            logits = logits[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            tokens = torch.cat([tokens, next_token], dim=1)
            if next_token.item() == endoftext_token:
                break

    return enc.decode(tokens[0].tolist())


def answer_question(model, question, max_tokens=200):
    """Use the model to answer a question."""
    prompt = f"Question: {question}\nAnswer:"
    return generate_text(model, prompt, max_tokens=max_tokens)


if __name__ == "__main__":
    print("Starting model generation demo...")
    model = load_model(TESTING_MODEL_PATH)

    print("\nExample 1: Direct text generation")
    prompt = "What is the relationship between kinetic and potential energy in a"
    print("Prompt:", prompt)
    print("Response:", generate_text(model, prompt))
    print("-" * 60)

    print("\nExample 2: Question answering")
    question = "Explain how conservation of energy applies to a pendulum's motion."
    print("Question:", question)
    print("Response:", answer_question(model, question))
    print("-" * 60)
