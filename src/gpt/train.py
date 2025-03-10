import os
import math
import time
import inspect
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
from dotenv import load_dotenv
import numpy as np
from collections import deque
from torch.amp import GradScaler
from .hellaswag import render_example, iterate_examples

load_dotenv()

# -----------------------------------------------------------------------------

# Model config
BLOCK_SIZE: int = int(os.environ.get("BLOCK_SIZE", "1024"))  # max sequence length
N_LAYER: int = int(os.environ.get("N_LAYER", "12"))  # number of layers
N_HEAD: int = int(os.environ.get("N_HEAD", "12"))  # number of heads
N_EMBD: int = int(os.environ.get("N_EMBD", "768"))  # embedding dimension
BIAS: bool = os.environ.get("BIAS", "False") == "True"

# Training hyperparameters from environment variables
TOTAL_BATCH_SIZE = int(
    os.environ.get("TOTAL_BATCH_SIZE", "524288")
)  # 2**19, ~0.5M tokens
B_CONFIG = int(os.environ.get("BATCH_SIZE", "64"))  # micro batch size
T_CONFIG = int(os.environ.get("SEQ_LENGTH", "1024"))  # sequence length
MAX_LR = float(os.environ.get("MAX_LR", "6e-4"))
MIN_LR = float(
    os.environ.get("MIN_LR", str(MAX_LR * 0.1))
)  # MIN_LR as 10% of MAX_LR if not explicitly set
WARMUP_STEPS = int(os.environ.get("WARMUP_STEPS", "715"))
MAX_STEPS = int(os.environ.get("MAX_STEPS", "19073"))  # ~1 epoch for 10B tokens
SAVE_CHECKPOINT_INTERVAL = int(os.environ.get("SAVE_CHECKPOINT_INTERVAL", "5000"))
RUN_EVAL_INTERVAL = int(os.environ.get("RUN_EVAL_INTERVAL", "250"))
WEIGHT_DECAY = float(os.environ.get("WEIGHT_DECAY", "0.1"))

# Dataset path, and generation if not exists
DATASET_PATH = os.environ.get("DATASET_PATH", "natural_reasoning")
if not os.path.exists(DATASET_PATH):
    raise FileNotFoundError(f"Dataset path not found: ${DATASET_PATH}")


# Track gradient norms for adaptive clipping
grad_norm_history = deque(maxlen=100)
CLIP_THRESHOLD = float(os.environ.get("GRAD_CLIP", "1.0"))
ADAPTIVE_CLIP_MULTIPLIER = 2.0

# Constants for vocabulary
GPT2_VOCAB_SIZE = 50257  # Original GPT-2 vocabulary size
# Note: 50257 is the default vocab size of the GPT-2 tokenizer, but we pad it to next multiple of 64 for efficiency
PADDED_VOCAB_SIZE = 50304


# -----------------------------------------------------------------------------


class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1
        # regularization
        self.n_head = config.n_head
        self.n_embd = config.n_embd

    def forward(self, x):
        B, T, C = (
            x.size()
        )  # batch size, sequence length, embedding dimensionality (n_embd)
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        # nh is "number of heads", hs is "head size", and C (number of channels) = nh * hs
        # e.g. in GPT-2 (124M), n_head=12, hs=64, so nh*hs=C=768 channels in the Transformer
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # (B, nh, T, hs)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)  # flash attention
        y = (
            y.transpose(1, 2).contiguous().view(B, T, C)
        )  # re-assemble all head outputs side by side
        # output projection
        y = self.c_proj(y)
        return y


class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU(approximate="tanh")
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1

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


@dataclass
class GPTConfig:
    block_size: int = BLOCK_SIZE  # max sequence length
    vocab_size: int = PADDED_VOCAB_SIZE  # Padded vocab size for efficiency
    n_layer: int = N_LAYER  # number of layers
    n_head: int = N_HEAD  # number of heads
    n_embd: int = N_EMBD  # embedding dimension
    bias: bool = BIAS  # whether to use bias in linear layers


class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config

        # Create embeddings with padded vocab size for efficiency
        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.vocab_size, config.n_embd),
                wpe=nn.Embedding(config.block_size, config.n_embd),
                h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
                ln_f=nn.LayerNorm(config.n_embd),
            )
        )
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=config.bias)

        # Zero out embeddings for padding tokens
        with torch.no_grad():
            self.transformer.wte.weight[GPT2_VOCAB_SIZE:].zero_()
            self.lm_head.weight[GPT2_VOCAB_SIZE:].zero_()

        # weight sharing scheme
        self.transformer.wte.weight = self.lm_head.weight

        # init params
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, "NANOGPT_SCALE_INIT"):
                std *= (2 * self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        # idx is of shape (B, T)
        B, T = idx.size()
        assert (
            T <= self.config.block_size
        ), f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"
        # forward the token and posisition embeddings
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)  # shape (T)
        pos_emb = self.transformer.wpe(pos)  # position embeddings of shape (T, n_embd)
        tok_emb = self.transformer.wte(idx)  # token embeddings of shape (B, T, n_embd)
        x = tok_emb + pos_emb
        # forward the blocks of the transformer
        for block in self.transformer.h:
            x = block(x)
        # forward the final layernorm and the classifier
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)  # (B, T, vocab_size)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @classmethod
    def from_pretrained(cls, model_type):
        """Loads pretrained GPT-2 model weights from huggingface"""
        assert model_type in {"gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"}
        from transformers import GPT2LMHeadModel

        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            "gpt2": dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            "gpt2-medium": dict(n_layer=24, n_head=16, n_embd=1024),  # 350M params
            "gpt2-large": dict(n_layer=36, n_head=20, n_embd=1280),  # 774M params
            "gpt2-xl": dict(n_layer=48, n_head=25, n_embd=1600),  # 1558M params
        }[model_type]
        config_args["vocab_size"] = (
            GPT2_VOCAB_SIZE  # always 50257 for GPT model checkpoints
        )
        config_args["block_size"] = 1024  # always 1024 for GPT model checkpoints
        # create a from-scratch initialized minGPT model
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [
            k for k in sd_keys if not k.endswith(".attn.bias")
        ]  # discard this mask / buffer, not a param

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [
            k for k in sd_keys_hf if not k.endswith(".attn.masked_bias")
        ]  # ignore these, just a buffer
        sd_keys_hf = [
            k for k in sd_keys_hf if not k.endswith(".attn.bias")
        ]  # same, just the mask (buffer)
        transposed = [
            "attn.c_attn.weight",
            "attn.c_proj.weight",
            "mlp.c_fc.weight",
            "mlp.c_proj.weight",
        ]
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(
            sd_keys
        ), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model

    def configure_optimizers(self, weight_decay, learning_rate, device_type):
        # start with all of the candidate parameters (that require grad)
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        if master_process:
            print(
                f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters"
            )
            print(
                f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters"
            )
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == "cuda"
        if master_process:
            print(f"using fused AdamW: {use_fused}")
        optimizer = torch.optim.AdamW(
            optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=use_fused
        )
        return optimizer


def calculate_parameter_count(config: GPTConfig):
    """Calculate and display detailed parameter counts and training statistics for the model."""
    vocab_size = config.vocab_size
    block_size = config.block_size
    n_layer = config.n_layer
    n_embd = config.n_embd
    n_head = config.n_head
    bias = config.bias

    # Token + Position Embeddings (wte is shared with lm_head through weight tying)
    embedding_params = (vocab_size + block_size) * n_embd

    # Per transformer block parameters
    attn_params = (n_embd * (3 * n_embd)) + (  # QKV projection
        n_embd * n_embd
    )  # Output projection
    if bias:
        attn_params += (3 * n_embd) + n_embd  # QKV and output projection biases

    mlp_params = (n_embd * (4 * n_embd)) + (  # First linear
        (4 * n_embd) * n_embd
    )  # Second linear
    if bias:
        mlp_params += (4 * n_embd) + n_embd  # First and second linear biases

    # Layer norms (gamma and beta)
    ln_params = 4 * n_embd  # Two layer norms per block + final layer norm

    # Total parameters per block
    block_params = attn_params + mlp_params + ln_params

    # Final layer norm parameters
    final_ln_params = 2 * n_embd

    # Calculate totals
    total_embedding = embedding_params
    total_blocks = n_layer * block_params
    total_params = total_embedding + total_blocks + final_ln_params

    # Model architecture comparisons
    gpt2_sizes = {"small": 124, "medium": 350, "large": 774, "xl": 1558}

    # Print breakdown
    print("\nModel Scale Analysis:")
    print(f"Architecture:")
    print(f"- Layers:           {n_layer}")
    print(f"- Embedding dim:    {n_embd}")
    print(f"- Attention heads:  {n_head}")
    print(f"- Head dim:         {n_embd // n_head}")
    print(f"- Sequence length:  {block_size}")
    print(f"- Vocab size:       {vocab_size}")

    print("\nParameter count breakdown:")
    print(f"- Embeddings:      {total_embedding:,} parameters")
    print(f"- Each block:      {block_params:,} parameters")
    print(f"  • Attention:     {attn_params:,}")
    print(f"  • MLP:           {mlp_params:,}")
    print(f"  • Layer norms:   {ln_params:,}")
    print(f"- All {n_layer} blocks:   {total_blocks:,} parameters")
    print(f"- Final layer norm: {final_ln_params:,} parameters")

    print(f"\nTotal model size: {total_params:,} parameters ({total_params/1e6:.2f}M)")
    print(f"Memory footprint:")
    print(f"- Model (fp32):        {total_params * 4 / 1024 / 1024:.1f} MB")
    print(f"- Model (fp16/bf16):   {total_params * 2 / 1024 / 1024:.1f} MB")
    print(
        f"- Optimizer (AdamW):   {total_params * 8 / 1024 / 1024:.1f} MB"
    )  # 2 states per param

    print("\nComparison to GPT-2 variants:")
    current_size = total_params / 1e6
    for name, size in gpt2_sizes.items():
        ratio = current_size / size
        print(f"- GPT-2 {name:6s} ({size:4d}M params): {ratio:.1%} the size")

    return total_params


# -----------------------------------------------------------------------------
import tiktoken
import numpy as np


def load_tokens(filename):
    npt = np.load(filename)
    npt = npt.astype(np.int32)  # added after video
    ptt = torch.tensor(npt, dtype=torch.long)
    return ptt


class DataLoaderLite:
    def __init__(self, B, T, process_rank, num_processes, split):
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.num_processes = num_processes
        assert split in {"train", "val"}

        # get the shard filenames
        data_root = DATASET_PATH
        shards = os.listdir(data_root)
        shards = [s for s in shards if split in s]
        shards = sorted(shards)
        shards = [os.path.join(data_root, s) for s in shards]
        self.shards = shards
        assert len(shards) > 0, f"no shards found for split {split}"
        if master_process:
            print(f"found {len(shards)} shards for split {split}")
        self.reset()

    def reset(self):
        # state, init at shard zero
        self.current_shard = 0
        self.tokens = load_tokens(self.shards[self.current_shard])
        self.current_position = self.B * self.T * self.process_rank

    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position : self.current_position + B * T + 1]
        x = (buf[:-1]).view(B, T)  # inputs
        y = (buf[1:]).view(B, T)  # targets
        # advance the position in the tensor
        self.current_position += B * T * self.num_processes
        # if loading the next batch would be out of bounds, advance to next shard
        if self.current_position + (B * T * self.num_processes + 1) > len(self.tokens):
            self.current_shard = (self.current_shard + 1) % len(self.shards)
            self.tokens = load_tokens(self.shards[self.current_shard])
            self.current_position = B * T * self.process_rank
        return x, y


# -----------------------------------------------------------------------------
# helper function for HellaSwag eval
# takes tokens, mask, and logits, returns the index of the completion with the lowest loss


def get_most_likely_row(tokens, mask, logits):
    # evaluate the autoregressive loss at all positions
    shift_logits = (logits[..., :-1, :]).contiguous()
    shift_tokens = (tokens[..., 1:]).contiguous()
    flat_shift_logits = shift_logits.view(-1, shift_logits.size(-1))
    flat_shift_tokens = shift_tokens.view(-1)
    shift_losses = F.cross_entropy(
        flat_shift_logits, flat_shift_tokens, reduction="none"
    )
    shift_losses = shift_losses.view(tokens.size(0), -1)
    # now get the average loss just for the completion region (where mask == 1), in each row
    shift_mask = (
        mask[..., 1:]
    ).contiguous()  # we must shift mask, so we start at the last prompt token
    masked_shift_losses = shift_losses * shift_mask
    # sum and divide by the number of 1s in the mask
    sum_loss = masked_shift_losses.sum(dim=1)
    avg_loss = sum_loss / shift_mask.sum(dim=1)
    # now we have a loss for each of the 4 completions
    # the one with the lowest loss should be the most likely
    pred_norm = avg_loss.argmin().item()
    return pred_norm


# Track gradient norms for adaptive clipping
def get_adaptive_clip_threshold():
    if len(grad_norm_history) < 10:
        return CLIP_THRESHOLD
    median_norm = np.median(list(grad_norm_history))
    return max(CLIP_THRESHOLD, median_norm * ADAPTIVE_CLIP_MULTIPLIER)


# -----------------------------------------------------------------------------
# simple launch:
# python train.py
# DDP launch for e.g. 8 GPUs:
# torchrun --standalone --nproc_per_node=8 train.py

# run the training loop
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

# set up DDP (distributed data parallel).
# torchrun command sets the env variables RANK, LOCAL_RANK, and WORLD_SIZE
ddp = int(os.environ.get("RANK", -1)) != -1  # is this a ddp run?
if ddp:
    # use of DDP atm demands CUDA, we set the device appropriately according to rank
    assert torch.cuda.is_available(), "note: need CUDA for DDP"
    init_process_group(backend="nccl")
    ddp_rank = int(os.environ["RANK"])
    ddp_local_rank = int(os.environ["LOCAL_RANK"])
    ddp_world_size = int(os.environ["WORLD_SIZE"])
    device = f"cuda:{ddp_local_rank}"
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0  # this process will do logging, checkpointing etc.
else:
    # vanilla, non-DDP run
    ddp_rank = 0
    ddp_local_rank = 0
    ddp_world_size = 1
    master_process = True
    # attempt to autodetect device
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    print(f"using device: {device}")
# added after video, pytorch can be serious about it's device vs. device_type distinction
device_type = "cuda" if device.startswith("cuda") else "cpu"

torch.manual_seed(1337)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337)

enc = tiktoken.get_encoding("gpt2")


if master_process:
    print("\nConfiguration from environment:")
    print(f"Model config:")
    print(f"- Block size: {GPTConfig.block_size}")
    print(f"- Vocab size: {GPTConfig.vocab_size}")
    print(f"- Layers: {GPTConfig.n_layer}")
    print(f"- Attention heads: {GPTConfig.n_head}")
    print(f"- Embedding dim: {GPTConfig.n_embd}")
    print(f"- Bias: {BIAS}")

    print("\nTraining config:")
    print(f"- Total batch size: {TOTAL_BATCH_SIZE}")
    print(f"- Micro batch size: {B_CONFIG}")
    print(f"- Sequence length: {T_CONFIG}")
    print(f"- Gradient accumulation steps: {TOTAL_BATCH_SIZE // (B_CONFIG * T_CONFIG)}")

    # Calculate total tokens that will be processed
    tokens_per_step = TOTAL_BATCH_SIZE
    total_tokens = tokens_per_step * MAX_STEPS
    print(f"\nTraining scale:")
    print(f"- Tokens per batch: {tokens_per_step:,}")
    print(f"- Total steps: {MAX_STEPS:,}")
    print(f"- Total tokens: {total_tokens:,} ({total_tokens/1e9:.2f}B)")
    print(f"- For comparison:")
    print(f"  • GPT-2:     40B tokens")
    print(f"  • GPT-3:    300B tokens")
    print(f"  • LLaMA:  1,000B tokens")

    print("\nOptimization:")
    print(f"- Max learning rate: {MAX_LR}")
    print(f"- Min learning rate: {MIN_LR}")
    print(f"- Warmup steps: {WARMUP_STEPS}")
    print(f"- Weight decay: {WEIGHT_DECAY}")
    print(f"- Device: {device}")

    print(f"\nDDP Configuration:")
    print(f"- DDP rank: {ddp_rank}")
    print(f"- DDP local rank: {ddp_local_rank}")
    print(f"- DDP world size: {ddp_world_size}")
    print("")

grad_accum_steps = TOTAL_BATCH_SIZE // (B_CONFIG * T_CONFIG * ddp_world_size)
if master_process:
    print(f"total desired batch size: {TOTAL_BATCH_SIZE}")
    print(f"=> calculated gradient accumulation steps: {grad_accum_steps}")

train_loader = DataLoaderLite(
    B=B_CONFIG,
    T=T_CONFIG,
    process_rank=ddp_rank,
    num_processes=ddp_world_size,
    split="train",
)
val_loader = DataLoaderLite(
    B=B_CONFIG,
    T=T_CONFIG,
    process_rank=ddp_rank,
    num_processes=ddp_world_size,
    split="val",
)

if device_type == "cuda":
    torch.set_float32_matmul_precision("high")

# create model
model = GPT(GPTConfig(vocab_size=PADDED_VOCAB_SIZE))
# model = GPT.from_pretrained("gpt2") # or init from OpenAI GPT-2
if master_process:
    calculate_parameter_count(model.config)
model.to(device)

# Initialize GradScaler for mixed precision training
scaler = GradScaler(device=device_type) if device_type == "cuda" else None

use_compile = (
    False  # torch.compile interferes with HellaSwag eval and Generation. TODO fix
)
if use_compile:
    model = torch.compile(model)
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])
raw_model = model.module if ddp else model  # always contains the "raw" unwrapped model


def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < WARMUP_STEPS:
        return MAX_LR * (it + 1) / WARMUP_STEPS
    # 2) if it > lr_decay_iters, return min learning rate
    if it > MAX_STEPS:
        return MIN_LR
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - WARMUP_STEPS) / (MAX_STEPS - WARMUP_STEPS)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (
        1.0 + math.cos(math.pi * decay_ratio)
    )  # coeff starts at 1 and goes to 0
    return MIN_LR + coeff * (MAX_LR - MIN_LR)


# Update optimizer configuration to use environment variables
optimizer = raw_model.configure_optimizers(
    weight_decay=WEIGHT_DECAY, learning_rate=MAX_LR, device_type=device_type
)

# create the log directory we will write checkpoints to and log to
log_dir = "log"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f"log.txt")
csv_file = os.path.join(log_dir, f"metrics.csv")

# Track best model
best_val_loss = float("inf")
best_step = None

# initialize CSV file with headers
if master_process:
    # write headers only if file doesn't exist or is empty
    write_headers = not os.path.exists(csv_file) or os.path.getsize(csv_file) == 0
    if write_headers:
        with open(csv_file, "w") as f:
            f.write("step,type,value\n")

    # clear the log.txt file
    with open(log_file, "w") as f:
        pass

# checkpoint loading
start_step = 0
if os.path.exists(os.path.join(log_dir, "latest.pt")) and master_process:
    print("Found existing checkpoint, attempting to resume training...")
    try:
        checkpoint = torch.load(
            os.path.join(log_dir, "latest.pt"), weights_only=False
        )  # need to load weights_only=False for our trusted checkpoint
        raw_model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_step = checkpoint["step"] + 1

        # restore RNG states
        if "rng_states" in checkpoint:
            torch.set_rng_state(checkpoint["rng_states"]["torch"])
            if (
                torch.cuda.is_available()
                and checkpoint["rng_states"]["cuda"] is not None
            ):
                torch.cuda.set_rng_state(checkpoint["rng_states"]["cuda"])
            np.random.set_state(checkpoint["rng_states"]["numpy"])

        print(f"Successfully resumed from step {start_step}")
    except Exception as e:
        print(f"Error loading checkpoint: {str(e)}")
        print("Starting training from scratch...")
        start_step = 0


# training loop to start from start_step (e.g., resume training)
for step in range(start_step, MAX_STEPS):
    last_step = step == MAX_STEPS - 1

    # Start measuring training time
    t0_step = time.time()

    # do one step of the optimization
    model.train()
    optimizer.zero_grad()
    loss_accum = 0.0
    for micro_step in range(grad_accum_steps):
        x, y = train_loader.next_batch()
        x, y = x.to(device), y.to(device)
        if ddp:
            model.require_backward_grad_sync = micro_step == grad_accum_steps - 1

        # Check gradients before backward pass
        with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
            logits, loss = model(x, y)
        loss = loss / grad_accum_steps
        loss_accum += loss.detach()

        # Use gradient scaling for mixed precision
        if scaler is not None:
            scaled_loss = scaler.scale(loss)
            scaled_loss.backward()
        else:
            loss.backward()

    if ddp:
        dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)

    # Check gradient norm before clipping
    unclipped_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float("inf"))
    if (
        torch.isnan(unclipped_norm)
        or torch.isinf(unclipped_norm)
        or unclipped_norm > CLIP_THRESHOLD * 10
    ):
        if master_process:
            print(
                f"WARNING: Gradient norm {unclipped_norm:.4f} too large. Skipping step."
            )
        optimizer.zero_grad()
        continue

    # Apply normal gradient clipping
    clip_threshold = get_adaptive_clip_threshold()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), clip_threshold)
    if not torch.isnan(norm) and not torch.isinf(norm):
        grad_norm_history.append(norm.item())

    # Update with gradient scaling
    lr = get_lr(step)
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

    if scaler is not None:
        scaler.step(optimizer)
        scaler.update()
    else:
        optimizer.step()

    if device_type == "cuda":
        torch.cuda.synchronize()

    # Calculate training step time
    t1_step = time.time()
    dt_step = t1_step - t0_step
    tokens_processed = (
        train_loader.B * train_loader.T * grad_accum_steps * ddp_world_size
    )
    tokens_per_sec = tokens_processed / dt_step

    # Add cooling break if we detect thermal throttling
    if (
        tokens_per_sec < 1000 and master_process
    ):  # Arbitrary threshold for detecting very slow steps
        print(
            f"WARNING: Detected very slow step ({tokens_per_sec:.2f} tokens/sec). Taking a cooling break..."
        )
        time.sleep(10)  # 10 second cooling break

    if master_process:
        print(
            f"step {step:5d} | loss: {loss_accum.item():.6f} | lr: {lr:.4e} | norm: {norm:.4f} | dt: {dt_step*1000:.2f}ms | tok/sec: {tokens_per_sec:.2f}"
        )
        # Append to log file
        with open(log_file, "a") as f:
            f.write(f"{step} train {loss_accum.item():.6f}\n")
        # Append to CSV file
        with open(csv_file, "a") as f:
            metrics = [
                ("train_loss", loss_accum.item()),
                ("learning_rate", lr),
                ("grad_norm", norm),
                ("step_time_ms", dt_step * 1000),
                ("tokens_per_sec", tokens_per_sec),
            ]
            for metric_type, value in metrics:
                f.write(f"{step},{metric_type},{value:.6f}\n")

    # Run validation and evaluation after timing the training step
    if step % SAVE_CHECKPOINT_INTERVAL == 0 or last_step:
        model.eval()
        val_loader.reset()
        with torch.no_grad():
            val_loss_accum = 0.0
            val_loss_steps = 20
            for _ in range(val_loss_steps):
                x, y = val_loader.next_batch()
                x, y = x.to(device), y.to(device)
                with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                    logits, loss = model(x, y)
                loss = loss / val_loss_steps
                val_loss_accum += loss.detach()
        if ddp:
            dist.all_reduce(val_loss_accum, op=dist.ReduceOp.AVG)
        if master_process:
            print(f"validation loss: {val_loss_accum.item():.4f}")
            with open(log_file, "a") as f:
                f.write(f"{step} val {val_loss_accum.item():.4f}\n")
            with open(csv_file, "a") as f:
                f.write(f"{step},val_loss,{val_loss_accum.item():.6f}\n")

            # Track best model
            if val_loss_accum.item() < best_val_loss:
                best_val_loss = val_loss_accum.item()
                best_step = step
                # Save best model checkpoint
                best_checkpoint_path = os.path.join(log_dir, "best_model.pt")
                checkpoint = {
                    "model": raw_model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "config": raw_model.config,
                    "step": step,
                    "val_loss": val_loss_accum.item(),
                    "rng_states": {
                        "torch": torch.get_rng_state(),
                        "cuda": (
                            torch.cuda.get_rng_state()
                            if torch.cuda.is_available()
                            else None
                        ),
                        "numpy": np.random.get_state(),
                    },
                    "training_args": {
                        "total_batch_size": TOTAL_BATCH_SIZE,
                        "batch_size": B_CONFIG,
                        "seq_length": T_CONFIG,
                        "max_lr": MAX_LR,
                        "min_lr": MIN_LR,
                        "warmup_steps": WARMUP_STEPS,
                        "max_steps": MAX_STEPS,
                        "weight_decay": WEIGHT_DECAY,
                    },
                }
                torch.save(checkpoint, best_checkpoint_path)
                print(
                    f"New best model at step {step} with validation loss {best_val_loss:.4f}"
                )

            # Regular checkpoint saving
            if step > 0 and (step % SAVE_CHECKPOINT_INTERVAL == 0 or last_step):
                # optionally write model checkpoints
                checkpoint_path = os.path.join(log_dir, f"model_{step:05d}.pt")
                checkpoint = {
                    "model": raw_model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "config": raw_model.config,
                    "step": step,
                    "val_loss": val_loss_accum.item(),
                    # Save RNG states
                    "rng_states": {
                        "torch": torch.get_rng_state(),
                        "cuda": (
                            torch.cuda.get_rng_state()
                            if torch.cuda.is_available()
                            else None
                        ),
                        "numpy": np.random.get_state(),
                    },
                    # Save training hyperparameters
                    "training_args": {
                        "total_batch_size": TOTAL_BATCH_SIZE,
                        "batch_size": B_CONFIG,
                        "seq_length": T_CONFIG,
                        "max_lr": MAX_LR,
                        "min_lr": MIN_LR,
                        "warmup_steps": WARMUP_STEPS,
                        "max_steps": MAX_STEPS,
                        "weight_decay": WEIGHT_DECAY,
                    },
                }
                torch.save(checkpoint, checkpoint_path)

                # save a special "latest.pt" checkpoint for easy resumption
                latest_path = os.path.join(log_dir, "latest.pt")
                torch.save(checkpoint, latest_path)

    # once in a while evaluate hellaswag
    if (step > 0 and step % RUN_EVAL_INTERVAL == 0 or last_step) and (not use_compile):
        # if (step % SAVE_CHECKPOINT_INTERVAL == 0 or last_step) and (not use_compile):
        print(f"Evaluating HellaSwag at step {step}...")
        num_correct_norm = 0
        num_total = 0
        for i, example in enumerate(iterate_examples("val")):
            # only process examples where i % ddp_world_size == ddp_rank
            if i % ddp_world_size != ddp_rank:
                continue
            # render the example into tokens and labels
            _, tokens, mask, label = render_example(example)
            tokens = tokens.to(device)
            mask = mask.to(device)
            # get the logits
            with torch.no_grad():
                with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                    logits, loss = model(tokens)
                pred_norm = get_most_likely_row(tokens, mask, logits)
            num_total += 1
            num_correct_norm += int(pred_norm == label)
        # reduce the stats across all processes
        if ddp:
            num_total = torch.tensor(num_total, dtype=torch.long, device=device)
            num_correct_norm = torch.tensor(
                num_correct_norm, dtype=torch.long, device=device
            )
            dist.all_reduce(num_total, op=dist.ReduceOp.SUM)
            dist.all_reduce(num_correct_norm, op=dist.ReduceOp.SUM)
            num_total = num_total.item()
            num_correct_norm = num_correct_norm.item()
        acc_norm = num_correct_norm / num_total
        if master_process:
            print(f"HellaSwag accuracy: {num_correct_norm}/{num_total}={acc_norm:.4f}")
            with open(log_file, "a") as f:
                f.write(f"{step} hella {acc_norm:.4f}\n")
            with open(csv_file, "a") as f:
                f.write(f"{step},hellaswag_acc,{acc_norm:.6f}\n")

    # once in a while generate from the model (except step 0, which is noise)
    if ((step > 0 and step % RUN_EVAL_INTERVAL == 0) or last_step) and (
        not use_compile
    ):
        model.eval()
        num_return_sequences = 4
        max_length = 32
        tokens = enc.encode("Hello, I'm a language model,")
        tokens = torch.tensor(tokens, dtype=torch.long)
        tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)
        xgen = tokens.to(device)
        sample_rng = torch.Generator(device=device)
        sample_rng.manual_seed(42 + ddp_rank)
        while xgen.size(1) < max_length:
            # forward the model to get the logits
            with torch.no_grad():
                with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                    logits, loss = model(xgen)  # (B, T, vocab_size)
                # take the logits at the last position
                logits = logits[:, -1, :]  # (B, vocab_size)
                # get the probabilities
                probs = F.softmax(logits, dim=-1)
                # do top-k sampling of 50 (huggingface pipeline default)
                # topk_probs here becomes (5, 50), topk_indices is (5, 50)
                topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
                # select a token from the top-k probabilities
                # note: multinomial does not demand the input to sum to 1
                ix = torch.multinomial(topk_probs, 1, generator=sample_rng)  # (B, 1)
                # gather the corresponding indices
                xcol = torch.gather(topk_indices, -1, ix)  # (B, 1)
                # append to the sequence
                xgen = torch.cat((xgen, xcol), dim=1)
        # print the generated text
        for i in range(num_return_sequences):
            tokens = xgen[i, :max_length].tolist()
            decoded = enc.decode(tokens)
            print(f"rank {ddp_rank} sample {i}: {decoded}")

if ddp:
    destroy_process_group()
