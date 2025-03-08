import os
import torch
import torch.nn as nn
from torch.nn import functional as F
import argparse
from tqdm import tqdm
from hellaswag import render_example, iterate_examples
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()

BLOCK_SIZE: int = int(os.environ.get("BLOCK_SIZE", "1024"))  # max sequence length
VOCAB_SIZE: int = int(os.environ.get("VOCAB_SIZE", "50304"))  # GPT-2 vocab size
N_LAYER: int = int(os.environ.get("N_LAYER", "12"))  # number of layers
N_HEAD: int = int(os.environ.get("N_HEAD", "12"))  # number of heads
N_EMBD: int = int(os.environ.get("N_EMBD", "768"))  # embedding dimension


# Model architecture (same as in generate.py)
@dataclass
class GPTConfig:
    block_size: int = BLOCK_SIZE
    vocab_size: int = VOCAB_SIZE
    n_layer: int = N_LAYER
    n_head: int = N_HEAD
    n_embd: int = N_EMBD


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


def evaluate_checkpoint(
    checkpoint_path, device="cuda" if torch.cuda.is_available() else "cpu"
):
    """Evaluate a specific checkpoint on HellaSwag validation set."""
    print(f"Loading checkpoint from {checkpoint_path}")

    # If we're given a directory, use the latest checkpoint
    if os.path.isdir(checkpoint_path):
        checkpoint_files = [
            f for f in os.listdir(checkpoint_path) if f.startswith("model_")
        ]
        if not checkpoint_files:
            raise FileNotFoundError(
                f"No checkpoint files found in directory: {checkpoint_path}"
            )
        checkpoint_path = os.path.join(checkpoint_path, sorted(checkpoint_files)[-1])
        print(f"Using latest checkpoint: {checkpoint_path}")

    # Load checkpoint with weights_only=False (for trusted checkpoints)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Create and load model
    print("Initializing model...")
    model = GPT(GPTConfig())
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    model.eval()
    print("Model loaded successfully!")

    # Track metrics
    num_correct_norm = 0
    num_total = 0
    results = []  # Store detailed results

    # Evaluate all examples
    print("Evaluating on HellaSwag validation set...")
    for example in tqdm(iterate_examples("val")):
        # Render the example into tokens and labels
        ctx, tokens, mask, label = render_example(example)
        tokens = tokens.to(device)
        mask = mask.to(device)

        # Get model predictions
        with torch.no_grad():
            with torch.autocast(
                device_type="cuda" if device == "cuda" else "cpu", dtype=torch.bfloat16
            ):
                logits = model(tokens)
                pred_norm = get_most_likely_row(tokens, mask, logits)

        # Update metrics
        num_total += 1
        is_correct = int(pred_norm == label)
        num_correct_norm += is_correct

        # Store detailed result
        results.append(
            {
                "id": example["ind"],
                "type": example["split_type"],
                "correct": is_correct,
                "prediction": pred_norm,
                "label": label,
                "context": ctx,
                "endings": example["endings"],
            }
        )

    # Calculate overall accuracy
    accuracy = num_correct_norm / num_total

    # Calculate accuracy by split type
    split_metrics = {}
    for split in ["zeroshot", "indomain"]:
        split_results = [r for r in results if r["type"] == split]
        if split_results:
            split_acc = sum(r["correct"] for r in split_results) / len(split_results)
            split_metrics[split] = split_acc

    return {
        "accuracy": accuracy,
        "total_examples": num_total,
        "correct_examples": num_correct_norm,
        "split_metrics": split_metrics,
        "detailed_results": results,
    }


def get_most_likely_row(tokens, mask, logits):
    """Calculate the most likely completion based on per-token losses."""
    shift_logits = logits[..., :-1, :].contiguous()
    shift_tokens = tokens[..., 1:].contiguous()
    flat_shift_logits = shift_logits.view(-1, shift_logits.size(-1))
    flat_shift_tokens = shift_tokens.view(-1)
    shift_losses = F.cross_entropy(
        flat_shift_logits, flat_shift_tokens, reduction="none"
    )
    shift_losses = shift_losses.view(tokens.size(0), -1)

    shift_mask = mask[..., 1:].contiguous()
    masked_shift_losses = shift_losses * shift_mask
    sum_loss = masked_shift_losses.sum(dim=1)
    avg_loss = sum_loss / shift_mask.sum(dim=1)

    return avg_loss.argmin().item()


def print_results(results, num_examples=5):
    """Print detailed analysis of evaluation results."""
    print("\nOverall Results:")
    print(
        f"Total Accuracy: {results['accuracy']:.4f} ({results['correct_examples']}/{results['total_examples']})"
    )

    print("\nAccuracy by Split Type:")
    for split, acc in results["split_metrics"].items():
        print(f"{split}: {acc:.4f}")

    print(f"\nDetailed Analysis of {num_examples} Random Examples:")
    import random

    sample_results = random.sample(results["detailed_results"], num_examples)

    for r in sample_results:
        print("\n" + "=" * 80)
        print(f"Example {r['id']} ({r['type']}):")
        print(f"Context: {r['context']}")
        print("\nPossible endings:")
        for i, ending in enumerate(r["endings"]):
            prefix = "✓" if i == r["label"] else " "
            pred = ">" if i == r["prediction"] else " "
            print(f"{prefix}{pred} {i}: {ending}")
        print(f"Correct: {'Yes' if r['correct'] else 'No'}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate GPT model on HellaSwag")
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to model checkpoint"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run evaluation on",
    )
    parser.add_argument(
        "--examples", type=int, default=5, help="Number of detailed examples to show"
    )
    args = parser.parse_args()

    # Run evaluation
    results = evaluate_checkpoint(args.checkpoint, args.device)

    # Print results
    print_results(results, args.examples)

    # Save detailed results
    output_dir = os.path.dirname(args.checkpoint)
    output_file = os.path.join(output_dir, "eval_results.pt")
    torch.save(results, output_file)
    print(f"\nDetailed results saved to {output_file}")


if __name__ == "__main__":
    main()
