import pandas as pd
import matplotlib.pyplot as plt

# Set style using a built-in style
plt.style.use("bmh")

# Read the CSV file
df = pd.read_csv("log/metrics.csv")

# Create a figure with subplots
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle("Training Metrics", fontsize=16)

# Plot 1: Losses
losses = df[df["type"].isin(["train_loss", "val_loss"])]
for loss_type in ["train_loss", "val_loss"]:
    data = losses[losses["type"] == loss_type]
    ax1.plot(data["step"], data["value"], label=loss_type)
ax1.set_title("Loss over time")
ax1.set_ylabel("Loss")
ax1.legend()

# Plot 2: HellaSwag Accuracy
accuracy = df[df["type"] == "hellaswag_acc"]
ax2.plot(accuracy["step"], accuracy["value"])
ax2.set_title("HellaSwag Accuracy")
ax2.set_ylabel("Accuracy")

# Plot 3: Learning Rate
lr = df[df["type"] == "learning_rate"]
ax3.plot(lr["step"], lr["value"])
ax3.set_title("Learning Rate Schedule")
ax3.set_ylabel("Learning Rate")

# Plot 4: Training Speed
tokens = df[df["type"] == "tokens_per_sec"]
ax4.plot(tokens["step"], tokens["value"])
ax4.set_title("Training Speed")
ax4.set_ylabel("Tokens per second")

# Adjust layout and save
plt.tight_layout()
plt.savefig("log/training_metrics.png", dpi=300, bbox_inches="tight")
plt.close()

# Print some statistics
print("\nTraining Statistics:")
print("-" * 50)

# Calculate statistics for the last 10% of training steps
last_10_percent = df["step"].max() * 0.9
recent_df = df[df["step"] > last_10_percent]

metrics = {
    "train_loss": "Training Loss",
    "val_loss": "Validation Loss",
    "hellaswag_acc": "HellaSwag Accuracy",
    "tokens_per_sec": "Training Speed",
}

for metric, name in metrics.items():
    data = recent_df[recent_df["type"] == metric]
    if not data.empty:
        mean_val = data["value"].mean()
        std_val = data["value"].std()
        print(f"{name:20s}: {mean_val:.4f} ± {std_val:.4f}")
