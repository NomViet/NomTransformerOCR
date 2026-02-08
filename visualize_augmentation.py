import os
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

# Style
plt.style.use("default")
plt.rcParams["figure.facecolor"] = "white"
plt.rcParams["axes.facecolor"] = "#f0f0f0"
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.alpha"] = 0.3


def load_vocab(vocab_path):
    """Load vocabulary from file (one char per line)."""
    with open(vocab_path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def load_labels(label_path, use_tab=False):
    """Load caption file and count character frequency."""
    char_counter = Counter()
    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if use_tab:
                parts = line.split("\t")
                if len(parts) == 2:
                    char_counter.update(list(parts[1]))
            else:
                parts = line.split()
                if len(parts) > 1:
                    char_counter.update(parts[1:])
    return char_counter


# =============== CONFIG ===============
VOCAB_FILE = "dictionary_7k5_new.txt"
TRAIN_FILE = "data/train/caption.txt"
AUG_FILE = "synthesize_train_version_64/caption.txt"
OUTPUT_DIR = "charts"
# ======================================


def main():
    print("Loading data...")
    vocab = load_vocab(VOCAB_FILE)
    train_counter = load_labels(TRAIN_FILE)
    aug_counter = load_labels(AUG_FILE, use_tab=True)

    combined_counter = train_counter + aug_counter
    vocab_set = set(vocab)

    train_freq = np.array([train_counter.get(c, 0) for c in vocab])
    combined_freq = np.array([combined_counter.get(c, 0) for c in vocab])

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ---------- Chart 1: Frequency Distribution ----------
    fig1 = plt.figure(figsize=(8, 6))
    ax1 = fig1.add_subplot(111)
    bins = np.logspace(0, np.log10(max(combined_freq) + 1), 50)
    ax1.hist(train_freq[train_freq > 0], bins=bins, alpha=0.6, label="Before Aug", color="blue")
    ax1.hist(combined_freq[combined_freq > 0], bins=bins, alpha=0.6, label="After Aug", color="red")
    ax1.set_xscale("log")
    ax1.set_xlabel("Character Frequency (log scale)", fontsize=12)
    ax1.set_ylabel("Number of Characters", fontsize=12)
    ax1.set_title("Character Frequency Distribution", fontsize=14, fontweight="bold")
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    plt.tight_layout()
    path1 = os.path.join(OUTPUT_DIR, "frequency_distribution.png")
    plt.savefig(path1, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path1}")

    # ---------- Chart 5: Distribution by Frequency Range ----------
    freq_ranges = [
        (0, 0),
        (1, 5),
        (6, 10),
        (11, 50),
        (51, 100),
        (100, 1000),
        (1000, float("inf")),
    ]
    range_labels = ["0", "1-5", "6-10", "11-50", "51-100", "100-1k", ">1k"]
    before_counts = []
    after_counts = []

    for low, high in freq_ranges:
        if low == 0:
            before_counts.append(int(np.sum(train_freq == 0)))
            after_counts.append(int(np.sum(combined_freq == 0)))
        else:
            before_counts.append(
                int(np.sum((train_freq >= low) & (train_freq <= high)))
            )
            after_counts.append(
                int(np.sum((combined_freq >= low) & (combined_freq <= high)))
            )

    fig5 = plt.figure(figsize=(12, 6))
    ax5 = fig5.add_subplot(111)
    x = np.arange(len(range_labels))
    width = 0.35
    bars1 = ax5.bar(
        x - width / 2, before_counts, width, label="Baseline (Re-split)", alpha=0.8, color="#4DB6AC"
    )
    bars2 = ax5.bar(
        x + width / 2, after_counts, width, label="With Synthetic Data", alpha=0.8, color="#F9A825"
    )
    ax5.set_xlabel("Frequency Range", fontsize=12)
    ax5.set_ylabel("Number of Characters", fontsize=12)
    ax5.set_title(
        "Character Frequency Distribution Before and After Data Augmentation",
        fontsize=14,
        fontweight="bold",
    )
    ax5.set_xticks(x)
    ax5.set_xticklabels(range_labels, fontsize=11)
    ax5.legend(fontsize=11)
    ax5.grid(True, alpha=0.3, axis="y")

    for bars in [bars1, bars2]:
        for bar in bars:
            h = bar.get_height()
            if h > 0:
                ax5.text(
                    bar.get_x() + bar.get_width() / 2,
                    h,
                    f"{int(h)}",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                )

    plt.tight_layout()
    path5 = os.path.join(OUTPUT_DIR, "distribution_by_range.png")
    plt.savefig(path5, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path5}")

    print(f"\nDone. Outputs in '{OUTPUT_DIR}/'")


if __name__ == "__main__":
    main()
