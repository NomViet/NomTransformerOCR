"""
Build vocabulary dictionary from caption files.
Output: comer/datamodule/dictionary.txt
"""
import os

from config import CoMERFormat as cfg

# Input: CoMER data directory
DATA_ROOT = f"{cfg.OUTPUT_DIR}/data"

# Output: dictionary file for CoMER vocab
DICT_OUTPUT = "comer/datamodule/dictionary.txt"

# Splits to scan
SPLITS = ["train", "train_generated", "val", "val_300", "test"]


def main():
    vocab = set()
    
    for split in SPLITS:
        caption_path = os.path.join(DATA_ROOT, split, "caption.txt")
        
        if not os.path.exists(caption_path):
            print(f"Skip (not found): {caption_path}")
            continue
        
        print(f"Reading: {caption_path}")
        
        with open(caption_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) > 1:
                    # Format: img000001 char1 char2 char3 ...
                    chars = parts[1:]
                    vocab.update(chars)
    
    print(f"\nTotal vocab: {len(vocab)} characters")
    
    # Write dictionary file
    os.makedirs(os.path.dirname(DICT_OUTPUT), exist_ok=True)
    
    with open(DICT_OUTPUT, "w", encoding="utf-8") as f:
        for ch in sorted(vocab):
            f.write(ch + "\n")
    
    print(f"Saved: {DICT_OUTPUT}")


if __name__ == "__main__":
    main()
