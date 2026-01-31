"""
File I/O utilities for data processing.
"""
import os
from collections import Counter
from threading import Lock

print_lock = Lock()


def safe_print(*args, **kwargs):
    """Thread-safe print function."""
    with print_lock:
        print(*args, **kwargs)


def load_vocab_from_file(file_path):
    """Load unique characters from transcript file."""
    vocab = set()
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if parts:
                    label = parts[-1]
                    vocab.update(set(label))
        return vocab
    except Exception as e:
        safe_print(f"Error reading {file_path}: {e}")
        return set()


def count_chars_in_file(file_path):
    """Count character occurrences in transcript file."""
    counter = Counter()
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if parts:
                    counter.update(parts[-1])
        return counter
    except Exception as e:
        safe_print(f"Error reading {file_path}: {e}")
        return Counter()


def load_statistics(file_path):
    """Load statistics from tab-separated file."""
    print(f"Loading statistics: {file_path}")
    counter = Counter()
    total = 0
    
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) != 2:
                    continue
                total += 1
                counter.update(set(parts[1]))
        
        print(f"Loaded: {total} images, {len(counter)} characters")
        return counter
    except Exception as e:
        print(f"Error: {e}")
        return Counter()


def build_char_catalog(char_dir):
    """Build character to folder path mapping."""
    print(f"Building catalog: {char_dir}")
    catalog = {}
    
    for folder_name in os.listdir(char_dir):
        folder_path = os.path.join(char_dir, folder_name)
        if os.path.isdir(folder_path):
            parts = folder_name.split("_", 1)
            char = parts[1] if len(parts) == 2 and parts[1] else "?"
            catalog[char] = folder_path
    
    print(f"Catalog: {len(catalog)} characters")
    return catalog


def convert_label_to_tokens(label):
    """Convert label string to space-separated tokens."""
    return ' '.join(list(label))


def read_transcript_file(file_path):
    """Read transcript file and return lines."""
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read().strip().split("\n")


def write_caption_file(path, lines):
    """Write caption file."""
    with open(path, 'w', encoding='utf-8') as f:
        f.writelines(lines)
