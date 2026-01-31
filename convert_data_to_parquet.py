"""
Convert CoMER dataset to Parquet format for HuggingFace.
"""
import math
from pathlib import Path
from datasets import Dataset, Features, Value, Image

from config import ParquetConversion as cfg

DATA_ROOT = Path(cfg.DATA_ROOT)
OUTPUT = Path(cfg.OUTPUT_DIR)
OUTPUT.mkdir(exist_ok=True)


def process_split(split):
    """Process a single data split."""
    root = DATA_ROOT / split
    cap_file = root / "caption.txt"
    img_dir = root / "img"
    
    print(f"\nProcessing {split}...")
    
    if not cap_file.exists():
        print(f"  Caption file not found: {cap_file}")
        return
    
    with open(cap_file, "r") as f:
        lines = f.readlines()
    
    total = len(lines)
    if total == 0:
        print(f"  No data found for {split}. Skipping.")
        return
    
    num_shards = math.ceil(total / cfg.SHARD_SIZE)
    print(f"  Items: {total} | Shards: {num_shards}")
    
    shard_id = 0
    start = 0
    
    while start < total:
        end = min(start + cfg.SHARD_SIZE, total)
        samples = []
        
        print(f"  Shard {shard_id+1}/{num_shards} (items {start}-{end})...")
        
        for line in lines[start:end]:
            try:
                parts = line.strip().split()
                img_name = parts[0] + ".bmp"
                text = "".join(parts[1:])
                img_path = img_dir / img_name
                
                if not img_path.exists():
                    continue
                
                with open(img_path, "rb") as img_f:
                    samples.append({"image": img_f.read(), "text": text})
            except Exception:
                continue
        
        if not samples:
            print("    No valid samples in shard.")
            shard_id += 1
            start = end
            continue
        
        ds = Dataset.from_list(
            samples,
            features=Features({"image": Image(), "text": Value("string")})
        )
        
        shard_path = OUTPUT / f"{split}-{shard_id:05d}-of-{num_shards:05d}.parquet"
        ds.to_parquet(str(shard_path))
        
        print(f"    Wrote: {shard_path.name} ({len(samples)} items)")
        
        shard_id += 1
        start = end


def main():
    print("="*60)
    print("CONVERTING TO PARQUET FORMAT")
    print("="*60)
    
    for split in cfg.SPLITS:
        process_split(split)
    
    print("\nDONE - Images embedded into parquet!")


if __name__ == "__main__":
    main()
