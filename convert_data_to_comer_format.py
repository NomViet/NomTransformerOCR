"""
Convert dataset to CoMER format.
"""
import os
import random
from PIL import Image

from config import CoMERFormat as cfg
from utils.file_utils import convert_label_to_tokens, read_transcript_file
from utils.image_utils import process_image_to_grayscale

os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
random.seed(cfg.RANDOM_SEED)

# Output directories
TRAIN_DIR = f'{cfg.OUTPUT_DIR}/data/train'
VAL_DIR = f'{cfg.OUTPUT_DIR}/data/val'
VAL_300_DIR = f'{cfg.OUTPUT_DIR}/data/val_300'
TEST_DIR = f'{cfg.OUTPUT_DIR}/data/test'
GENERATED_DIR = f'{cfg.OUTPUT_DIR}/data/train_generated'

TRAIN_IMG_DIR = f'{TRAIN_DIR}/img'
VAL_IMG_DIR = f'{VAL_DIR}/img'
VAL_300_IMG_DIR = f'{VAL_300_DIR}/img'
TEST_IMG_DIR = f'{TEST_DIR}/img'
GENERATED_IMG_DIR = f'{GENERATED_DIR}/img'

TRAIN_CAPTION = f'{TRAIN_DIR}/caption.txt'
VAL_CAPTION = f'{VAL_DIR}/caption.txt'
VAL_300_CAPTION = f'{VAL_300_DIR}/caption.txt'
TEST_CAPTION = f'{TEST_DIR}/caption.txt'
GENERATED_CAPTION = f'{GENERATED_DIR}/caption.txt'


def create_directories():
    dirs = [TRAIN_IMG_DIR, VAL_IMG_DIR, VAL_300_IMG_DIR, TEST_IMG_DIR, GENERATED_IMG_DIR]
    for d in dirs:
        os.makedirs(d, exist_ok=True)
    print(f"Created directories in {cfg.OUTPUT_DIR}")


def process_split(lines, img_dir, caption_path, prefix):
    captions = []
    success, errors = 0, 0
    
    for i, line in enumerate(lines):
        try:
            file_path, label = line.split("\t")
            full_path = os.path.join(cfg.DATASET_DIR, file_path)
            img_name = f"{prefix}{i:06d}"
            
            if not os.path.exists(full_path):
                errors += 1
                continue
            
            output_path = f'{img_dir}/{img_name}.bmp'
            if process_image_to_grayscale(full_path, output_path, cfg.ROTATE):
                tokens = convert_label_to_tokens(label)
                captions.append(f"{img_name} {tokens}\n")
                success += 1
            else:
                errors += 1
        except Exception:
            errors += 1
    
    with open(caption_path, 'w', encoding='utf-8') as f:
        f.writelines(captions)
    
    return success, errors


def create_comer_dataset():
    """Create CoMER dataset from original data."""
    print("\n" + "="*60)
    print("CONVERTING TO CoMER FORMAT")
    print("="*60)
    
    create_directories()
    
    # Load data
    print(f"\nReading: {cfg.TRAIN_TRANSCRIPTS_PATH}")
    all_lines = read_transcript_file(cfg.TRAIN_TRANSCRIPTS_PATH)
    total = len(all_lines)
    print(f"Total lines: {total}")
    
    # Limit samples
    num_samples = min(cfg.NUM_TRAIN_SAMPLES, total)
    all_lines = all_lines[:num_samples]
    
    # Split data
    random.shuffle(all_lines)
    split_idx = int(len(all_lines) * cfg.VAL_RATIO)
    train_lines = all_lines[:split_idx]
    val_lines = all_lines[split_idx:]
    val_300_lines = val_lines[:cfg.MAX_VAL_300]
    
    # Load test data
    print(f"Reading: {cfg.VAL_TRANSCRIPTS_PATH}")
    test_lines = read_transcript_file(cfg.VAL_TRANSCRIPTS_PATH)
    
    print(f"\nSplit: train={len(train_lines)}, val={len(val_lines)}, "
          f"val_300={len(val_300_lines)}, test={len(test_lines)}")
    
    # Process splits
    splits = [
        (train_lines, "train", TRAIN_IMG_DIR, TRAIN_CAPTION, "img"),
        (val_300_lines, "val_300", VAL_300_IMG_DIR, VAL_300_CAPTION, "val"),
        (val_lines, "val", VAL_IMG_DIR, VAL_CAPTION, "val"),
        (test_lines, "test", TEST_IMG_DIR, TEST_CAPTION, "val"),
    ]
    
    results = {}
    for lines, name, img_dir, caption_path, prefix in splits:
        print(f"\nProcessing {name}...")
        success, errors = process_split(lines, img_dir, caption_path, prefix)
        results[name] = (success, errors, len(lines))
        print(f"  Success: {success}/{len(lines)}, Errors: {errors}")
    
    print("\n" + "="*60)
    print("CONVERSION COMPLETE")
    print("="*60)


def create_generated_dataset():
    """Process synthetic data into train_generated folder."""
    print("\n" + "="*60)
    print("PROCESSING GENERATED DATA")
    print(f"Source: {cfg.GENERATED_SOURCE_DIR}")
    print("="*60)
    
    source_img = os.path.join(cfg.GENERATED_SOURCE_DIR, "img")
    source_caption = os.path.join(cfg.GENERATED_SOURCE_DIR, "caption.txt")
    
    if not os.path.exists(source_img):
        print(f"Source img directory not found: {source_img}")
        return
    
    if not os.path.exists(source_caption):
        print(f"Source caption not found: {source_caption}")
        return
    
    lines = read_transcript_file(source_caption)
    if not lines:
        print("Empty caption file.")
        return
    
    print(f"Found {len(lines)} samples")
    
    success, errors = 0, 0
    
    with open(GENERATED_CAPTION, "w", encoding="utf-8") as f_out:
        for i, line in enumerate(lines):
            try:
                img_name_full, text = line.split("\t")
                img_name = img_name_full.split("/")[-1]
                src_path = os.path.join(source_img, img_name)
                
                new_name = f"gen_img{i:06d}"
                dst_path = os.path.join(GENERATED_IMG_DIR, f"{new_name}.bmp")
                
                if not os.path.exists(src_path):
                    errors += 1
                    continue
                
                img = Image.open(src_path)
                if img.mode != 'L':
                    img = img.convert('L')
                img.save(dst_path, "BMP")
                
                tokens = convert_label_to_tokens(text)
                f_out.write(f"{new_name} {tokens}\n")
                success += 1
            except Exception:
                errors += 1
    
    print(f"\nSuccess: {success}/{len(lines)}")
    print(f"Errors: {errors}/{len(lines)}")
    print(f"Output: {GENERATED_IMG_DIR}")
    print("="*60)


def main():
    create_comer_dataset()
    create_generated_dataset()
    print("\nDONE!")


if __name__ == "__main__":
    main()
