"""
Generate synthetic Nom character images for OCR training.
"""
import os
import random
import json
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
from collections import Counter
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

from config import DataGeneration as cfg
from utils.file_utils import (
    safe_print, load_vocab_from_file, count_chars_in_file,
    load_statistics, build_char_catalog
)
from utils.image_utils import (
    get_random_variant, augment_char_image, add_noise,
    blend_with_paper, random_paper_texture, apply_random_crop
)

IMG_DIR = os.path.join(cfg.OUTPUT_DIR, cfg.IMG_SUBDIR)
LABEL_DIR = os.path.join(cfg.OUTPUT_DIR, cfg.LABEL_SUBDIR)

os.makedirs(IMG_DIR, exist_ok=True)
os.makedirs(LABEL_DIR, exist_ok=True)

metadata_lock = Lock()
counter_lock = Lock()


# =============================================================================
# TIER CALCULATION
# =============================================================================

def get_avg_page_dimensions(stats_file, sample_percent=0.05):
    """Calculate average page dimensions from sample images."""
    print("Calculating average dimensions...")
    base_dir = os.path.dirname(os.path.abspath(stats_file))
    
    image_paths = []
    with open(stats_file, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) == 2:
                image_paths.append(parts[0])
    
    if not image_paths:
        return 0, 0
    
    num_samples = max(1, int(len(image_paths) * sample_percent))
    sampled = random.sample(image_paths, min(num_samples, len(image_paths)))
    
    widths, heights = [], []
    for rel_path in sampled:
        try:
            with Image.open(os.path.join(base_dir, rel_path)) as img:
                widths.append(img.width)
                heights.append(img.height)
        except Exception as e:
            print(f"Error processing image {rel_path}")
            continue
    
    if not widths:
        return 0, 0
    
    avg_w, avg_h = int(np.mean(widths)), int(np.mean(heights))
    print(f"Average size: {avg_w}x{avg_h}px")
    return avg_w, avg_h


def calculate_tiers(char_catalog, train_vocab, val_vocab, train_counts):
    """
    Classify characters into tiers based on frequency.
    
    Tier 0: Val-only (missing in train)
    Tier 1: Low frequency (count <= TIER_1_THRESHOLD)
    Tier 2: Medium frequency
    Tier 3: High frequency
    """
    print("\n" + "="*70)
    print("TIER CLASSIFICATION")
    print("="*70)
    
    tier_0, tier_1, tier_2, tier_3 = [], [], [], []
    missing_in_train = val_vocab - train_vocab
    
    for char in char_catalog.keys():
        if char in missing_in_train:
            tier_0.append(char)
        elif char in train_vocab:
            count = train_counts.get(char, 0)
            if count <= cfg.TIER_1_THRESHOLD:
                tier_1.append(char)
            elif count <= cfg.TIER_2_THRESHOLD:
                tier_2.append(char)
            else:
                tier_3.append(char)
        else:
            tier_3.append(char)
    
    print(f"\nTier 0 (Val-only): {len(tier_0)} chars")
    print(f"Tier 1 (count <= {cfg.TIER_1_THRESHOLD}): {len(tier_1)} chars")
    print(f"Tier 2 ({cfg.TIER_1_THRESHOLD} < count <= {cfg.TIER_2_THRESHOLD}): {len(tier_2)} chars")
    print(f"Tier 3 (count > {cfg.TIER_2_THRESHOLD}): {len(tier_3)} chars")
    
    # Build pages list
    pages = []
    for char in tier_0:
        pages.extend([(char, 0)] * cfg.PAGES_PER_TIER_0)
    for char in tier_1:
        pages.extend([(char, 1)] * cfg.PAGES_PER_TIER_1)
    for char in tier_2:
        pages.extend([(char, 2)] * cfg.PAGES_PER_TIER_2)
    for char in tier_3:
        pages.extend([(char, 3)] * cfg.PAGES_PER_TIER_3)
    
    random.shuffle(pages)
    
    print(f"\nTotal pages to generate: {len(pages)}")
    print("="*70)
    
    return tier_0, tier_1, tier_2, tier_3, pages


# =============================================================================
# PAGE GENERATION
# =============================================================================

def select_page_chars(main_char, main_tier, tier_0, tier_1, tier_2, tier_3):
    """Select characters for a single page."""
    num_t0 = random.randint(*cfg.TIER_0_RANGE)
    num_t1 = random.randint(*cfg.TIER_1_RANGE)
    num_t2 = random.randint(*cfg.TIER_2_RANGE)
    num_t3 = random.randint(*cfg.TIER_3_RANGE)
    
    # Ensure main_char is included
    if main_tier == 0:
        num_t0 = max(num_t0, 1)
    elif main_tier == 1:
        num_t1 = max(num_t1, 1)
    elif main_tier == 2:
        num_t2 = max(num_t2, 1)
    
    total = num_t0 + num_t1 + num_t2 + num_t3
    
    # Trim from high tiers if over limit
    while total > cfg.MAX_CHARS_PER_PAGE:
        if num_t3 > 0:
            num_t3 -= 1
        elif num_t2 > 0:
            num_t2 -= 1
        elif num_t1 > 0:
            num_t1 -= 1
        elif num_t0 > 1:
            num_t0 -= 1
        else:
            break
        total -= 1
    
    # Pad if under minimum
    while total < cfg.MIN_CHARS_PER_PAGE and tier_3:
        num_t3 += 1
        total += 1
    
    # Select characters
    chosen = [main_char]
    
    def sample_from_tier(tier, count, exclude=None):
        if count <= 0 or not tier:
            return []
        pool = [c for c in tier if c != exclude]
        return random.sample(pool, min(count, len(pool)))
    
    if main_tier == 0:
        num_t0 -= 1
    chosen.extend(sample_from_tier(tier_0, num_t0, main_char))
    
    if main_tier == 1:
        num_t1 -= 1
    chosen.extend(sample_from_tier(tier_1, num_t1, main_char))
    
    if main_tier == 2:
        num_t2 -= 1
    chosen.extend(sample_from_tier(tier_2, num_t2, main_char))
    
    chosen.extend(sample_from_tier(tier_3, num_t3))
    
    return chosen


def generate_single_page(chosen_chars, char_catalog, avg_width):
    """Generate a single page image."""
    try:
        if avg_width <= 0:
            avg_width = 300
        
        scale = random.uniform(*cfg.PAGE_SCALE_FACTOR_RANGE)
        page_width = max(int(avg_width * scale), 100)
        content_width = max(page_width - (cfg.PAGE_MARGIN + cfg.PAGE_BUFFER) * 2, 20)
        
        # Load and resize characters
        imgs, chars = [], []
        for char in chosen_chars:
            img = get_random_variant(char, char_catalog)
            w, h = img.size
            if w == 0 or h == 0:
                continue
            ratio = h / w
            new_h = int(content_width * ratio)
            if new_h <= 0:
                continue
            img = img.resize((content_width, new_h), Image.LANCZOS)
            imgs.append(img)
            chars.append(char)
        
        if not imgs:
            return None, None
        
        # Calculate spacing
        spacings = [int(random.randint(*cfg.LINE_SPACING_RANGE) * 
                       random.uniform(*cfg.SPACING_MULTIPLIER_RANGE)) for _ in imgs]
        
        # Calculate page height
        total_height = sum(img.height + s for img, s in zip(imgs, spacings))
        page_height = total_height + (cfg.PAGE_MARGIN + cfg.PAGE_BUFFER) * 2
        
        # Create background
        paper = random_paper_texture()
        if paper:
            paper = paper.resize((page_width, page_height), Image.LANCZOS)
        else:
            paper = Image.new("RGB", (page_width, page_height), cfg.DEFAULT_PAPER_COLOR)
        
        # Paste characters
        y = cfg.PAGE_MARGIN + cfg.PAGE_BUFFER
        for img, spacing in zip(imgs, spacings):
            img = augment_char_image(img)
            x = paper.width // 2 - img.width // 2 + random.randint(*cfg.HORIZONTAL_DRIFT_RANGE)
            y_pos = y + random.randint(*cfg.VERTICAL_DRIFT_RANGE)
            x = max(0, min(x, paper.width - img.width))
            y_pos = max(0, min(y_pos, paper.height - img.height))
            
            sub_paper = paper.crop((x, y_pos, x + img.width, y_pos + img.height))
            blended = blend_with_paper(sub_paper, img)
            paper.paste(blended, (x, y_pos))
            y += img.height + spacing
        
        # Apply effects
        paper = paper.filter(ImageFilter.GaussianBlur(random.uniform(0.3, 0.6)))
        paper = add_noise(paper)
        paper = ImageEnhance.Contrast(paper).enhance(random.uniform(0.9, 1.0))
        paper = apply_random_crop(paper)
        
        return paper, "".join(chars)
    except Exception as e:
        print(f"Error generating single page: {e}")
        return None, None


def process_page(task):
    """Worker function for page generation."""
    page_idx, main_char, main_tier, tiers, char_catalog, avg_width, _ = task
    tier_0, tier_1, tier_2, tier_3 = tiers
    
    try:
        chosen = select_page_chars(main_char, main_tier, tier_0, tier_1, tier_2, tier_3)
        paper, text = generate_single_page(chosen, char_catalog, avg_width)
        
        if paper is None:
            return {'success': False, 'page_idx': page_idx}
        
        page_name = f"page_{page_idx:04d}.jpg"
        paper.save(os.path.join(IMG_DIR, page_name), quality=95)
        
        label_name = f"page_{page_idx:04d}.txt"
        with open(os.path.join(LABEL_DIR, label_name), "w", encoding="utf-8") as f:
            f.write(text)
        
        return {
            'success': True,
            'page_idx': page_idx,
            'page_name': page_name,
            'text': text,
            'main_char': main_char,
            'main_tier': main_tier,
            'size': f"{paper.width}x{paper.height}"
        }
    except Exception:
        return {'success': False, 'page_idx': page_idx}


def convert_to_training_format():
    """
    Đọc img/ và label/ trong OUTPUT_DIR, tạo file caption.txt.

    Cấu trúc sau khi hoàn tất:
        synthesize_train/
        ├── img/
        │   ├── page_0001.jpg
        │   ├── page_0002.jpg
        │   └── ...
        ├── label/
        │   ├── page_0001.txt
        │   ├── page_0002.txt
        │   └── ...
        └── caption.txt  (page_0001.jpg\tlabel1\npage_0002.jpg\tlabel2\n...)
    """
    print(f"\n{'='*70}")
    print("CHUYỂN ĐỔI SANG TRAINING FORMAT")
    print(f"{'='*70}")

    caption_path = os.path.join(cfg.OUTPUT_DIR, "caption.txt")

    label_files = sorted([
        f for f in os.listdir(LABEL_DIR)
        if f.endswith('.txt')
    ])

    if not label_files:
        print(f"Không tìm thấy file label nào trong {LABEL_DIR}")
        return

    captions = []
    success_count = 0
    failed_count = 0

    for label_file in tqdm(label_files, desc="Building caption.txt"):
        try:
            label_path = os.path.join(LABEL_DIR, label_file)
            with open(label_path, 'r', encoding='utf-8') as f:
                label = f.read().strip()

            if not label:
                failed_count += 1
                continue

            img_name = label_file.replace('.txt', '.jpg')
            img_path = os.path.join(IMG_DIR, img_name)

            if not os.path.exists(img_path):
                print(f"Không tìm thấy ảnh {img_name}, bỏ qua")
                failed_count += 1
                continue

            captions.append(f"{img_name}\t{label}")
            success_count += 1

        except Exception as e:
            print(f"Lỗi xử lý {label_file}: {e}")
            failed_count += 1
            continue

    with open(caption_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(captions))

    print(f"\nCaption: {caption_path}")
    print(f"Thành công: {success_count}, Lỗi: {failed_count}")
    print(f"{'='*70}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("Loading vocabularies...")
    train_vocab = load_vocab_from_file(cfg.TRAIN_FILE)
    val_vocab = load_vocab_from_file(cfg.VAL_FILE)
    train_counts = count_chars_in_file(cfg.TRAIN_FILE)
    
    print(f"Train vocab: {len(train_vocab)} chars")
    print(f"Val vocab: {len(val_vocab)} chars")
    print(f"Val-only: {len(val_vocab - train_vocab)} chars")
    
    char_stats = load_statistics(cfg.STATS_FILE)
    avg_width, avg_height = get_avg_page_dimensions(cfg.STATS_FILE, cfg.STATS_SAMPLE_PERCENT)
    char_catalog = build_char_catalog(cfg.CHAR_DIR)
    
    if not char_catalog:
        print("ERROR: No characters found!")
        return
    
    tier_0, tier_1, tier_2, tier_3, pages = calculate_tiers(
        char_catalog, train_vocab, val_vocab, train_counts
    )
    
    if not pages:
        print("ERROR: No pages to generate!")
        return
    
    print(f"\nGenerating {len(pages)} pages with {cfg.MAX_WORKERS} threads...")
    
    metadata = {}
    success_count = 0
    failed_count = 0
    gen_count = Counter()
    
    # Prepare tasks
    tasks = [
        (idx, char, tier, (tier_0, tier_1, tier_2, tier_3), char_catalog, avg_width, char_stats)
        for idx, (char, tier) in enumerate(pages, start=1)
    ]
    
    # Execute
    with ThreadPoolExecutor(max_workers=cfg.MAX_WORKERS) as executor:
        futures = {executor.submit(process_page, task): task for task in tasks}
        
        for future in tqdm(as_completed(futures), total=len(tasks), desc="Generating"):
            try:
                result = future.result()
                if result['success']:
                    with metadata_lock:
                        metadata[result['page_name']] = {
                            'text': result['text'],
                            'main_char': result['main_char'],
                            'main_tier': result['main_tier'],
                            'size': result['size']
                        }
                    with counter_lock:
                        success_count += 1
                        gen_count.update(result['text'])
                else:
                    with counter_lock:
                        failed_count += 1
            except Exception as e:
                with counter_lock:
                    failed_count += 1
                safe_print(f"Exception: {e}")
    
    # Save metadata
    with open(os.path.join(cfg.OUTPUT_DIR, cfg.METADATA_FILE), 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    
    # Print summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Success: {success_count}/{len(pages)}")
    print(f"Failed: {failed_count}/{len(pages)}")
    print(f"Output: {cfg.OUTPUT_DIR}")
    
    # Coverage
    print("\nCoverage:")
    for i, tier in enumerate([tier_0, tier_1, tier_2, tier_3]):
        covered = sum(1 for c in tier if gen_count.get(c, 0) > 0)
        pct = covered / max(len(tier), 1) * 100
        print(f"  Tier {i}: {covered}/{len(tier)} ({pct:.1f}%)")
    
    print("="*70)
    
    # Tạo caption.txt từ img/ + label/
    convert_to_training_format()


if __name__ == "__main__":
    main()
