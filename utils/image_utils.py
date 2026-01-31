"""
Image processing utilities for synthetic data generation.
"""
import os
import random
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageDraw, ImageFont, ImageOps
from functools import lru_cache

from config import DataGeneration as cfg


@lru_cache(maxsize=cfg.CHAR_CACHE_SIZE)
def load_char_variants(folder_path):
    """Load and normalize character image variants from folder."""
    variants = []
    try:
        for fname in os.listdir(folder_path):
            if not fname.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                continue
            if any(s in fname.lower() for s in cfg.EXCLUDED_STYLES):
                continue
            try:
                img = Image.open(os.path.join(folder_path, fname)).convert("RGBA")
                variants.append(normalize_char_image(img))
            except Exception as e:
                print(f"Error loading character variant {fname}: {e}")
                continue
    except Exception as e:
        print(f"Error loading character variants from {folder_path}: {e}")
        pass
    return variants


def normalize_char_image(img):
    """Normalize character image to have transparent background."""
    gray = img.convert("L")
    if np.array(gray).mean() < 128:
        gray = ImageOps.invert(gray)
    alpha = gray.point(lambda p: 255 - p)
    alpha = alpha.filter(ImageFilter.GaussianBlur(0.8))
    result = Image.new("RGBA", img.size, (0, 0, 0, 0))
    result.putalpha(alpha)
    return result


def render_fallback_text(char_text, font_path=None, font_size=None):
    """Render character using font as fallback."""
    font_path = font_path or cfg.FALLBACK_FONT
    font_size = font_size or cfg.FALLBACK_FONT_SIZE
    
    try:
        font = ImageFont.truetype(font_path, font_size)
    except Exception as e:
        print(f"Error loading font {font_path}: {e}")
        font = ImageFont.load_default()
    
    dummy = Image.new("RGBA", (10, 10))
    draw = ImageDraw.Draw(dummy)
    try:
        bbox = draw.textbbox((0, 0), char_text, font=font)
        w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    except Exception as e:
        print(f"Error getting textbbox for {char_text}: {e}")
        w, h = draw.textsize(char_text, font=font)
    
    img = Image.new("RGBA", (w + 20, h + 20), (255, 255, 255, 0))
    draw = ImageDraw.Draw(img)
    draw.text((10, 10), char_text, font=font, fill=(40, 40, 40, 255))
    return img


def get_random_variant(char, char_catalog):
    """Get random variant image for a character."""
    if char not in char_catalog:
        return render_fallback_text(char)
    variants = load_char_variants(char_catalog[char])
    if not variants:
        return render_fallback_text(char)
    return random.choice(variants)


def augment_char_image(img):
    """Apply random augmentation to character image."""
    try:
        angle = random.uniform(*cfg.ROTATE_RANGE)
        img = img.rotate(angle, expand=True, fillcolor=(0, 0, 0, 0), resample=Image.BICUBIC)
        img = ImageEnhance.Brightness(img).enhance(random.uniform(*cfg.BRIGHTNESS_RANGE))
        img = ImageEnhance.Contrast(img).enhance(random.uniform(*cfg.CONTRAST_RANGE))
        blur = random.uniform(*cfg.BLUR_RANGE)
        if blur > 0:
            img = img.filter(ImageFilter.GaussianBlur(blur))
        return img
    except Exception as e:
        print(f"Error augmenting character image: {e}")
        return img


def add_noise(img):
    """Add random noise to image."""
    try:
        arr = np.array(img).astype(np.int16)
        noise = np.random.normal(0, cfg.NOISE_STD, arr.shape)
        return Image.fromarray(np.clip(arr + noise, 0, 255).astype(np.uint8))
    except Exception as e:
        print(f"Error adding noise to image: {e}")
        return img


def blend_with_paper(paper, ink_img):
    """Blend ink image with paper texture."""
    try:
        ink_blur = ink_img.filter(ImageFilter.GaussianBlur(0.2))
        blended = Image.alpha_composite(paper.convert("RGBA"), ink_blur)
        alpha = random.randrange(45, 70) / 100
        return Image.blend(paper.convert("RGB"), blended.convert("RGB"), alpha)
    except:
        return paper


def random_paper_texture():
    """Load random paper texture image."""
    if not os.path.exists(cfg.PAPER_DIR):
        return None
    try:
        files = [os.path.join(cfg.PAPER_DIR, f) for f in os.listdir(cfg.PAPER_DIR) 
                 if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        return Image.open(random.choice(files)).convert("RGB") if files else None
    except:
        return None


def apply_random_crop(img, mode=None):
    """Apply random cropping to image."""
    if not cfg.CROP_ENABLED:
        return img
    
    if mode is None:
        mode = random.choices(cfg.CROP_MODES, weights=cfg.CROP_MODE_WEIGHTS, k=1)[0]
    
    width, height = img.size
    
    if mode == "symmetric":
        margin = random.randint(*cfg.CROP_MARGIN_RANGE)
        crop_left = crop_top = crop_right = crop_bottom = margin
    elif mode == "asymmetric":
        crop_left = random.randint(*cfg.CROP_MARGIN_RANGE)
        crop_top = random.randint(*cfg.CROP_MARGIN_RANGE)
        crop_right = random.randint(*cfg.CROP_MARGIN_RANGE)
        crop_bottom = random.randint(*cfg.CROP_MARGIN_RANGE)
    else:  # edge_only
        crop_left = random.randint(*cfg.CROP_MARGIN_RANGE) if random.random() < cfg.EDGE_CROP_CHANCE else 0
        crop_top = random.randint(*cfg.CROP_MARGIN_RANGE) if random.random() < cfg.EDGE_CROP_CHANCE else 0
        crop_right = random.randint(*cfg.CROP_MARGIN_RANGE) if random.random() < cfg.EDGE_CROP_CHANCE else 0
        crop_bottom = random.randint(*cfg.CROP_MARGIN_RANGE) if random.random() < cfg.EDGE_CROP_CHANCE else 0
        if crop_left + crop_top + crop_right + crop_bottom == 0:
            margin = random.randint(*cfg.CROP_MARGIN_RANGE)
            crop_left = margin
    
    # Limit crop to 25% of dimensions
    crop_left = min(crop_left, width * 0.25)
    crop_right = min(crop_right, width * 0.25)
    crop_top = min(crop_top, height * 0.25)
    crop_bottom = min(crop_bottom, height * 0.25)
    
    left, top = int(crop_left), int(crop_top)
    right, bottom = int(width - crop_right), int(height - crop_bottom)
    
    if right <= left or bottom <= top:
        return img
    
    try:
        return img.crop((left, top, right, bottom))
    except Exception as e:
        print(f"Error applying random crop: {e}")
        return img


def process_image_to_grayscale(src_path, output_path, rotate=False):
    """Convert image to grayscale BMP format."""
    try:
        img = Image.open(src_path)
        if rotate:
            img = img.rotate(90, expand=True)
        if img.mode != 'L':
            img = img.convert('L')
        img.save(output_path, 'BMP')
        return True
    except Exception as e:
        print(f"Error processing {src_path}: {e}")
        return False
