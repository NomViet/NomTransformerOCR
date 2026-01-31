"""
Crawl Nom character images from zi.tools API.
"""
import os
import json
import base64
import time
import requests
from urllib.parse import quote
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

from config import Crawler as cfg

print_lock = Lock()


def safe_print(*args, **kwargs):
    with print_lock:
        print(*args, **kwargs)


def download_char_images(char, index, force_rerun=False):
    """Download all images for a single character."""
    folder_name = f"{index}_{char}"
    folder_path = os.path.join(cfg.OUTPUT_DIR, folder_name)
    
    # Check if already downloaded
    if not force_rerun and os.path.isdir(folder_path):
        existing = len([f for f in os.listdir(folder_path) if f.endswith('.png')])
        if existing >= cfg.MIN_IMAGES:
            return {'index': index, 'char': char, 'status': 'skipped', 'count': existing}
        safe_print(f"[{index}] '{char}' - Thư mục có {existing} file, tải lại...")
    
    os.makedirs(folder_path, exist_ok=True)
    safe_print(f"[{index}] Đang xử lý: '{char}'")
    
    headers = {
        'Accept': 'application/json, text/plain, */*',
        'Content-Type': 'application/json;charset=UTF-8',
        'Origin': 'https://zi.tools',
        'Referer': f"{cfg.BASE_URL}{quote(char)}",
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
    }
    
    try:
        response = requests.post(
            cfg.API_URL,
            data=json.dumps({"zi": char}),
            headers=headers,
            timeout=30
        )
        response.raise_for_status()
        data = response.json()
        
        count = 0
        if 'tu' in data and data['tu']:
            for style_group in data['tu'][0].get('styles', []):
                for row in style_group.get('rows', []):
                    if isinstance(row, list) and len(row) >= 5:
                        b64_data = row[4]
                        if isinstance(b64_data, str) and b64_data.startswith('iVBOR'):
                            try:
                                filename = f"{row[0]}_{row[1]}.png"
                                filepath = os.path.join(folder_path, filename)
                                with open(filepath, 'wb') as f:
                                    f.write(base64.b64decode(b64_data))
                                count += 1
                            except:
                                continue
        
        if count == 0:
            safe_print(f"[{index}] '{char}' - Không tìm thấy hình ảnh")
        else:
            safe_print(f"[{index}] '{char}' - Đã lưu {count} hình ảnh")
        
        return {'index': index, 'char': char, 'status': 'success', 'count': count}
    
    except requests.exceptions.HTTPError as e:
        safe_print(f"[{index}] '{char}' - HTTP Error: {e}")
        return {'index': index, 'char': char, 'status': 'error', 'count': 0}
    except Exception as e:
        safe_print(f"[{index}] '{char}' - Error: {e}")
        return {'index': index, 'char': char, 'status': 'error', 'count': 0}


def main():
    print("="*70)
    print("NOM CHARACTER IMAGE CRAWLER")
    print("="*70)
    
    # Load characters
    try:
        with open(cfg.INPUT_FILE, "r", encoding="utf-8") as f:
            characters = [line.strip() for line in f if line.strip()]
        print(f"Loaded {len(characters)} characters from '{cfg.INPUT_FILE}'")
    except FileNotFoundError:
        print(f"ERROR: File not found '{cfg.INPUT_FILE}'")
        return
    except Exception as e:
        print(f"ERROR: {e}")
        return
    
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    
    print(f"Output: {cfg.OUTPUT_DIR}")
    print(f"Workers: {cfg.MAX_WORKERS}")
    print(f"Force rerun: {cfg.FORCE_RERUN}")
    print("="*70)
    
    if not characters:
        print("No characters to process.")
        return
    
    total = len(characters)
    results = []
    start_time = time.time()
    
    print(f"\nStarting download of {total} characters...\n")
    
    with ThreadPoolExecutor(max_workers=cfg.MAX_WORKERS) as executor:
        futures = {
            executor.submit(download_char_images, char, idx, cfg.FORCE_RERUN): (idx, char)
            for idx, char in enumerate(characters)
        }
        
        completed = 0
        for future in as_completed(futures):
            completed += 1
            try:
                results.append(future.result())
            except Exception:
                idx, char = futures[future]
                results.append({'index': idx, 'char': char, 'status': 'exception', 'count': 0})
            
            if completed % 10 == 0 or completed == total:
                elapsed = time.time() - start_time
                pct = completed / total * 100
                safe_print(f"\n--- Progress: {completed}/{total} ({pct:.1f}%) - {elapsed:.1f}s ---\n")
    
    # Summary
    elapsed = time.time() - start_time
    success = sum(1 for r in results if r['status'] == 'success')
    skipped = sum(1 for r in results if r['status'] == 'skipped')
    errors = sum(1 for r in results if r['status'] in ['error', 'exception'])
    total_images = sum(r['count'] for r in results)
    
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Total characters: {total}")
    print(f"Success: {success}")
    print(f"Skipped: {skipped}")
    print(f"Errors: {errors}")
    print(f"Total images: {total_images}")
    print(f"Time: {elapsed:.2f}s")
    print(f"Speed: {total/elapsed:.2f} chars/sec")
    print("="*70)


if __name__ == "__main__":
    main()
