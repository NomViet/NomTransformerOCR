# NomTransformerOCR

OCR chữ Nôm (Hán Nôm) dựa trên CoMER (Coverage-based Model for Handwritten Mathematical Expression Recognition). Pipeline: tạo dữ liệu → chuẩn hóa định dạng → build vocab → train → test/inference.

## Cấu trúc project

```
NomTransformerOCR/
├── config.py                    # Cấu hình tập trung (config.py)
├── utils/                       # Tiện ích dùng chung
│   ├── image_utils.py
│   └── file_utils.py
├── comer/                       # Model CoMER
│   ├── datamodule/              # DataLoader, vocab, dictionary.txt
│   ├── model/
│   └── lit_comer.py
├── crawl_nom_characters_zi_tools.py   # Bước 0: Tải ảnh ký tự từ zi.tools
├── gen_nom_ocr_data.py                # Bước 1: Sinh dữ liệu synthetic
├── convert_data_to_comer_format.py   # Bước 2: Chuyển sang định dạng CoMER
├── build_vocab_dict.py                # Bước 3: Build từ điển (dictionary.txt)
├── train.py                           # Bước 4: Train model
├── evaluate.py                        # Bước 5: Đánh giá (metrics)
├── run_single.py                      # Inference 1 ảnh
├── convert_data_to_parquet.py         # (Tùy chọn) Chuyển sang Parquet
└── requirements.txt
```

## Cài đặt

```bash
# Tạo môi trường
conda create -y -n nom_ocr python=3.10
conda activate nom_ocr

# PyTorch (chọn phiên bản phù hợp CUDA)
conda install pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia

# Các dependency khác
pip install -e .
pip install pytorch-lightning pillow tqdm
# Nếu chạy evaluate.py: pip install tensorflow
```

## Quy trình từ data đến train và test

### Bước 0 (tùy chọn): Tải ảnh ký tự từ zi.tools

Dùng khi cần thêm ảnh ký tự để sinh data. Cấu hình trong `config.py` → class `Crawler`.

```bash
# Chuẩn bị file danh sách ký tự: dictionary_7k5.txt (mỗi dòng một ký tự)
python crawl_nom_characters_zi_tools.py
```

Output: thư mục `all_character_images/` (mặc định), mỗi ký tự một thư mục con.

---

### Bước 1: Sinh dữ liệu synthetic

Đọc Train/Validate, phân tier ký tự, sinh ảnh giả lập và ghi `caption.txt`. Cấu hình trong `config.py` → class `DataGeneration`.

```bash
python gen_nom_ocr_data.py
```

Output: thư mục `outputs_balanced/` (mặc định), gồm ảnh + `caption.txt`.

---

### Bước 2: Chuyển sang định dạng CoMER

- Đọc dữ liệu gốc (Train.txt, Validate.txt) và/hoặc thư mục synthetic.
- Tạo cấu trúc `train`, `val`, `val_300`, `test`, `train_generated` với `img/` và `caption.txt`.

Cấu hình trong `config.py` → class `CoMERFormat` (đường dẫn dataset, output, `GENERATED_SOURCE_DIR` nếu dùng synthetic).

```bash
python convert_data_to_comer_format.py
```

Output: thư mục dạng `comer_data_synthesize_10k/data/` (mặc định), bên trong có `train/`, `val/`, `val_300/`, `test/`, `train_generated/`, mỗi thư mục có `img/` và `caption.txt`.

---

### Bước 3: Build từ điển (vocab)

Tạo `comer/datamodule/dictionary.txt` từ các file caption đã tạo ở bước 2.

```bash
python build_vocab_dict.py
```

Cần chạy sau khi đã có dữ liệu CoMER (bước 2). Sau khi đổi dataset, nên chạy lại bước 3 rồi mới train.

---

### Bước 4: Chuẩn bị data zip cho training

`train.py` đọc dữ liệu từ một file zip. Cấu trúc trong zip: có thư mục `data/`, bên trong là `train/`, `val_300/`, … (mỗi thư mục có `img/` và `caption.txt`).

Ví dụ tạo zip từ thư mục output của bước 2:

```bash
# Giả sử output bước 2 là comer_data_synthesize_10k/data/
mkdir -p comer_data
cp -r comer_data_synthesize_10k/data comer_data/
cd comer_data && zip -r data.zip data && cd ..
```

Hoặc nếu bạn đặt tên thư mục là `comer_data/data/` từ đầu:

```bash
cd comer_data && zip -r data.zip data && cd ..
```

Kết quả: `comer_data/data.zip` chứa `data/train/`, `data/val_300/`, …

Trong `train.py` cần chỉnh `zipfile_path` trỏ đúng file zip này (mặc định có thể là `comer_data/data.zip`).

---

### Bước 5: Train model

```bash
python train.py
```

Trong `train.py` có thể chỉnh:

- `zipfile_path`: đường dẫn tới `comer_data/data.zip` (hoặc file zip tương ứng).
- `test_year`: split dùng để validation (ví dụ `"val_300"`).
- Các hyperparameter của `LitCoMER` và `CROHMEDatamodule` (batch size, epoch, …).

Checkpoint được lưu trong `lightning_logs/version_*/checkpoints/`.

---

### Bước 6: Test / inference

**Inference một ảnh (không cần label, không tính metrics):**

```bash
python run_single.py <đường_dẫn_ảnh> <đường_dẫn_checkpoint> [device]
```

Ví dụ:

```bash
python run_single.py comer_data/data/val/img/val000001.bmp lightning_logs/version_0/checkpoints/epoch=100-step=xxx.ckpt cuda:0
```

**Đánh giá theo metrics (Sequence Accuracy, Character Accuracy, CER):**

Chỉnh trong `evaluate.py` (hoặc gọi từ code): danh sách ảnh, file caption để lấy label, đường dẫn checkpoint. Sau đó chạy:

```bash
python evaluate.py
```

---

## Tóm tắt thứ tự chạy

| Bước | Script | Ghi chú |
|------|--------|--------|
| 0 | `crawl_nom_characters_zi_tools.py` | Tùy chọn, khi cần thêm ảnh ký tự |
| 1 | `gen_nom_ocr_data.py` | Sinh data synthetic → `outputs_balanced/` |
| 2 | `convert_data_to_comer_format.py` | Tạo cấu trúc CoMER (train/val/val_300/test/train_generated) |
| 3 | `build_vocab_dict.py` | Tạo `comer/datamodule/dictionary.txt` |
| 4 | Tạo zip `comer_data/data.zip` | Thủ công hoặc script, từ thư mục `data/` |
| 5 | `train.py` | Train model |
| 6 | `run_single.py` hoặc `evaluate.py` | Inference 1 ảnh hoặc đánh giá hàng loạt |

## Cấu hình

Phần lớn đường dẫn và tham số nằm trong `config.py`:

- **DataGeneration**: đường dẫn Train/Val/All, thư mục ký tự, output sinh data, quota tier, augmentation.
- **CoMERFormat**: đường dẫn dataset gốc, thư mục output CoMER, `GENERATED_SOURCE_DIR`.
- **ParquetConversion**: dùng cho `convert_data_to_parquet.py` (data root, splits, shard size).
- **Crawler**: file danh sách ký tự, thư mục lưu ảnh, số worker.

Chỉnh các class tương ứng khi đổi dataset hoặc môi trường chạy.
