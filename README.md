# NomTransformerOCR

OCR chữ Nôm (Hán Nôm) dựa trên CoMER (Coverage-based Model for Handwritten Mathematical Expression Recognition). Pipeline: tải dataset → sinh dữ liệu synthetic → chuẩn hóa định dạng CoMER → build vocab → train → test/inference.

## Cấu trúc project

```
NomTransformerOCR/
├── config.py                         # Cấu hình tập trung
├── pyproject.toml
├── requirements.txt
├── .gitignore
├── utils/
│   ├── image_utils.py
│   └── file_utils.py
├── comer/                            # Model CoMER
│   ├── datamodule/                   # DataLoader, vocab, dictionary.txt
│   ├── model/
│   └── lit_comer.py
├── crawl_nom_characters_zi_tools.py  # Bước 0: Tải ảnh ký tự từ zi.tools
├── gen_nom_ocr_data.py               # Bước 1: Sinh dữ liệu synthetic
├── convert_data_to_comer_format.py   # Bước 2: Chuyển sang định dạng CoMER
├── build_vocab_dict.py               # Bước 3: Build từ điển (dictionary.txt)
├── train.py                          # Bước 5: Train model
├── evaluate.py                       # Bước 6: Đánh giá (metrics)
├── run_single.py                     # Inference 1 ảnh
├── convert_data_to_parquet.py        # (Tùy chọn) Chuyển sang Parquet
└── visualize_augmentation.py        # (Tùy chọn) Xem phân bố tier / augmentation
```

## Chuẩn bị dữ liệu

### Tải dataset NomNaOCR từ Kaggle

Trước khi chạy pipeline, cần có dataset NomNaOCR (Train.txt, Validate.txt, All.txt và thư mục ảnh tương ứng):

1. Truy cập: https://www.kaggle.com/datasets/quandang/nomnaocr
2. Đăng nhập Kaggle và tải về (hoặc dùng Kaggle CLI).

**Cách 1: Tải thủ công**  
Nhấn nút "Download" trên trang Kaggle, giải nén vào thư mục (ví dụ `NomNaOCR_dataset/` hoặc đường dẫn bạn dùng trong `config.py`).

**Cách 2: Dùng Kaggle CLI**

```bash
pip install kaggle
# Cấu hình API key: đặt kaggle.json vào ~/.kaggle/ (lấy từ Kaggle → Account → Create New API Token)

kaggle datasets download -d quandang/nomnaocr
unzip nomnaocr.zip -d NomNaOCR_dataset
```

Sau khi tải, chỉnh trong `config.py` → `DataGeneration` và `CoMERFormat`: `STATS_FILE`, `TRAIN_FILE`, `VAL_FILE`, `DATASET_DIR` trỏ đúng đường dẫn chứa Train.txt, Validate.txt, All.txt và ảnh.

---

## Cài đặt

Dùng [uv](https://docs.astral.sh/uv/) (hoặc pip/conda tương ứng).

```bash
# Cài uv (macOS/Linux)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Tạo virtual env
uv venv --python 3.10
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# Dependency
uv pip install -r requirements.txt
uv pip install torch torchvision pytorch-lightning
uv pip install -e .

# (Tùy chọn) Nếu chạy evaluate.py: uv pip install tensorflow
```

---

## Quy trình từ data đến train và test

### Bước 0 (tùy chọn): Tải ảnh ký tự từ zi.tools

Dùng khi cần thêm ảnh ký tự Nôm để sinh data. Cấu hình trong `config.py` → class `Crawler`.

```bash
# Chuẩn bị file danh sách ký tự: dictionary_7k5.txt (mỗi dòng một ký tự)
python crawl_nom_characters_zi_tools.py
```

**Output:** thư mục `all_character_images/` (mặc định), mỗi ký tự một thư mục con chứa ảnh.

---

### Bước 1: Sinh dữ liệu synthetic

Đọc Train/Validate từ NomNaOCR, phân tier ký tự (theo tần suất), sinh ảnh trang giả lập và lưu **ảnh vào `img/`**, **label (text) vào `label/`**. Cuối script tự chạy **convert to training format** để tạo file `caption.txt` từ `img/` + `label/`.

Cấu hình: `config.py` → class `DataGeneration` (đường dẫn Train/Val/All, `CHAR_DIR`, `OUTPUT_DIR`, quota tier, augmentation).

```bash
python gen_nom_ocr_data.py
```

**Output:** thư mục `synthesize_train/` (mặc định), cấu trúc:

```
synthesize_train/
├── img/                    # Ảnh trang: page_0001.jpg, page_0002.jpg, ...
├── label/                  # Label từng trang: page_0001.txt, page_0002.txt, ... (chỉ nội dung text)
├── caption.txt             # File ghép cho training: mỗi dòng "page_XXXX.jpg\t<label>"
└── generation_metadata.json
```

- Ảnh được ghi trực tiếp vào `img/`, label vào `label/` trong lúc sinh.
- Sau khi sinh xong, script gọi nội bộ `convert_to_training_format()` để tạo `caption.txt` từ `img/` và `label/`.

---

### Bước 2: Chuyển sang định dạng CoMER

- Đọc dữ liệu gốc từ NomNaOCR (Train.txt, Validate.txt) → tạo các split `train`, `val`, `val_300`, `test` (ảnh grayscale + caption CoMER).
- Đọc dữ liệu synthetic từ `synthesize_train/` (thư mục `img/` + file `caption.txt`) → tạo split `train_generated`.

Cấu hình: `config.py` → class `CoMERFormat` (`DATASET_DIR`, `OUTPUT_DIR`, `GENERATED_SOURCE_DIR` = nguồn synthetic, mặc định trùng `DataGeneration.OUTPUT_DIR` = `synthesize_train`).

```bash
python convert_data_to_comer_format.py
```

**Output:** thư mục `comer_data/` (mặc định), bên trong:

```
comer_data/
└── data/
    ├── train/          # img/ + caption.txt (định dạng CoMER)
    ├── val/
    ├── val_300/
    ├── test/
    └── train_generated # Từ synthesize_train/img/ + caption.txt
```

Mỗi split có `img/` (ảnh .bmp grayscale) và `caption.txt` (định dạng token CoMER).

---

### Bước 3: Build từ điển (vocab)

Tạo `comer/datamodule/dictionary.txt` từ các file caption đã tạo ở bước 2.

```bash
python build_vocab_dict.py
```

Cần chạy sau khi đã có `comer_data/data/`. Khi đổi dataset hoặc thêm ký tự mới, nên chạy lại bước 3 rồi mới train.

---

### Bước 4: Chuẩn bị data zip cho training

`train.py` đọc dữ liệu từ **một file zip**. Trong zip phải có thư mục `data/`, bên trong là `train/`, `val_300/`, … (mỗi thư mục có `img/` và `caption.txt`).

Tạo zip từ output của bước 2 (đã có sẵn `comer_data/data/`):

```bash
cd comer_data && zip -r data.zip data && cd ..
```

Kết quả: `comer_data/data.zip` chứa `data/train/`, `data/val_300/`, … Trong `train.py` chỉnh `zipfile_path` trỏ tới file zip này (mặc định: `comer_data/data.zip`).

---

### Bước 5: Train model

```bash
python train.py
```

Trong `train.py` có thể chỉnh: `zipfile_path`, `test_year` (split validation, ví dụ `"val_300"`), batch size, epoch, … Checkpoint lưu trong `lightning_logs/version_*/checkpoints/`.

---

### Bước 6: Test / inference

**Inference một ảnh:**

```bash
python run_single.py <đường_dẫn_ảnh> <đường_dẫn_checkpoint> [device]
```

Ví dụ:

```bash
python run_single.py comer_data/data/val/img/val000001.bmp lightning_logs/version_0/checkpoints/epoch=100-step=xxx.ckpt cuda:0
```

**Đánh giá theo metrics (Sequence Accuracy, Character Accuracy, CER):** chỉnh trong `evaluate.py` (danh sách ảnh, file caption, đường dẫn checkpoint), rồi chạy:

```bash
python evaluate.py
```

---

## Tóm tắt thứ tự chạy

| Bước | Script | Ý nghĩa / Output |
|------|--------|-------------------|
| - | Tải NomNaOCR từ Kaggle | Cần có Train.txt, Validate.txt, All.txt + ảnh trước khi chạy pipeline |
| 0 | `crawl_nom_characters_zi_tools.py` | Tùy chọn: tải ảnh ký tự → `all_character_images/` |
| 1 | `gen_nom_ocr_data.py` | Sinh data synthetic → `synthesize_train/` (img/, label/, caption.txt) |
| 2 | `convert_data_to_comer_format.py` | Chuyển sang CoMER → `comer_data/data/` (train, val, val_300, test, train_generated) |
| 3 | `build_vocab_dict.py` | Tạo `comer/datamodule/dictionary.txt` |
| 4 | Tạo zip `comer_data/data.zip` | Từ `comer_data/data/` để train.py đọc |
| 5 | `train.py` | Train model; checkpoint trong `lightning_logs/` |
| 6 | `run_single.py` / `evaluate.py` | Inference 1 ảnh hoặc đánh giá hàng loạt |

## Cấu hình

Mọi đường dẫn và tham số chính nằm trong `config.py`:

- **DataGeneration**: `STATS_FILE`, `TRAIN_FILE`, `VAL_FILE`, `CHAR_DIR`, `OUTPUT_DIR` (= `synthesize_train`), `IMG_SUBDIR`, `LABEL_SUBDIR`, quota tier, augmentation.
- **CoMERFormat**: `DATASET_DIR`, `OUTPUT_DIR` (= `comer_data`), `GENERATED_SOURCE_DIR` (= nguồn synthetic, mặc định `synthesize_train`).
- **ParquetConversion**: dùng cho `convert_data_to_parquet.py` (data root, splits, shard size).
- **Crawler**: file danh sách ký tự, thư mục lưu ảnh, số worker.

Chỉnh các class tương ứng khi đổi dataset hoặc môi trường.
