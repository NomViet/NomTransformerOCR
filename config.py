"""
Centralized configuration for NomTransformerOCR project.
"""

# =============================================================================
# DATA GENERATION CONFIG (gen_nom_ocr_data.py)
# =============================================================================

class DataGeneration:
    # Paths
    STATS_FILE = "/bao/Nom-Viet/NomNaOCR/Datasets/Patches/All.txt"
    TRAIN_FILE = "/bao/Nom-Viet/NomNaOCR/Datasets/Patches/Train.txt"
    VAL_FILE = "/bao/Nom-Viet/NomNaOCR/Datasets/Patches/Validate.txt"
    CHAR_DIR = "all_character_images"
    PAPER_DIR = "papers"
    FALLBACK_FONT = "NomNaTong-Regular.ttf"
    OUTPUT_DIR = "synthesize_train"
    IMG_SUBDIR = "img"
    LABEL_SUBDIR = "label"
    METADATA_FILE = "generation_metadata.json"

    # Quota strategy
    PAGES_PER_TIER_0 = 15
    PAGES_PER_TIER_1 = 3
    PAGES_PER_TIER_2 = 0
    PAGES_PER_TIER_3 = 0

    TIER_1_THRESHOLD = 10
    TIER_2_THRESHOLD = 50

    MIN_CHARS_PER_PAGE = 3
    MAX_CHARS_PER_PAGE = 10

    TIER_0_RANGE = (1, 3)
    TIER_1_RANGE = (1, 3)
    TIER_2_RANGE = (0, 3)
    TIER_3_RANGE = (0, 4)

    # Multi-threading
    MAX_WORKERS = 16

    # Augmentation
    PAGE_MARGIN = 20
    PAGE_BUFFER = 15
    DEFAULT_PAPER_COLOR = (245, 238, 220)
    FALLBACK_FONT_SIZE = 64
    LINE_SPACING_RANGE = (5, 10)
    SPACING_MULTIPLIER_RANGE = (0.9, 1.2)
    ROTATE_RANGE = (-2.5, 2.5)
    VERTICAL_DRIFT_RANGE = (-4, 4)
    HORIZONTAL_DRIFT_RANGE = (-6, 6)
    BRIGHTNESS_RANGE = (0.85, 1.15)
    CONTRAST_RANGE = (0.85, 1.15)
    BLUR_RANGE = (0, 0.8)
    NOISE_STD = 5

    # Random crop
    CROP_ENABLED = True
    CROP_MODES = ["symmetric", "asymmetric", "edge_only"]
    CROP_MODE_WEIGHTS = [0.3, 0.5, 0.2]
    CROP_MARGIN_RANGE = (10, 30)
    EDGE_CROP_CHANCE = 0.7

    # Scale
    PAGE_SCALE_FACTOR_RANGE = (0.8, 1.2)

    # Style filtering
    EXCLUDED_STYLES = ["chuanchao", "calligraphy_zhuan", "jiaguwen", "jinwen"]

    # Performance
    CHAR_CACHE_SIZE = 500
    STATS_SAMPLE_PERCENT = 0.05


# =============================================================================
# COMER FORMAT CONFIG (convert_data_to_comer_format.py)
# =============================================================================

class CoMERFormat:
    # Input paths
    DATASET_DIR = '/bao/Nom-Viet/NomNaOCR/Datasets/Patches'
    TRAIN_TRANSCRIPTS_PATH = f'{DATASET_DIR}/Train.txt'
    VAL_TRANSCRIPTS_PATH = f'{DATASET_DIR}/Validate.txt'

    # Output
    OUTPUT_DIR = './comer_data'

    # Processing params
    NUM_TRAIN_SAMPLES = 1000000
    MAX_VAL_300 = 300
    VAL_RATIO = 0.9
    ROTATE = False
    RANDOM_SEED = 1234

    # Source for generated data
    GENERATED_SOURCE_DIR = "synthesize_train"


# =============================================================================
# PARQUET CONVERSION CONFIG (convert_data_to_parquet.py)
# =============================================================================

class ParquetConversion:
    DATA_ROOT = "comer_data/data"
    OUTPUT_DIR = "hf_dataset"
    SPLITS = ["train", "train_generated", "val_300", "val", "test"]
    SHARD_SIZE = 30000


# =============================================================================
# CRAWLER CONFIG (crawl_nom_characters_zi_tools.py)
# =============================================================================

class Crawler:
    INPUT_FILE = "dictionary_7k5.txt"
    OUTPUT_DIR = "all_character_images"
    FORCE_RERUN = False
    MIN_IMAGES = 1
    MAX_WORKERS = 10
    API_URL = "https://zi.tools/api/tu/"
    BASE_URL = "https://zi.tools/zi/"
