from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "fiqa"
CORPUS_PATH = RAW_DIR / "corpus.jsonl"
INDEX_INPUT_DIR = DATA_DIR / "fiqa_index_input"
QUERIES_PATH = RAW_DIR / "queries.tsv"
QRELS_PATH = RAW_DIR / "qrels.tsv"
QRELS_TREC_PATH = RAW_DIR / "qrels.trec"
METADATA_PATH = RAW_DIR / "metadata.json"
INDEX_DIR = PROJECT_ROOT / "indexes" / "fiqa"
RUNS_DIR = PROJECT_ROOT / "runs"
TRACE_DIR = PROJECT_ROOT / "trace"
RESULTS_DIR = PROJECT_ROOT / "results"

DATASET_NAME = "beir/fiqa/test"

BM25_K1 = 0.9
BM25_B = 0.4
BM25_BASELINE_TOPK = 100

TOPK_LIST = [10, 20, 50]
ALPHAS = [0.2, 0.5, 0.8]

DEFAULT_MODEL_NAME = "distilgpt2"
DEFAULT_DEVICE = "auto"
DEFAULT_BATCH_SIZE = 4
MAX_DOC_CHARS = 1200
MAX_INPUT_LENGTH = 1024
HYBRID_NORMALIZATION = "minmax"

for directory in [
    DATA_DIR,
    RAW_DIR,
    INDEX_INPUT_DIR,
    INDEX_DIR.parent,
    RUNS_DIR,
    TRACE_DIR,
    RESULTS_DIR,
]:
    directory.mkdir(parents=True, exist_ok=True)
