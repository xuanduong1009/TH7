import argparse

from tqdm import tqdm

from config import BM25_B, BM25_BASELINE_TOPK, BM25_K1, INDEX_DIR, RUNS_DIR, TRACE_DIR
from utils import configure_java_environment, ensure_exists, load_queries, save_run_bundle


def run_bm25(topk: int = BM25_BASELINE_TOPK):
    ensure_exists(INDEX_DIR, "Lucene index directory")
    configure_java_environment()
    from pyserini.search.lucene import LuceneSearcher

    searcher = LuceneSearcher(str(INDEX_DIR))
    searcher.set_bm25(k1=BM25_K1, b=BM25_B)

    run = {}
    queries = load_queries()
    for query_id, query_text in tqdm(
        queries.items(),
        total=len(queries),
        desc=f"BM25@{topk}",
    ):
        hits = searcher.search(query_text, k=topk)
        run[query_id] = {hit.docid: float(hit.score) for hit in hits}

    return run


def main():
    parser = argparse.ArgumentParser(description="Run BM25 retrieval on FiQA.")
    parser.add_argument("--topk", type=int, default=BM25_BASELINE_TOPK)
    args = parser.parse_args()

    run = run_bm25(topk=args.topk)
    save_run_bundle(
        run,
        RUNS_DIR / f"bm25_top{args.topk}.json",
        RUNS_DIR / f"bm25_top{args.topk}.trec",
        TRACE_DIR / "bm25.txt",
        run_name="bm25",
    )
    print("Saved BM25 outputs:")
    print(f"- {RUNS_DIR / f'bm25_top{args.topk}.json'}")
    print(f"- {RUNS_DIR / f'bm25_top{args.topk}.trec'}")
    print(f"- {TRACE_DIR / 'bm25.txt'}")


if __name__ == "__main__":
    main()
