import argparse

from tqdm import tqdm

from config import (
    ALPHAS,
    DEFAULT_BATCH_SIZE,
    DEFAULT_DEVICE,
    DEFAULT_MODEL_NAME,
    HYBRID_NORMALIZATION,
    RUNS_DIR,
    TRACE_DIR,
)
from llm_qlm import LLMQLMScorer
from retrieve_bm25 import run_bm25
from utils import (
    alpha_to_tag,
    build_hybrid_scores,
    ensure_exists,
    load_corpus,
    load_json,
    load_queries,
    save_run_bundle,
    sort_doc_scores,
    trim_run,
)


def rerank_candidates(
    bm25_candidates,
    queries,
    corpus,
    scorer: LLMQLMScorer,
    alphas=ALPHAS,
    normalization: str = HYBRID_NORMALIZATION,
):
    qlm_run = {}
    hybrid_runs = {alpha: {} for alpha in alphas}

    for query_id, bm25_scores in tqdm(
        bm25_candidates.items(),
        total=len(bm25_candidates),
        desc="LLM-QLM rerank",
    ):
        ranked_doc_ids = [doc_id for doc_id, _ in sort_doc_scores(bm25_scores)]
        document_texts = [corpus[doc_id] for doc_id in ranked_doc_ids]
        qlm_scores_list = scorer.score_documents(queries[query_id], document_texts)
        qlm_scores = {
            doc_id: score for doc_id, score in zip(ranked_doc_ids, qlm_scores_list)
        }
        qlm_run[query_id] = qlm_scores

        for alpha in alphas:
            hybrid_runs[alpha][query_id] = build_hybrid_scores(
                bm25_scores,
                qlm_scores,
                alpha=alpha,
                normalization=normalization,
            )

    return qlm_run, hybrid_runs


def main():
    parser = argparse.ArgumentParser(description="Re-rank BM25 candidates with LLM-QLM.")
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--model_name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--device", default=DEFAULT_DEVICE)
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--normalization", default=HYBRID_NORMALIZATION)
    parser.add_argument(
        "--bm25_run",
        default="",
        help="Optional path to a saved BM25 JSON run. If omitted, BM25 is executed first.",
    )
    args = parser.parse_args()

    queries = load_queries()
    corpus = load_corpus()

    if args.bm25_run:
        ensure_exists(args.bm25_run, "BM25 run file")
        bm25_candidates = trim_run(load_json(args.bm25_run), args.topk)
    else:
        bm25_candidates = run_bm25(topk=args.topk)

    scorer = LLMQLMScorer(
        model_name=args.model_name,
        device=args.device,
        batch_size=args.batch_size,
    )
    qlm_run, hybrid_runs = rerank_candidates(
        bm25_candidates=bm25_candidates,
        queries=queries,
        corpus=corpus,
        scorer=scorer,
        normalization=args.normalization,
    )

    save_run_bundle(
        qlm_run,
        RUNS_DIR / f"qlm_k{args.topk}.json",
        RUNS_DIR / f"qlm_k{args.topk}.trec",
        TRACE_DIR / f"qlm_k{args.topk}.txt",
        run_name=f"qlm_k{args.topk}",
    )

    for alpha, run in hybrid_runs.items():
        alpha_tag = alpha_to_tag(alpha)
        save_run_bundle(
            run,
            RUNS_DIR / f"hybrid_k{args.topk}_a{alpha_tag}.json",
            RUNS_DIR / f"hybrid_k{args.topk}_a{alpha_tag}.trec",
            TRACE_DIR / f"hybrid_k{args.topk}_a{alpha_tag}.txt",
            run_name=f"hybrid_k{args.topk}_a{alpha_tag}",
        )

    print(f"Saved QLM and Hybrid runs for top-k={args.topk}")


if __name__ == "__main__":
    main()
