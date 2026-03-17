import argparse

from config import (
    ALPHAS,
    BM25_B,
    BM25_BASELINE_TOPK,
    BM25_K1,
    DEFAULT_BATCH_SIZE,
    DEFAULT_DEVICE,
    DEFAULT_MODEL_NAME,
    HYBRID_NORMALIZATION,
    RESULTS_DIR,
    RUNS_DIR,
    TOPK_LIST,
    TRACE_DIR,
)
from llm_qlm import LLMQLMScorer
from rerank_qlm import rerank_candidates
from retrieve_bm25 import run_bm25
from utils import (
    alpha_to_tag,
    load_corpus,
    load_metadata,
    load_queries,
    save_json,
    save_run_bundle,
    trim_run,
)


def restrict_run_to_candidates(scored_run, candidate_run):
    restricted = {}
    for query_id, candidates in candidate_run.items():
        restricted[query_id] = {
            doc_id: scored_run[query_id][doc_id]
            for doc_id in candidates
            if doc_id in scored_run[query_id]
        }
    return restricted


def main():
    parser = argparse.ArgumentParser(description="Run the full FiQA BM25 + LLM-QLM experiment suite.")
    parser.add_argument("--model_name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--device", default=DEFAULT_DEVICE)
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--bm25_topk", type=int, default=BM25_BASELINE_TOPK)
    parser.add_argument("--topk_list", type=int, nargs="*", default=TOPK_LIST)
    parser.add_argument("--alphas", type=float, nargs="*", default=ALPHAS)
    parser.add_argument("--normalization", default=HYBRID_NORMALIZATION)
    args = parser.parse_args()

    if any(topk > args.bm25_topk for topk in args.topk_list):
        raise ValueError(
            f"All rerank top-k values must be <= bm25_topk ({args.bm25_topk})."
        )
    if any(alpha < 0.0 or alpha > 1.0 for alpha in args.alphas):
        raise ValueError("All alpha values must be between 0 and 1.")

    queries = load_queries()
    corpus = load_corpus()

    bm25_run = run_bm25(topk=args.bm25_topk)
    save_run_bundle(
        bm25_run,
        RUNS_DIR / f"bm25_top{args.bm25_topk}.json",
        RUNS_DIR / f"bm25_top{args.bm25_topk}.trec",
        TRACE_DIR / "bm25.txt",
        run_name="bm25",
    )

    scorer = LLMQLMScorer(
        model_name=args.model_name,
        device=args.device,
        batch_size=args.batch_size,
    )

    experiment_setup = {
        "dataset": load_metadata(),
        "bm25": {
            "k1": BM25_K1,
            "b": BM25_B,
            "topk": args.bm25_topk,
        },
        "qlm_model": scorer.describe(),
        "rerank_topk_list": args.topk_list,
        "rerank_topk_max": max(args.topk_list),
        "alphas": args.alphas,
        "hybrid_normalization": args.normalization,
    }
    save_json(experiment_setup, RESULTS_DIR / "experiment_setup.json")

    bm25_max_candidates = trim_run(bm25_run, max(args.topk_list))
    qlm_max_run, hybrid_max_runs = rerank_candidates(
        bm25_candidates=bm25_max_candidates,
        queries=queries,
        corpus=corpus,
        scorer=scorer,
        alphas=args.alphas,
        normalization=args.normalization,
    )

    for topk in args.topk_list:
        bm25_candidates = trim_run(bm25_max_candidates, topk)
        qlm_run = restrict_run_to_candidates(qlm_max_run, bm25_candidates)
        save_run_bundle(
            qlm_run,
            RUNS_DIR / f"qlm_k{topk}.json",
            RUNS_DIR / f"qlm_k{topk}.trec",
            TRACE_DIR / f"qlm_k{topk}.txt",
            run_name=f"qlm_k{topk}",
        )

        for alpha, hybrid_max_run in hybrid_max_runs.items():
            alpha_tag = alpha_to_tag(alpha)
            hybrid_run = restrict_run_to_candidates(hybrid_max_run, bm25_candidates)
            save_run_bundle(
                hybrid_run,
                RUNS_DIR / f"hybrid_k{topk}_a{alpha_tag}.json",
                RUNS_DIR / f"hybrid_k{topk}_a{alpha_tag}.trec",
                TRACE_DIR / f"hybrid_k{topk}_a{alpha_tag}.txt",
                run_name=f"hybrid_k{topk}_a{alpha_tag}",
            )

    print("Completed FiQA experiments.")
    print(f"- Runs: {RUNS_DIR}")
    print(f"- Traces: {TRACE_DIR}")
    print(f"- Setup: {RESULTS_DIR / 'experiment_setup.json'}")


if __name__ == "__main__":
    main()
