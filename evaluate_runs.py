import re
import shutil
import subprocess

import pandas as pd
from ranx import Qrels, Run, evaluate

from config import QRELS_TREC_PATH, RESULTS_DIR, RUNS_DIR
from utils import (
    iter_assignment_rows,
    load_json,
    load_metadata,
    load_qrels,
    save_json,
    tag_to_alpha,
)


def qrels_from_local_file():
    return Qrels(load_qrels())


def parse_run_name(run_file_name: str):
    stem = run_file_name.rsplit(".", 1)[0]

    if stem.startswith("bm25_"):
        match = re.search(r"top(\d+)", stem)
        return {
            "method": "BM25",
            "k": None,
            "alpha": None,
            "run_depth": int(match.group(1)) if match else None,
        }

    match = re.fullmatch(r"qlm_k(\d+)", stem)
    if match:
        return {"method": "LLM-QLM", "k": int(match.group(1)), "alpha": None}

    match = re.fullmatch(r"hybrid_k(\d+)_a(\d+)", stem)
    if match:
        return {
            "method": "Hybrid",
            "k": int(match.group(1)),
            "alpha": tag_to_alpha(match.group(2)),
        }

    return {"method": stem, "k": None, "alpha": None}


def evaluate_with_trec_eval(run_path):
    if shutil.which("trec_eval") is None or not QRELS_TREC_PATH.exists():
        return {}

    cmd = [
        "trec_eval",
        "-m",
        "ndcg_cut.10",
        "-m",
        "recall.100",
        str(QRELS_TREC_PATH),
        str(run_path),
    ]
    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return {"trec_eval_error": result.stderr.strip() or result.stdout.strip()}

    metrics = {}
    for line in result.stdout.splitlines():
        parts = line.split()
        if len(parts) == 3:
            metric_name, _, value = parts
            if metric_name == "ndcg_cut_10":
                metrics["trec_eval_ndcg@10"] = float(value)
            elif metric_name == "recall_100":
                metrics["trec_eval_recall@100"] = float(value)
    return metrics


def build_assignment_table(all_metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for method, topk, alpha in iter_assignment_rows():
        method_mask = all_metrics["method"] == method
        k_mask = (
            all_metrics["k"].isna()
            if topk is None
            else all_metrics["k"].fillna(-1).astype(int) == int(topk)
        )
        alpha_mask = (
            all_metrics["alpha"].isna()
            if alpha is None
            else all_metrics["alpha"].round(4) == round(alpha, 4)
        )
        subset = all_metrics[method_mask & k_mask & alpha_mask]
        if subset.empty:
            rows.append(
                {
                    "Method": method,
                    "k": "-" if topk is None else topk,
                    "alpha": "-" if alpha is None else alpha,
                    "nDCG@10": None,
                    "Recall@100": None,
                }
            )
            continue

        best_row = subset.sort_values(by="ndcg@10", ascending=False).iloc[0]
        rows.append(
            {
                "Method": method,
                "k": "-" if topk is None else topk,
                "alpha": "-" if alpha is None else alpha,
                "nDCG@10": best_row["ndcg@10"],
                "Recall@100": best_row["recall@100"],
            }
        )

    return pd.DataFrame(rows)


def write_report_markdown(assignment_table: pd.DataFrame):
    metadata = load_metadata()
    report_path = RESULTS_DIR / "report_summary.md"
    lines = [
        "# Bai Thuc Hanh 7 - FiQA",
        "",
        "## Dataset",
        f"- Documents: {metadata.get('documents', 'N/A')}",
        f"- Queries: {metadata.get('queries', 'N/A')}",
        f"- Relevance judgments: {metadata.get('relevance_judgments', 'N/A')}",
        "",
        "## Ket qua chinh",
        "| Method | k | alpha | nDCG@10 | Recall@100 |",
        "| --- | --- | --- | --- | --- |",
    ]

    for _, row in assignment_table.iterrows():
        ndcg = "" if pd.isna(row["nDCG@10"]) else f"{row['nDCG@10']:.4f}"
        recall = "" if pd.isna(row["Recall@100"]) else f"{row['Recall@100']:.4f}"
        lines.append(
            f"| {row['Method']} | {row['k']} | {row['alpha']} | {ndcg} | {recall} |"
        )

    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


def main():
    qrels = qrels_from_local_file()
    rows = []

    for run_file in sorted(RUNS_DIR.glob("*.json")):
        run = Run(load_json(run_file))
        metrics = evaluate(qrels, run, metrics=["ndcg@10", "recall@100"])
        row = {"run_file": run_file.name}
        row.update(parse_run_name(run_file.name))
        row.update(metrics)

        trec_metrics = evaluate_with_trec_eval(RUNS_DIR / f"{run_file.stem}.trec")
        row.update(trec_metrics)
        rows.append(row)

    if not rows:
        raise FileNotFoundError(
            f"No run files found in {RUNS_DIR}. Run `python src/run_experiments.py` first."
        )

    metrics_df = pd.DataFrame(rows).sort_values(
        by=["method", "k", "alpha", "ndcg@10"],
        ascending=[True, True, True, False],
    )
    metrics_path = RESULTS_DIR / "metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)

    assignment_table = build_assignment_table(metrics_df)
    assignment_table_path = RESULTS_DIR / "assignment_table.csv"
    assignment_table.to_csv(assignment_table_path, index=False)

    summary_payload = {
        "dataset": load_metadata(),
        "metrics_file": str(metrics_path),
        "assignment_table_file": str(assignment_table_path),
    }
    save_json(summary_payload, RESULTS_DIR / "evaluation_summary.json")
    report_path = write_report_markdown(assignment_table)

    print(metrics_df.to_string(index=False))
    print(f"Saved metrics to {metrics_path}")
    print(f"Saved assignment table to {assignment_table_path}")
    print(f"Saved report summary to {report_path}")


if __name__ == "__main__":
    main()
