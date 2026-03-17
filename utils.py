import json
import os
import re
import shutil
from pathlib import Path
from typing import Dict, Iterable, Mapping

from config import CORPUS_PATH, METADATA_PATH, QRELS_PATH, QUERIES_PATH


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def save_json(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(obj, file, ensure_ascii=False, indent=2)


def load_json(path):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def load_metadata(path=METADATA_PATH):
    path = Path(path)
    if not path.exists():
        return {}
    return load_json(path)


def ensure_exists(path: Path, description: str) -> None:
    if not Path(path).exists():
        raise FileNotFoundError(
            f"Missing {description}: {path}. Run the prerequisite script first."
        )


def discover_java_home() -> Path | None:
    java_home = os.environ.get("JAVA_HOME")
    if java_home and Path(java_home).exists():
        return Path(java_home)

    candidate_dirs: list[Path] = []
    for base_dir in (
        Path("C:/Program Files/Eclipse Adoptium"),
        Path("C:/Program Files/Java"),
        Path("C:/Program Files (x86)/Java"),
    ):
        if base_dir.exists():
            candidate_dirs.extend(
                candidate for candidate in base_dir.iterdir() if candidate.is_dir()
            )

    if candidate_dirs:
        def _java_sort_key(path: Path):
            name = path.name.lower()
            preferred_vendor = 1 if "temurin" in str(path).lower() or "jdk" in name else 0
            version_parts = [int(part) for part in re.findall(r"\d+", name)]
            return (preferred_vendor, version_parts, name)

        return sorted(candidate_dirs, key=_java_sort_key, reverse=True)[0]

    java_executable = shutil.which("java")
    if java_executable:
        java_path = Path(java_executable).resolve()
        if java_path.parent.name.lower() == "bin":
            return java_path.parent.parent

    return None


def configure_java_environment() -> Path | None:
    java_home = discover_java_home()
    if java_home is None:
        return None

    os.environ["JAVA_HOME"] = str(java_home)
    java_bin = str(java_home / "bin")
    current_path = os.environ.get("PATH", "")
    path_entries = current_path.split(os.pathsep) if current_path else []
    if java_bin not in path_entries:
        os.environ["PATH"] = java_bin + os.pathsep + current_path

    return java_home


def load_queries(path=QUERIES_PATH) -> Dict[str, str]:
    ensure_exists(Path(path), "prepared FiQA queries")
    queries = {}
    with open(path, "r", encoding="utf-8") as file:
        for raw_line in file:
            line = raw_line.rstrip("\n")
            if not line:
                continue
            query_id, text = line.split("\t", 1)
            queries[query_id] = text
    return queries


def load_corpus(path=CORPUS_PATH) -> Dict[str, str]:
    ensure_exists(Path(path), "prepared FiQA corpus")
    corpus = {}
    with open(path, "r", encoding="utf-8") as file:
        for line in file:
            record = json.loads(line)
            corpus[record["id"]] = record["contents"]
    return corpus


def load_qrels(path=QRELS_PATH) -> Dict[str, Dict[str, int]]:
    ensure_exists(Path(path), "prepared FiQA qrels")
    qrels: Dict[str, Dict[str, int]] = {}
    with open(path, "r", encoding="utf-8") as file:
        for raw_line in file:
            line = raw_line.strip()
            if not line or line.startswith("query_id\t"):
                continue
            query_id, _, doc_id, relevance = line.split("\t")
            qrels.setdefault(query_id, {})[doc_id] = int(relevance)
    return qrels


def sort_doc_scores(doc_scores: Mapping[str, float]):
    return sorted(doc_scores.items(), key=lambda item: (-item[1], item[0]))


def trim_run(run_dict: Mapping[str, Mapping[str, float]], topk: int):
    trimmed = {}
    for query_id, doc_scores in run_dict.items():
        trimmed[query_id] = dict(sort_doc_scores(doc_scores)[:topk])
    return trimmed


def write_trec_run(run_dict: Dict[str, Dict[str, float]], output_path, run_name: str = "run"):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as file:
        for query_id in sorted(run_dict):
            ranked = sort_doc_scores(run_dict[query_id])
            for rank, (doc_id, score) in enumerate(ranked, start=1):
                file.write(f"{query_id} Q0 {doc_id} {rank} {score:.6f} {run_name}\n")


def write_trace(run_dict: Dict[str, Dict[str, float]], output_path, include_header: bool = False):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as file:
        if include_header:
            file.write("query_id\tdoc_id\tscore\trank\n")
        for query_id in sorted(run_dict):
            ranked = sort_doc_scores(run_dict[query_id])
            for rank, (doc_id, score) in enumerate(ranked, start=1):
                file.write(f"{query_id}\t{doc_id}\t{score:.6f}\t{rank}\n")


def alpha_to_tag(alpha: float) -> str:
    alpha_text = f"{alpha:.3f}".rstrip("0").rstrip(".")
    return alpha_text.replace(".", "")


def tag_to_alpha(alpha_tag: str) -> float:
    if len(alpha_tag) == 1:
        return float(alpha_tag)
    return float(f"{alpha_tag[0]}.{alpha_tag[1:]}")


def normalize_scores(scores: Mapping[str, float], method: str = "minmax") -> Dict[str, float]:
    if method in (None, "", "none"):
        return dict(scores)

    values = list(scores.values())
    if not values:
        return {}

    if method != "minmax":
        raise ValueError(f"Unsupported normalization method: {method}")

    min_value = min(values)
    max_value = max(values)
    if min_value == max_value:
        return {doc_id: 1.0 for doc_id in scores}

    return {
        doc_id: (score - min_value) / (max_value - min_value)
        for doc_id, score in scores.items()
    }


def build_hybrid_scores(
    bm25_scores: Mapping[str, float],
    qlm_scores: Mapping[str, float],
    alpha: float,
    normalization: str = "minmax",
) -> Dict[str, float]:
    bm25_component = normalize_scores(bm25_scores, normalization)
    qlm_component = normalize_scores(qlm_scores, normalization)

    return {
        doc_id: alpha * bm25_component[doc_id] + (1.0 - alpha) * qlm_component[doc_id]
        for doc_id in bm25_scores
    }


def save_run_bundle(
    run_dict: Dict[str, Dict[str, float]],
    json_path,
    trec_path,
    trace_path,
    run_name: str,
    trace_header: bool = False,
) -> None:
    save_json(run_dict, json_path)
    write_trec_run(run_dict, trec_path, run_name)
    write_trace(run_dict, trace_path, include_header=trace_header)


def iter_assignment_rows() -> Iterable[tuple[str, int | None, float | None]]:
    yield ("BM25", None, None)
    for topk in (10, 20, 50):
        yield ("LLM-QLM", topk, None)
    yield ("Hybrid", 10, 0.2)
    yield ("Hybrid", 20, 0.5)
    yield ("Hybrid", 50, 0.8)
