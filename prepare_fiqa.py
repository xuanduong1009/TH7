import json

import ir_datasets

from config import (
    CORPUS_PATH,
    DATASET_NAME,
    METADATA_PATH,
    QRELS_PATH,
    QRELS_TREC_PATH,
    QUERIES_PATH,
)
from utils import normalize_whitespace, save_json


def main():
    dataset = ir_datasets.load(DATASET_NAME)

    metadata = {
        "dataset": DATASET_NAME,
        "documents": dataset.docs_count(),
        "queries": dataset.queries_count(),
        "relevance_judgments": dataset.qrels_count(),
    }

    with open(CORPUS_PATH, "w", encoding="utf-8") as file:
        for doc in dataset.docs_iter():
            title = normalize_whitespace(getattr(doc, "title", "") or "")
            text = normalize_whitespace(getattr(doc, "text", "") or "")
            contents = normalize_whitespace(f"{title} {text}")
            record = {
                "id": doc.doc_id,
                "title": title,
                "text": text,
                "contents": contents,
            }
            file.write(json.dumps(record, ensure_ascii=False) + "\n")

    with open(QUERIES_PATH, "w", encoding="utf-8") as file:
        for query in dataset.queries_iter():
            file.write(f"{query.query_id}\t{normalize_whitespace(query.text)}\n")

    with open(QRELS_PATH, "w", encoding="utf-8") as tsv_file, open(
        QRELS_TREC_PATH, "w", encoding="utf-8"
    ) as trec_file:
        tsv_file.write("query_id\tQ0\tdoc_id\trelevance\n")
        for qrel in dataset.qrels_iter():
            relevance = int(qrel.relevance)
            tsv_file.write(f"{qrel.query_id}\tQ0\t{qrel.doc_id}\t{relevance}\n")
            trec_file.write(f"{qrel.query_id} 0 {qrel.doc_id} {relevance}\n")

    save_json(metadata, METADATA_PATH)

    print("Saved FiQA dataset files:")
    print(f"- {CORPUS_PATH}")
    print(f"- {QUERIES_PATH}")
    print(f"- {QRELS_PATH}")
    print(f"- {QRELS_TREC_PATH}")
    print(f"- {METADATA_PATH}")
    print(
        "Dataset counts: "
        f"documents={metadata['documents']}, "
        f"queries={metadata['queries']}, "
        f"qrels={metadata['relevance_judgments']}"
    )


if __name__ == "__main__":
    main()
