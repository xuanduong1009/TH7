# TH7

Re-ranking tai lieu bang LLM-QLM tren bo du lieu FiQA.

## Files

- `prepare_fiqa.py`: tai va chuan hoa du lieu FiQA
- `build_index.py`: build Lucene index cho BM25
- `retrieve_bm25.py`: chay baseline BM25
- `llm_qlm.py`: tinh diem Query Likelihood bang LLM
- `rerank_qlm.py`: re-rank bang QLM va Hybrid
- `run_experiments.py`: chay toan bo thuc nghiem
- `evaluate_runs.py`: danh gia ket qua

## Run

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python src\prepare_fiqa.py
python src\build_index.py
python src\run_experiments.py --model_name distilgpt2 --batch_size 4
python src\evaluate_runs.py
```

## Reports

- `HUONG_DAN_SU_DUNG_VA_BAO_CAO.md`
- `TODO_NGAY_MAI.md`
- `BAO_CAO_TH7_MAU.md`
