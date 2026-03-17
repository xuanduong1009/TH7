# TODO Ngay Mai - Bai Thuc Hanh 7

## Nhung gi da xong

- Code da hoan thien cho:
  - `prepare_fiqa.py`
  - `build_index.py`
  - `retrieve_bm25.py`
  - `llm_qlm.py`
  - `rerank_qlm.py`
  - `run_experiments.py`
  - `evaluate_runs.py`
- Da tao wrapper `src/` dung lenh de bai.
- Da cai xong `.venv` va dependencies.
- Da tai xong FiQA.
- Da build xong Lucene index.
- Da chay xong BM25 baseline.
- Da co:
  - `runs/bm25_top100.json`
  - `runs/bm25_top100.trec`
  - `trace/bm25.txt`
  - `results/experiment_setup.json`

## Nhung gi chua xong

### 1. Chua chay xong LLM-QLM

Can sinh them:
- `runs/qlm_k10.json`
- `runs/qlm_k20.json`
- `runs/qlm_k50.json`
- `runs/qlm_k10.trec`
- `runs/qlm_k20.trec`
- `runs/qlm_k50.trec`
- `trace/qlm_k10.txt`
- `trace/qlm_k20.txt`
- `trace/qlm_k50.txt`

### 2. Chua chay xong Hybrid

Can sinh them:
- `runs/hybrid_k10_a02.json`
- `runs/hybrid_k20_a05.json`
- `runs/hybrid_k50_a08.json`
- `runs/hybrid_k10_a02.trec`
- `runs/hybrid_k20_a05.trec`
- `runs/hybrid_k50_a08.trec`
- `trace/hybrid_k10_a02.txt`
- `trace/hybrid_k20_a05.txt`
- `trace/hybrid_k50_a08.txt`

### 3. Chua danh gia ket qua

Can chay:

```powershell
python src\evaluate_runs.py
```

Sau do moi co:
- `results/metrics.csv`
- `results/assignment_table.csv`
- `results/evaluation_summary.json`
- `results/report_summary.md`

### 4. Chua dien bang ket qua vao bao cao

Bang nay chua co so:
- BM25
- LLM-QLM k=10
- LLM-QLM k=20
- LLM-QLM k=50
- Hybrid k=10 alpha=0.2
- Hybrid k=20 alpha=0.5
- Hybrid k=50 alpha=0.8

### 5. Chua viet phan phan tich cuoi bai

Can viet sau khi co `results/assignment_table.csv`:
- LLM-QLM co cai thien so voi BM25 khong
- k nao tot nhat
- Hybrid co tot hon khong

### 6. Chua ve bieu do

Neu can them:
- `nDCG@10 vs k`
- So sanh BM25 va LLM-QLM

## Viec can lam ngay mai theo thu tu

1. Mo project:

```powershell
cd C:\Users\ADMIN\Downloads\TH7
.venv\Scripts\activate
```

2. Chay full experiment:

```powershell
python src\run_experiments.py --model_name distilgpt2 --batch_size 4
```

3. Danh gia:

```powershell
python src\evaluate_runs.py
```

4. Mo file:
- `results/assignment_table.csv`
- `results/report_summary.md`
- `BAO_CAO_TH7_MAU.md`

5. Chep so lieu vao bao cao va nop.

## So lieu da biet san de dien

- Documents: `57638`
- Queries: `648`
- Relevance judgments: `1706`
- Model: `distilgpt2`
- Size: `81912576` parameters
- Framework: `transformers`

