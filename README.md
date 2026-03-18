# TH7

## Re-ranking tai lieu bang LLM-QLM tren bo du lieu FiQA

Repo nay chua ma nguon, ket qua thuc nghiem va file bao cao Word cho Bai thuc hanh 7.

## 1. Cau truc chinh

- `prepare_fiqa.py`: tai va chuan hoa du lieu FiQA.
- `build_index.py`: xay dung Lucene index cho BM25 bang Pyserini.
- `retrieve_bm25.py`: chay baseline BM25.
- `llm_qlm.py`: tinh diem Query Likelihood bang mo hinh ngon ngu.
- `rerank_qlm.py`: re-rank bang LLM-QLM va Hybrid.
- `run_experiments.py`: chay toan bo thuc nghiem.
- `evaluate_runs.py`: danh gia ket qua bang `ranx`, dong thoi ho tro `trec_eval` neu may co san.
- `src/`: wrapper de chay dung theo dinh dang lenh `python src/...`.
- `BaiThucHanh7.docx`: bao cao hoan chinh de nop.

## 2. Yeu cau moi truong

### Python

Khuyen nghi dung Python 3.10+.

### Java

Pyserini can Java de build index va truy hoi BM25. Khuyen nghi dung JDK 21.

Kiem tra:

```powershell
java -version
```

Neu may chua co Java, co the cai bang:

```powershell
winget install --id EclipseAdoptium.Temurin.21.JDK --accept-package-agreements --accept-source-agreements --silent
```

Neu can gan thu cong `JAVA_HOME` trong phien PowerShell hien tai:

```powershell
$env:JAVA_HOME = (Get-ChildItem 'C:\Program Files\Eclipse Adoptium' -Directory | Sort-Object Name -Descending | Select-Object -First 1).FullName
$env:PATH = "$env:JAVA_HOME\bin;$env:PATH"
java -version
```

## 3. Cai dat moi truong

Tai thu muc project:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m ensurepip --upgrade
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Hoac neu muon kich hoat moi truong:

```powershell
.\.venv\Scripts\activate
pip install -r requirements.txt
```

## 4. Cac buoc chay dung theo de bai

### Buoc 1. Tai du lieu FiQA

```powershell
python src\prepare_fiqa.py
```

Ket qua sinh ra trong `data/fiqa/`:
- `corpus.jsonl`
- `queries.tsv`
- `qrels.tsv`
- `qrels.trec`
- `metadata.json`

### Buoc 2. Build index BM25

```powershell
python src\build_index.py --threads 4 --overwrite
```

Index duoc luu tai:
- `indexes/fiqa/`

### Buoc 3. Chay baseline BM25

```powershell
python src\retrieve_bm25.py --topk 100
```

Ket qua duoc luu tai:
- `runs/bm25_top100.json`
- `runs/bm25_top100.trec`
- `trace/bm25.txt`

### Buoc 4. Chay toan bo thuc nghiem

```powershell
python src\run_experiments.py --model_name distilgpt2 --batch_size 4
```

Script nay se:
- Chay BM25 baseline top-100
- Re-rank voi `k = 10, 20, 50`
- Tinh Hybrid voi `alpha = 0.2, 0.5, 0.8`
- Luu run files va trace files

### Buoc 5. Danh gia ket qua

```powershell
python src\evaluate_runs.py
```

Ket qua danh gia duoc luu tai:
- `results/metrics.csv`
- `results/assignment_table.csv`
- `results/evaluation_summary.json`
- `results/report_summary.md`

## 5. Bo lenh chay day du tu dau

```powershell
cd C:\Users\ADMIN\Downloads\TH7

python -m venv .venv
.\.venv\Scripts\python.exe -m ensurepip --upgrade
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt

winget install --id EclipseAdoptium.Temurin.21.JDK --accept-package-agreements --accept-source-agreements --silent

$env:JAVA_HOME = (Get-ChildItem 'C:\Program Files\Eclipse Adoptium' -Directory | Sort-Object Name -Descending | Select-Object -First 1).FullName
$env:PATH = "$env:JAVA_HOME\bin;$env:PATH"

python src\prepare_fiqa.py
python src\build_index.py --threads 4 --overwrite
python src\retrieve_bm25.py --topk 100
python src\run_experiments.py --model_name distilgpt2 --batch_size 4
python src\evaluate_runs.py
```

## 6. Neu chi muon chay tiep tu phan con do

Neu du lieu, index va BM25 da co san, chi can:

```powershell
cd C:\Users\ADMIN\Downloads\TH7

$env:JAVA_HOME = (Get-ChildItem 'C:\Program Files\Eclipse Adoptium' -Directory | Sort-Object Name -Descending | Select-Object -First 1).FullName
$env:PATH = "$env:JAVA_HOME\bin;$env:PATH"

.\.venv\Scripts\python.exe src\run_experiments.py --model_name distilgpt2 --batch_size 4
.\.venv\Scripts\python.exe src\evaluate_runs.py
```

## 7. Thong tin thuc nghiem hien tai

Thong tin dataset:
- Documents: `57638`
- Queries: `648`
- Relevance judgments: `1706`

Thong tin model:
- Model: `distilgpt2`
- Size: `81,912,576` parameters
- Framework: `transformers`
- Device: `cpu`
- Batch size: `4`

Thong tin BM25:
- `k1 = 0.9`
- `b = 0.4`

## 8. Cac file dau ra quan trong

### Trace bat buoc

Trong `trace/` can co:
- `bm25.txt`
- `qlm_k10.txt`
- `qlm_k20.txt`
- `qlm_k50.txt`
- `hybrid_k10_a02.txt`
- `hybrid_k20_a05.txt`
- `hybrid_k50_a08.txt`

### Ket qua danh gia

Trong `results/`:
- `metrics.csv`
- `assignment_table.csv`
- `evaluation_summary.json`
- `report_summary.md`
- `experiment_setup.json`

## 9. Ket qua chinh hien tai

Theo `results/assignment_table.csv`:

| Method | k | alpha | nDCG@10 | Recall@100 |
| --- | --- | --- | --- | --- |
| BM25 | - | - | 0.2361 | 0.5395 |
| LLM-QLM | 10 | - | 0.2469 | 0.2951 |
| LLM-QLM | 20 | - | 0.2542 | 0.3712 |
| LLM-QLM | 50 | - | 0.2553 | 0.4599 |
| Hybrid | 10 | 0.2 | 0.2554 | 0.2951 |
| Hybrid | 20 | 0.5 | 0.2710 | 0.3712 |
| Hybrid | 50 | 0.8 | 0.2481 | 0.4599 |

Neu xet toan bo cac cau hinh da chay trong `metrics.csv`, cau hinh tot nhat la:
- `Hybrid (k = 50, alpha = 0.2)` voi `nDCG@10 = 0.2779`

## 10. Goi y nop bai

De nop bai, mo file:

- `BaiThucHanh7.docx`

Sau do chi can dien them:
- Ho va ten
- MSSV

Phan con lai cua bao cao da co san ket qua thuc nghiem, phan tich va mo ta trace.
