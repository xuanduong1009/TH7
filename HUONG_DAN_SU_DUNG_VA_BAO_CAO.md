# Bai Thuc Hanh 7 - Huong Dan Su Dung Va Bao Cao

## 1. Tinh trang hien tai

Da xong:
- Da sua xong code cho cac file `prepare_fiqa.py`, `build_index.py`, `retrieve_bm25.py`, `llm_qlm.py`, `rerank_qlm.py`, `run_experiments.py`, `evaluate_runs.py`.
- Da tao wrapper dung yeu cau de bai trong thu muc `src/`.
- Da cai xong dependencies trong `.venv`.
- Da tai xong du lieu FiQA.
- Da build xong Lucene index BM25.
- Da chay xong BM25 baseline va luu:
  - `runs/bm25_top100.json`
  - `runs/bm25_top100.trec`
  - `trace/bm25.txt`
- Da luu setup thuc nghiem ban dau:
  - `results/experiment_setup.json`

Chua xong:
- Chua chay xong LLM-QLM va Hybrid, vi ban da dung giua chung.
- Chua chay `evaluate_runs.py`, vi can co them cac run QLM/Hybrid.

## 2. So lieu FiQA de dien vao bao cao

- Documents: `57638`
- Queries: `648`
- Relevance judgments: `1706`

Luu y:
- Lucene chi index `57600` tai lieu vi co `38` document rong sau khi ghep title + text.
- Day la hanh vi binh thuong cua indexer, khong phai loi code.

## 3. Cach tiep tuc khi mo may lai

Mo terminal tai thu muc project:

```powershell
cd C:\Users\ADMIN\Downloads\TH7
```

Kich hoat moi truong:

```powershell
.venv\Scripts\activate
```

Neu chi muon chay lai BM25:

```powershell
python src\retrieve_bm25.py --topk 100
```

Neu muon chay full thuc nghiem:

```powershell
python src\run_experiments.py --model_name distilgpt2 --batch_size 4
```

Neu muon danh gia ket qua sau khi co day du run:

```powershell
python src\evaluate_runs.py
```

## 4. Thoi gian du kien

Voi may hien tai:
- BM25: nhanh, da xong.
- LLM-QLM voi `distilgpt2`, CPU, `batch_size=4`: du kien rat lau, khoang hon 1 gio.

Neu muon nhanh hon:
- Giu nguyen `distilgpt2` neu can bam de bai.
- Neu chi can demo pipeline, co the thu model nho hon, nhung ket qua se khong con dung setup ban dau.

## 5. Thu tu chay dung theo de bai

```powershell
python src\prepare_fiqa.py
python src\build_index.py --threads 4 --overwrite
python src\retrieve_bm25.py --topk 100
python src\run_experiments.py --model_name distilgpt2 --batch_size 4
python src\evaluate_runs.py
```

## 6. File output can nop

Sau khi chay xong full:

Trong `trace/` can co:
- `bm25.txt`
- `qlm_k10.txt`
- `qlm_k20.txt`
- `qlm_k50.txt`
- `hybrid_k10_a02.txt`
- `hybrid_k20_a05.txt`
- `hybrid_k50_a08.txt`

Trong `results/` se co:
- `metrics.csv`
- `assignment_table.csv`
- `evaluation_summary.json`
- `report_summary.md`

## 7. Mau viet bao cao

### 1. Gioi thieu

Trong bai thuc hanh nay, em thuc hien bai toan re-ranking tai lieu tren bo du lieu FiQA bang phuong phap LLM-QLM. Muc tieu la xay dung baseline BM25, tinh diem Query Likelihood bang Large Language Model, ket hop BM25 va QLM trong mo hinh Hybrid, sau do so sanh ket qua theo chi so nDCG@10 va Recall@100.

### 2. Bo du lieu

Bo du lieu su dung la `beir/fiqa/test` trong `ir_datasets`.

- Documents: `57638`
- Queries: `648`
- Relevance judgments: `1706`

FiQA la bo du lieu truy hoi thong tin trong linh vuc tai chinh, gom cac cau hoi cua nguoi dung va cac tai lieu lien quan.

### 3. Phuong phap

#### 3.1 BM25

He thong baseline su dung BM25 de truy hoi top-k tai lieu ung vien tu Lucene index.

#### 3.2 LLM-QLM

Diem cua moi cap truy van - tai lieu duoc tinh theo:

```text
Score(q, d) = log P(q | d)
```

Trong code, tai lieu duoc dua vao prompt dang:

```text
Document: <noi_dung_tai_lieu>
Query:
```

Sau do `distilgpt2` duoc dung de tinh tong log-probability cua chuoi query.

#### 3.3 Hybrid

He thong Hybrid ket hop diem BM25 va diem QLM:

```text
Score(q, d) = alpha * BM25_score + (1 - alpha) * QLM_score
```

Trong trien khai, diem BM25 va QLM duoc chuan hoa min-max theo tung query truoc khi cong de tranh lech scale.

### 4. Thiet lap thuc nghiem

- BM25 baseline: top-100 de danh gia Recall@100
- Re-ranking top-k voi `k = 10, 20, 50`
- Hybrid voi `alpha = 0.2, 0.5, 0.8`
- Model LLM: `distilgpt2`
- Framework: `transformers`

Phan thong tin `Size` co the lay sau khi chay xong full tu file:

```text
results/experiment_setup.json
```

### 5. Chi so danh gia

- `nDCG@10`
- `Recall@100`

Thu vien su dung:
- `ranx`
- `trec_eval` neu co cai trong may

### 6. Bang ket qua

Sau khi chay `python src\evaluate_runs.py`, lay file:

```text
results/assignment_table.csv
```

va dien vao bang:

| Method | k | alpha | nDCG@10 | Recall@100 |
| --- | --- | --- | --- | --- |
| BM25 | - | - | ... | ... |
| LLM-QLM | 10 | - | ... | ... |
| LLM-QLM | 20 | - | ... | ... |
| LLM-QLM | 50 | - | ... | ... |
| Hybrid | 10 | 0.2 | ... | ... |
| Hybrid | 20 | 0.5 | ... | ... |
| Hybrid | 50 | 0.8 | ... | ... |

### 7. Phan tich ket qua

Ban co the viet theo mau sau:

- LLM-QLM co the cai thien nDCG@10 so voi BM25 neu model khai thac duoc thong tin ngu nghia ma BM25 bo sot.
- Khi `k` tang, he thong co them ung vien de re-rank, nhung thoi gian tinh toan cung tang.
- Hybrid thuong on dinh hon vi ket hop duoc matching tu khoa cua BM25 va kha nang mo hinh hoa ngu nghia cua LLM.
- Gia tri `k` tot nhat la gia tri cho nDCG@10 cao nhat trong bang ket qua.
- Neu Hybrid cho ket qua cao hon ca BM25 va LLM-QLM, co the ket luan viec ket hop hai nguon diem mang lai hieu qua tot hon.

### 8. Trace

Mo ta ngan:

He thong luu trace cho moi truy van gom `query_id`, `doc_id`, `score`, `rank`. Cac file trace duoc dung de kiem tra thu tu xep hang va doi chieu ket qua giua BM25, QLM va Hybrid.

### 9. Tai lieu tham khao

Zhuang, S., Liu, B., Koopman, B., & Zuccon, G. (2023). Open-source Large Language Models are Strong Zero-shot Query Likelihood Models for Document Ranking. Findings of EMNLP.

