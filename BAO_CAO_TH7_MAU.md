# BAI THUC HANH 7
## Re-ranking tai lieu bang LLM-QLM tren bo du lieu FiQA

Ho va ten: ...

MSSV: ...

### 1. Gioi thieu

Trong he thong truy hoi thong tin, nhiem vu chinh la tim va xep hang cac tai lieu lien quan den truy van cua nguoi dung.
CAC he thong truyen thong thuong su dung cac mo hinh lexical nhu BM25 de truy hoi tai lieu dua tren su trung khop tu khoa.
Bai thuc hanh nay cai dat lai y tuong tu bai bao:
"Open-source Large Language Models are Strong Zero-shot Query Likelihood Models for Document Ranking (EMNLP Findings 2023)".
Phuong phap su dung Large Language Model (LLM) de uoc luong xac suat truy van duoc sinh ra tu tai lieu theo tu tuong Query Likelihood Model.

Muc tieu:
- Xay dung baseline BM25
- Cai dat LLM-QLM re-ranking
- So sanh BM25, LLM-QLM va Hybrid

### 2. Bo du lieu (FiQA)

FiQA la dataset truy hoi thong tin trong linh vuc tai chinh gom cac cau hoi va tai lieu lien quan.

Sinh vien tim hieu dataset va dien so luong vao bang:

- Documents: `57638`
- Queries: `648`
- Relevance judgments: `1706`

Vi du:
- Query: What is a good strategy for long term investing?
- Document: Long term investors often diversify their portfolio across different asset classes.
- Relevance: Relevant

### 3. Phuong phap

#### 3.1 Baseline: BM25

Query -> BM25 -> Top-k documents

Trong bai nay, BM25 duoc dung de truy hoi cac tai lieu ung vien tu Lucene index.

#### 3.2 LLM-QLM Re-ranking

Score(q,d) = log P(q | d)

Trong trien khai, moi tai lieu duoc dua vao prompt:

```text
Document: <document_text>
Query:
```

Sau do model `distilgpt2` duoc dung de tinh tong log-probability cua truy van dieu kien theo tai lieu.

#### 3.3 Hybrid Ranking

Score(q,d) = alpha * BM25_score + (1 - alpha) * QLM_score

Thu nghiem cac gia tri alpha = 0.2, 0.5, 0.8.

Trong code, diem BM25 va QLM duoc chuan hoa min-max theo tung query truoc khi ket hop.

### 4. Thiet lap thuc nghiem

Phuong phap so sanh:
- BM25
- LLM-QLM
- Hybrid

Tham so:
- Top-k documents: k = 10, 20, 50
- Tham so alpha: 0.2, 0.5, 0.8

Mo hinh LLM su dung:
- Model: `distilgpt2`
- Size: `81912576` parameters
- Framework: `transformers`

Thong so BM25:
- k1 = `0.9`
- b = `0.4`

### 5. Chi so danh gia

- nDCG@10: danh gia chat luong xep hang trong top-10.
- Recall@100: danh gia kha nang truy hoi tai lieu lien quan.

Thu vien:
- ranx
- trec_eval

### 6. Ket qua thuc nghiem

Sinh vien trinh bay bang ket qua:

| Method | k | alpha | nDCG@10 |
| --- | --- | --- | --- |
| BM25 | - | - | ... |
| LLM-QLM | 10 | - | ... |
| LLM-QLM | 20 | - | ... |
| LLM-QLM | 50 | - | ... |
| Hybrid | 10 | 0.2 | ... |
| Hybrid | 20 | 0.5 | ... |
| Hybrid | 50 | 0.8 | ... |

Neu muon them Recall@100 co the bo sung bang phu hoac mo ta ngan sau bang.

Sinh vien co the ve bieu do:
- nDCG@10 vs k
- Comparison giua BM25 va LLM-QLM

### 7. Phan tich ket qua

Phan tich:
- LLM-QLM co cai thien so voi BM25 khong
- Gia tri k tot nhat
- Hybrid co hieu qua hon khong

Vi du truy van:
- Query: How to diversify investments?

Goi y viet:

`[Dien sau khi co ket qua that tu assignment_table.csv]`

### 8. Trace

Sinh vien luu cac file:

```text
trace/
   bm25.txt
   qlm_k10.txt
   qlm_k20.txt
   qlm_k50.txt
   hybrid_k10_a02.txt
   hybrid_k20_a05.txt
   hybrid_k50_a08.txt
```

Moi file gom:
- query_id
- doc_id
- score
- rank

Tinh trang hien tai:
- Da co: `bm25.txt`
- Chua co: `qlm_k10.txt`, `qlm_k20.txt`, `qlm_k50.txt`, `hybrid_k10_a02.txt`, `hybrid_k20_a05.txt`, `hybrid_k50_a08.txt`

### 10. Tai lieu tham khao

Zhuang, S., Liu, B., Koopman, B., & Zuccon, G. (2023). Open-source Large Language Models are Strong Zero-shot Query Likelihood Models for Document Ranking. Findings of EMNLP.

---

## Huong dan chay dung y de bai

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python src\prepare_fiqa.py
python src\build_index.py
python src\run_experiments.py --model_name distilgpt2
python src\evaluate_runs.py
```

Trong may hien tai, phan `prepare_fiqa.py`, `build_index.py`, `retrieve_bm25.py` da chay xong.
Ngay mai chi can chay tiep:

```powershell
.venv\Scripts\activate
python src\run_experiments.py --model_name distilgpt2 --batch_size 4
python src\evaluate_runs.py
```
