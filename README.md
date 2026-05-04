# BIS Standards Recommendation Engine

AI-powered RAG pipeline that recommends relevant **Bureau of Indian Standards (BIS)** for any building material product description — in seconds.

🏆 Built for **BIS × SS 2026 Hackathon**
🎯 Track: AI / Retrieval Augmented Generation (RAG)

---

## 🚀 What it does

1. Input a product description
   *(e.g., "43 grade OPC cement for residential construction")*

2. Hybrid retrieval finds the most relevant IS standards from BIS SP 21

3. LLM generates a concise compliance rationale

---

## ⚡ Key Features

* 🔍 Hybrid Retrieval (Semantic + Keyword)
* 🤖 LLM-powered rationale (Groq LLaMA 3.3)
* 🛡 Hallucination Guard (only valid IS numbers returned)
* ⚡ Ultra-fast (~0.07s latency)
* 📊 Evaluation-ready pipeline for judges

---

## 🏗 System Architecture

```
Product Description
      │
      ▼
SentenceTransformer (all-MiniLM-L6-v2)
      │
      ▼
ChromaDB Vector Store  ← BIS SP 21 chunks
      │
      ▼
Hallucination Filter
      │
      ▼
Groq LLM (llama-3.3-70b)
      │
      ▼
Top 3–5 IS Standards + Rationale
```

---

## 📁 Project Structure

```
bis-rag/
├── src/
│   ├── app.py
│   ├── retriever.py
│   ├── generator.py
│   ├── ingest.py
│   └── index.html
├── data/
│   ├── BIS_SP21.pdf
│   ├── chunks.json
│   └── sample_results.json
├── vectorstore/
├── inference.py
├── eval_script.py
├── requirements.txt
└── presentation.pdf
```

---

## ⚙️ Setup Instructions

### 1. Clone repository

```bash
git clone https://github.com/Sheeja2006/bis-rag.git
cd bis-rag
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Create environment file

```bash
cp .env.example .env
```

### 4. Add your Groq API key

Open `.env` and add:

```
GROQ_API_KEY=your_groq_api_key_here
```

### 5. Run the server

```bash
uvicorn src.app:app --reload
```

Open:
👉 http://localhost:8000

---

## 🧪 Running Inference (Judge Entry)

```bash
python inference.py --input hidden_private_dataset.json --output team_results.json
```

Output format:

```json
[
  {
    "id": "1",
    "retrieved_standards": ["IS 269", "IS 8112", "IS 12269"],
    "latency_seconds": 0.08
  }
]
```

---

## 📊 Running Evaluation

```bash
python eval_script.py --results data/sample_results.json --ground_truth data/sample_test.json
```

Metrics:

* Hit Rate @3
* MRR @5
* Average Latency

---

## 📈 Evaluation Results (Public Demo)

| Metric      | Score  | Note                                   |
| ----------- | ------ | -------------------------------------- |
| Hit Rate @3 | 100%   | All queries retrieved correct standard |
| MRR @5      | ~0.67  | Strong ranking performance             |
| Avg Latency | ~0.07s | 70× faster than target                 |

> ⚠ Evaluated on small public dataset (3 queries). Larger evaluation is future work.

---

## 🧠 Tech Stack

* Retriever: `sentence-transformers`, `ChromaDB`
* Generator: `Groq` (llama-3.3-70b-versatile)
* API: `FastAPI`
* Dataset: BIS SP 21 (Building Materials)

---

## 👥 Team

Add your team members here:

* Sheeja (Lead Developer)
* [Teammate 2]
* [Teammate 3]

---

## 🔐 Security Note

API keys are managed using environment variables (`.env`) and are **not stored in the repository** for security reasons.

---

## 💡 Future Improvements

* Larger evaluation dataset
* Better ranking optimization (BM25 + FAISS hybrid)
* UI enhancements with richer explainability
* Support for more BIS domains

---

## 🙌 Acknowledgements

* Bureau of Indian Standards (BIS SP 21 dataset)
* Groq API
* Hugging Face (sentence-transformers)
* FAISS / Vector DB ecosystem
