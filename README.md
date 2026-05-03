# BIS Standards Recommendation Engine

An AI-powered RAG pipeline that recommends relevant Bureau of Indian Standards (BIS) for any building material product description — in seconds.

Built for the **BIS × SS 2026 Hackathon** | Track: AI / Retrieval Augmented Generation

---

## What it does

1. You describe a building material product (e.g. "43 grade OPC cement for residential construction")
2. The engine retrieves the most relevant IS standards from BIS SP 21 using semantic search
3. An LLM generates a brief rationale for each recommended standard

---

## Project Structure

```
bis-rag/
├── src/
│   ├── app.py          # FastAPI server
│   ├── retriever.py    # ChromaDB semantic retriever
│   ├── generator.py    # Groq LLM rationale generator
│   ├── ingest.py       # PDF ingestion + chunking
│   └── index.html      # Frontend UI
├── data/
│   ├── BIS_SP21.pdf    # Source dataset
│   ├── chunks.json     # Pre-processed chunks
│   └── sample_results.json  # Public test set results
├── vectorstore/        # ChromaDB persistent store
├── inference.py        # Judge evaluation entry point
├── eval_script.py      # Evaluation metrics script
├── requirements.txt
└── presentation.pdf
```

---

## Setup

### 1. Clone and install dependencies

```bash
git clone <your-repo-url>
cd bis-rag
pip install -r requirements.txt
```

### 2. Set your Groq API key

```bash
export GROQ_API_KEY=your_key_here
```

> The vectorstore is pre-built and included. Skip to step 3.
> To rebuild from scratch: `python src/ingest.py`

### 3. Start the server

```bash
uvicorn src.app:app --reload
```

Open `http://localhost:8000` in your browser.

---

## Running Inference (for judges)

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

## Running Evaluation

```bash
python eval_script.py --results data/sample_results.json --ground_truth data/sample_test.json
```

Outputs: Hit Rate @3, MRR @5, Average Latency

---

## System Architecture

```
Product Description
      │
      ▼
  SentenceTransformer (all-MiniLM-L6-v2)
      │  semantic embedding
      ▼
  ChromaDB Vector Store  ←── BIS SP 21 PDF chunks
      │  top-k retrieval
      ▼
  Hallucination Filter   ←── only real IS numbers pass
      │
      ▼
  Groq LLM (llama-3.3-70b)
      │  rationale generation
      ▼
  Top 3–5 IS Standards + Rationale
```

---

## Evaluation Results (Public Test Set)

| Metric | Score | Target |
|---|---|---|
| Hit Rate @3 | — | >80% |
| MRR @5 | — | >0.7 |
| Avg Latency | ~0.07s | <5s |

---

## Tech Stack

- Retriever: `sentence-transformers` + `ChromaDB`
- Generator: `Groq` (llama-3.3-70b-versatile)
- API: `FastAPI`
- Dataset: BIS SP 21 (Building Materials)

---

## Team

Built during BIS × SS 2026 Hackathon.