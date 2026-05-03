from contextlib import asynccontextmanager
from pathlib import Path
import json
import re

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from src.retriever import retrieve, get_model, get_collection
from src.generator import generate_recommendation
from groq import Groq
import os

_INDEX_HTML = Path(__file__).parent / "index.html"
_CHUNKS_PATH = Path(__file__).parent.parent / "data" / "chunks.json"

# Load all chunks once for the /explain endpoint
_all_chunks = None
def get_all_chunks():
    global _all_chunks
    if _all_chunks is None:
        with open(_CHUNKS_PATH, encoding="utf-8") as f:
            _all_chunks = json.load(f)
    return _all_chunks


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Warming up model and vectorstore...")
    get_model()
    get_collection()
    get_all_chunks()
    print("Ready!")
    yield


app = FastAPI(title="BIS Standards Recommendation Engine", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class Query(BaseModel):
    product_description: str

class ExplainRequest(BaseModel):
    standard_id: str


def clean_title(raw: str, standard_id: str) -> str:
    t = raw.strip()
    t = re.sub(r'^\d[\d\s]*:\s*\d{4}\s*', '', t).strip()
    t = re.sub(r'\(.*?[Rr]evision.*?\)', '', t).strip()
    if len(t) < 6:
        return standard_id
    return t


# ── /retrieve — fast, no LLM ──────────────────────────────────────────────────
@app.post("/retrieve")
def retrieve_only(query: Query):
    try:
        chunks = retrieve(query.product_description, top_k=5)
        standards = [
            {"id": c["standard_id"], "title": clean_title(c.get("title", ""), c["standard_id"])}
            for c in chunks
        ]
        return {
            "query":         query.product_description,
            "top_standards": [s["id"] for s in standards],
            "standards":     standards,
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


# ── /rationale — LLM rationale for top 3 ─────────────────────────────────────
@app.post("/rationale")
def rationale_only(query: Query):
    try:
        chunks    = retrieve(query.product_description, top_k=10)
        rationale = generate_recommendation(query.product_description, chunks)
        return {"rationale": rationale}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


# ── /explain — LLM deep-dive on a single standard ────────────────────────────
@app.post("/explain")
def explain_standard(req: ExplainRequest):
    try:
        all_chunks = get_all_chunks()
        # Gather all chunks for this standard
        std_chunks = [c for c in all_chunks if c.get("standard_id") == req.standard_id]
        if not std_chunks:
            raise HTTPException(status_code=404, detail=f"No data found for {req.standard_id}")

        # Combine chunk texts (up to ~2000 chars)
        combined = "\n\n".join(c["text"] for c in std_chunks)[:2000]

        groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
        prompt = f"""You are a BIS compliance expert helping an Indian small business owner.

Explain the following BIS standard in simple, plain English. Structure your response as:

**What it covers:** (1-2 sentences)
**Who needs it:** (which products/industries must comply)
**Key requirements:** (3-4 bullet points of the most important specifications)
**Why it matters:** (1 sentence on the benefit to consumers/industry)

Standard: {req.standard_id}
Source text from BIS SP 21:
{combined}

Keep the explanation clear and practical. Avoid technical jargon where possible."""

        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=400,
            temperature=0.3
        )
        explanation = response.choices[0].message.content
        return {"standard_id": req.standard_id, "explanation": explanation}

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


# ── /recommend — original combined endpoint (for inference.py) ────────────────
@app.post("/recommend")
def recommend(query: Query):
    try:
        chunks    = retrieve(query.product_description, top_k=10)
        rationale = generate_recommendation(query.product_description, chunks)
        standards = [
            {"id": c["standard_id"], "title": clean_title(c.get("title", ""), c["standard_id"])}
            for c in chunks[:5]
        ]
        return {
            "query":         query.product_description,
            "top_standards": [s["id"] for s in standards],
            "standards":     standards,
            "rationale":     rationale,
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/", response_class=HTMLResponse)
def home():
    if not _INDEX_HTML.exists():
        raise HTTPException(status_code=404, detail="index.html not found")
    return _INDEX_HTML.read_text(encoding="utf-8")