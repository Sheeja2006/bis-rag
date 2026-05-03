import os
import re
from groq import Groq

# ── API key — must be set via environment variable ────────────────────────────
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise EnvironmentError(
        "GROQ_API_KEY environment variable is not set.\n"
        "Run:  export GROQ_API_KEY=your_key_here"
    )

client = Groq(api_key=GROQ_API_KEY)


def normalise_is_id(raw: str) -> str:
    """
    Normalise any IS-number variant to canonical form: 'IS 269'.
    Handles: 'IS269', 'IS:269', 'IS-269', 'is 269', 'IS:  269', etc.
    """
    raw = raw.strip()
    # Remove everything between IS and the digits, keep the digits + suffix
    normalised = re.sub(r'(?i)IS[\s:\-]*(\d[\w\-]*)', r'IS \1', raw)
    return normalised.strip().upper()


def build_allowed_ids(retrieved_chunks):
    """
    Build a set of all normalised IS IDs from retrieved chunks.
    Ensures the hallucination guard and LLM output use the same format.
    """
    return {normalise_is_id(c["standard_id"]) for c in retrieved_chunks}


def extract_valid_is_numbers(text, allowed_ids):
    """
    Pull every IS XXXX mention from text.
    Normalise each found ID and keep only ones in allowed_ids.
    """
    found = re.findall(r'IS[\s:\-]*\d[\w\-]*', text, re.IGNORECASE)
    valid = []
    seen = set()
    for f in found:
        norm = normalise_is_id(f)
        if norm in allowed_ids and norm not in seen:
            seen.add(norm)
            valid.append(norm)
    return valid


def generate_recommendation(query, retrieved_chunks):
    # Use normalised IDs throughout — fixes guard vs chunk mismatch
    allowed_ids = build_allowed_ids(retrieved_chunks)

    # Increased context from 300 → 700 chars for richer LLM understanding
    context = "\n\n".join(
        f"[{normalise_is_id(c['standard_id'])}] {c['title']}\n{c['text'][:700]}"
        for c in retrieved_chunks
    )

    prompt = f"""You are a BIS compliance expert. Product: "{query}"

From the context below, pick the 3 most relevant IS standards.
STRICT RULES:
- Only use IS numbers explicitly listed in the context below
- One line per standard, reason in 10 words or fewer
- If fewer than 3 clearly apply, list only those that do
- Do NOT invent or guess IS numbers not in the context
- Do NOT add commentary, caveats, or extra text

Context:
{context}

Output format (strictly follow this):
1. IS XXXX – short reason
2. IS XXXX – short reason
3. IS XXXX – short reason"""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=200,
        temperature=0.0
    )

    raw_text = response.choices[0].message.content

    # ── Hallucination guard ──────────────────────────────────────────────────
    # Parse each line, normalise the IS number, and keep only verified ones.
    lines = raw_text.strip().split("\n")
    clean_lines = []
    for line in lines:
        match = re.search(r'(IS[\s:\-]*\d[\w\-]*)', line, re.IGNORECASE)
        if match:
            norm_id = normalise_is_id(match.group(1))
            if norm_id in allowed_ids:
                # Replace whatever format the LLM used with the canonical form
                clean_line = re.sub(
                    r'IS[\s:\-]*\d[\w\-]*', norm_id, line, count=1, flags=re.IGNORECASE
                )
                clean_lines.append(clean_line.strip())
            # Lines with unrecognised IS numbers are silently dropped

    # Fallback: if guard removed everything, list retrieved standards directly
    if not clean_lines:
        clean_lines = [
            f"{normalise_is_id(c['standard_id'])} – covers {c['title'][:50]}"
            for c in retrieved_chunks[:3]
        ]

    # Re-number surviving lines sequentially (fixes gap bug: "1. … 3. …")
    clean_response = "\n".join(
        f"{i + 1}. {re.sub(r'^\d+\.\s*', '', line)}"
        for i, line in enumerate(clean_lines)
    )

    return clean_response


if __name__ == "__main__":
    from retriever import retrieve
    query = "ordinary Portland cement for construction"
    chunks = retrieve(query, top_k=10)
    print("Retrieved:", [c["standard_id"] for c in chunks])
    print("\nGenerating recommendation...\n")
    response = generate_recommendation(query, chunks)
    print(response)