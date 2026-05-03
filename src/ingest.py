import fitz  # pymupdf  — add to requirements: pymupdf>=1.23.0
import os
import json
import re
from pathlib import Path
from tqdm import tqdm

# ── Paths resolved relative to this file ─────────────────────────────────────
_SRC_DIR  = Path(__file__).parent
_ROOT_DIR = _SRC_DIR.parent
DEFAULT_PDF_PATH    = str(_ROOT_DIR / "data" / "BIS_SP21.pdf")
DEFAULT_CHUNKS_PATH = str(_ROOT_DIR / "data" / "chunks.json")

MAX_CHUNK_CHARS = 1500   # hard ceiling per chunk


def extract_text_from_pdf(pdf_path):
    doc   = fitz.open(pdf_path)
    pages = []
    for i, page in enumerate(doc):
        text = page.get_text()
        pages.append({"page": i + 1, "text": text})
    print(f"✅ Extracted {len(pages)} pages")
    return pages


def truncate_at_sentence(text, max_chars=MAX_CHUNK_CHARS):
    """
    Truncate text to at most max_chars, ending at the last sentence boundary
    (. ! ?) before the limit. Falls back to hard cut if no boundary found.
    """
    if len(text) <= max_chars:
        return text
    window = text[:max_chars]
    # Find last sentence-ending punctuation followed by whitespace or end
    match = re.search(r'[.!?](?=\s|$)', window[::-1])
    if match:
        cut = max_chars - match.start()
        return text[:cut].rstrip()
    return window.rstrip()


def extract_title(lines, std_num):
    """
    Robustly extract the standard's title from the lines that follow the IS number.
    Skips short part/amendment suffixes (e.g. 'Part 1', 'Amendment No. 2')
    and returns the first substantive line.
    """
    skip_patterns = re.compile(
        r'^(part\s*\d|section\s*\d|amendment|amd|erratum|reaffirmed|\d{4}$)',
        re.IGNORECASE
    )
    for line in lines[1:]:   # lines[0] is the IS number itself
        stripped = line.strip()
        if not stripped:
            continue
        if skip_patterns.match(stripped):
            continue
        if len(stripped) < 5:
            continue
        return stripped[:100]
    return std_num


def chunk_by_standard(pages):
    full_text = "\n".join(p["text"] for p in pages)

    # Split on BIS standard patterns: 'IS 269', 'IS: 269', 'IS:269'
    pattern  = r'(?=\bIS\s*:?\s*\d{2,5}|\bIS\s+\d{2,5})'
    sections = re.split(pattern, full_text)

    # Use a dict keyed by standard_id to deduplicate — keep the longest chunk
    # for each standard (more text = richer context for retrieval).
    chunk_map = {}

    for section in sections:
        section = section.strip()
        if len(section) < 50:
            continue

        match   = re.match(r'IS\s*:?\s*(\d+)', section)
        std_num = f"IS {match.group(1)}" if match else "Unknown"
        if std_num == "Unknown":
            continue

        lines = [l.strip() for l in section.splitlines() if l.strip()]
        title = extract_title(lines, std_num)
        text  = truncate_at_sentence(section)

        # Deduplication: keep the version with more content
        if std_num not in chunk_map or len(text) > len(chunk_map[std_num]["text"]):
            chunk_map[std_num] = {
                "text":        text,
                "standard_id": std_num,
                "title":       title,
                "source":      "BIS_SP21",
            }

    chunks = list(chunk_map.values())
    return chunks


def save_chunks(chunks, out_path=DEFAULT_CHUNKS_PATH):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)
    print(f"✅ Saved {len(chunks)} chunks to {out_path}")


if __name__ == "__main__":
    pdf_path = DEFAULT_PDF_PATH

    if not os.path.exists(pdf_path):
        print("❌ PDF not found at", pdf_path)
        print("   Please place your BIS SP 21 PDF there and rename it BIS_SP21.pdf")
        exit(1)

    print("📄 Reading PDF...")
    pages = extract_text_from_pdf(pdf_path)

    print("✂️  Chunking by BIS standard...")
    chunks = chunk_by_standard(pages)

    print(f"📦 Total unique chunks created: {len(chunks)}")

    print("\n--- Preview of first 3 chunks ---")
    for c in chunks[:3]:
        print(f"  Standard : {c['standard_id']}")
        print(f"  Title    : {c['title']}")
        print(f"  Text     : {c['text'][:120]}...")
        print()

    save_chunks(chunks)