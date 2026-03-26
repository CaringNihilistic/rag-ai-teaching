import os
import json
from pypdf import PdfReader

BOOKS_DIR = "Books"
OUT_DIR = "chunks"

MAX_WORDS = 180
OVERLAP = 30


def read_pdf(path):
    reader = PdfReader(path)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text


def chunk_text(text):
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = " ".join(words[i:i + MAX_WORDS])
        chunks.append(chunk)
        i += MAX_WORDS - OVERLAP
    return chunks


BOOK_DOMAIN_MAP = {
    "Hands-On_Machine_Learning": "ml",
    "Deep Learning": "dl"
}

os.makedirs(OUT_DIR, exist_ok=True)

for book in os.listdir(BOOKS_DIR):
    if not book.lower().endswith(".pdf"):
        continue

    domain = "ml" if "Machine" in book else "dl"
    out_path = os.path.join(OUT_DIR, f"books_{domain}")
    os.makedirs(out_path, exist_ok=True)

    print(f"📘 Extracting book: {book}")

    full_text = read_pdf(os.path.join(BOOKS_DIR, book))
    chunks = chunk_text(full_text)

    records = []
    for c in chunks:
        records.append({
            "text": c,
            "source_type": "book",
            "domain": domain,
            "title": book,
            "playlist": None,
            "tutorial_number": None,
            "start": None,
            "end": None
        })

    out_file = os.path.join(out_path, book.replace(".pdf", ".json"))
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    print(f"✅ Book chunked → {out_file}")

print("\n🎉 All books processed successfully")
