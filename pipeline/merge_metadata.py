import json
import os

BASE_DIR = r"C:\Project- RAG Based Al Teaching"
BOOKS_DL_DIR = os.path.join(BASE_DIR, "chunks", "books_dl")
BOOKS_ML_DIR = os.path.join(BASE_DIR, "chunks", "books_ml")
VIDEO_METADATA = os.path.join(BASE_DIR, "faiss_metadata_fixed.json")
OUTPUT_FILE = os.path.join(BASE_DIR, "faiss_metadata_complete.json")

print("="*60)
print("🔀 MERGING METADATA")
print("="*60 + "\n")

# Load video metadata
print("📂 Loading video metadata...")
with open(VIDEO_METADATA, encoding="utf-8") as f:
    video_metadata = json.load(f)
video_chunks = [c for c in video_metadata if c.get("source_type") == "video"]
print(f"   Videos: {len(video_chunks)}\n")

# Load books
print("📂 Loading books...")
dl_file = os.path.join(BOOKS_DL_DIR, "Deep Learning by Ian Goodfellow, Yoshua Bengio, Aaron Courville.json")
with open(dl_file, encoding="utf-8") as f:
    dl_chunks = json.load(f)

ml_file = os.path.join(BOOKS_ML_DIR, "Hands-On_Machine_Learning_with_Scikit-Learn_Keras_and_Tensorflow_-_Aurelien_Geron.json")
with open(ml_file, encoding="utf-8") as f:
    ml_chunks = json.load(f)

print(f"   DL book: {len(dl_chunks)} chunks")
print(f"   ML book: {len(ml_chunks)} chunks\n")

# Combine: Videos first (preserve order), then books
all_chunks = video_chunks + dl_chunks + ml_chunks

print("="*60)
print("📊 COMBINED")
print("="*60)
print(f"Total: {len(all_chunks)}")
print(f"  🎥 Videos: {len(video_chunks)}")
print(f"  📚 Books: {len(dl_chunks) + len(ml_chunks)}")

# Save
print(f"\n💾 Saving to: {OUTPUT_FILE}")
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(all_chunks, f, ensure_ascii=False, indent=2)

print("\n✅ METADATA MERGED!")