import json
import os
from collections import defaultdict

BASE_DIR = r"C:\Project- RAG Based Al Teaching"
CURRENT_METADATA = os.path.join(BASE_DIR, "faiss_metadata_complete.json")
BOOKS_DL_DIR = os.path.join(BASE_DIR, "chunks", "books_dl")
BOOKS_ML_DIR = os.path.join(BASE_DIR, "chunks", "books_ml")
VIDEO_METADATA_FIXED = os.path.join(BASE_DIR, "faiss_metadata_fixed.json")
OUTPUT_CLEAN = os.path.join(BASE_DIR, "faiss_metadata_clean.json")

print("="*80)
print("🧹 CLEANING METADATA")
print("="*80 + "\n")

# Load current metadata
print("📂 Loading current metadata...")
with open(CURRENT_METADATA, encoding="utf-8") as f:
    current_metadata = json.load(f)

print(f"   Current total: {len(current_metadata)} chunks\n")

# Separate by type
video_chunks = [c for c in current_metadata if c.get("source_type") == "video"]
book_chunks = [c for c in current_metadata if c.get("source_type") == "book"]

print(f"   Videos: {len(video_chunks)} chunks")
print(f"   Books: {len(book_chunks)} chunks\n")

# Check for duplicates in videos
print("🔍 Checking for duplicates...")
seen_chunks = set()
unique_videos = []
duplicate_count = 0

for chunk in video_chunks:
    # Create unique identifier for chunk
    url = chunk.get("source_url", "")
    start = chunk.get("start", 0)
    text = chunk.get("text", "")[:50]  # First 50 chars
    
    chunk_id = f"{url}_{start}_{text}"
    
    if chunk_id not in seen_chunks:
        seen_chunks.add(chunk_id)
        unique_videos.append(chunk)
    else:
        duplicate_count += 1

print(f"   Duplicates removed: {duplicate_count}")
print(f"   Unique video chunks: {len(unique_videos)}\n")

# Count unique video URLs
unique_urls = len(set(c.get("source_url") for c in unique_videos if c.get("source_url")))
print(f"   Unique video URLs: {unique_urls}")

# Verify this matches your expectation
if unique_urls != 236:
    print(f"   ⚠️  Expected 236 videos, found {unique_urls}")
    print(f"   ⚠️  Some videos may still be duplicated or missing\n")
else:
    print(f"   ✅ Matches expected count (153 + 83 = 236)\n")

# Reload books fresh from source
print("📚 Reloading books from source...")
dl_file = os.path.join(BOOKS_DL_DIR, "Deep Learning by Ian Goodfellow, Yoshua Bengio, Aaron Courville.json")
ml_file = os.path.join(BOOKS_ML_DIR, "Hands-On_Machine_Learning_with_Scikit-Learn_Keras_and_Tensorflow_-_Aurelien_Geron.json")

with open(dl_file, encoding="utf-8") as f:
    dl_books = json.load(f)

with open(ml_file, encoding="utf-8") as f:
    ml_books = json.load(f)

print(f"   DL book: {len(dl_books)} chunks")
print(f"   ML book: {len(ml_books)} chunks\n")

# Combine clean data
clean_metadata = unique_videos + dl_books + ml_books

print("="*80)
print("📊 CLEAN METADATA SUMMARY")
print("="*80)
print(f"Total chunks: {len(clean_metadata)}")
print(f"  🎥 Video chunks: {len(unique_videos)}")
print(f"  📚 Book chunks: {len(dl_books) + len(ml_books)}")
print(f"  🎬 Unique videos: {unique_urls}")

# Save clean metadata
print(f"\n💾 Saving clean metadata to: {OUTPUT_CLEAN}")
with open(OUTPUT_CLEAN, "w", encoding="utf-8") as f:
    json.dump(clean_metadata, f, ensure_ascii=False, indent=2)

# Verify CNN videos are present
print("\n🔍 Verifying CNN videos in clean metadata...")
cnn_videos = []
for chunk in clean_metadata:
    if chunk.get("source_type") == "video":
        title = chunk.get("title", "").lower()
        if 'cnn' in title or 'convolutional' in title:
            url = chunk.get("source_url")
            if url not in [v['url'] for v in cnn_videos]:
                cnn_videos.append({
                    'title': chunk.get("title"),
                    'url': url
                })

print(f"✅ Found {len(cnn_videos)} CNN videos:")
for i, v in enumerate(cnn_videos, 1):
    title = v['title'].replace('_', ' ')
    print(f"   {i}. {title}")

print("\n" + "="*80)
print("✅ CLEANING COMPLETE")
print("="*80)
print("\n📝 Next steps:")
print("1. Verify the clean metadata looks correct")
print("2. Rebuild FAISS index with clean metadata")
print("3. Test the system")