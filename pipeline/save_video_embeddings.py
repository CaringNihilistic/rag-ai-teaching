import json
import faiss
import numpy as np
import os

BASE_DIR = r"C:\Project- RAG Based Al Teaching"
VIDEO_INDEX = os.path.join(BASE_DIR, "faiss_rebuilt.index")  # Your completed index
VIDEO_METADATA = os.path.join(BASE_DIR, "faiss_metadata_fixed.json")
VIDEO_EMBEDDINGS_OUTPUT = os.path.join(BASE_DIR, "video_embeddings.npy")

print("="*60)
print("💾 SAVING VIDEO EMBEDDINGS")
print("="*60 + "\n")

# Load video index
print("📂 Loading video index...")
video_index = faiss.read_index(VIDEO_INDEX)
print(f"   Vectors: {video_index.ntotal}, Dimension: {video_index.d}\n")

# Extract all embeddings
print("🔄 Extracting embeddings...")
video_embeddings = np.zeros((video_index.ntotal, video_index.d), dtype='float32')

for i in range(video_index.ntotal):
    video_embeddings[i] = video_index.reconstruct(i)
    if (i + 1) % 1000 == 0:
        print(f"   Extracted {i + 1}/{video_index.ntotal}...")

print(f"\n✅ Extracted {len(video_embeddings)} video embeddings")
print(f"   Shape: {video_embeddings.shape}\n")

# Save
print(f"💾 Saving to: {VIDEO_EMBEDDINGS_OUTPUT}")
np.save(VIDEO_EMBEDDINGS_OUTPUT, video_embeddings)

print("\n✅ VIDEO EMBEDDINGS SAVED!")
print("   Your 5 hours of work is preserved! 🎉")