import json
import faiss
import numpy as np
import requests
import os
import time

BASE_DIR = r"C:\Project- RAG Based Al Teaching"
COMPLETE_METADATA = os.path.join(BASE_DIR, "faiss_metadata_complete.json")
VIDEO_EMBEDDINGS = os.path.join(BASE_DIR, "video_embeddings.npy")
FINAL_INDEX = os.path.join(BASE_DIR, "faiss_complete.index")

EMBED_MODEL = "nomic-embed-text"
OLLAMA_EMBED_URL = "http://localhost:11434/api/embeddings"

def embed_text(text):
    try:
        r = requests.post(
            OLLAMA_EMBED_URL,
            json={"model": EMBED_MODEL, "prompt": text[:500]},
            timeout=30
        )
        r.raise_for_status()
        emb = r.json().get("embedding", [])
        return np.array(emb, dtype='float32') if emb else None
    except:
        return None

def add_book_embeddings():
    print("="*60)
    print("🔧 ADDING BOOK EMBEDDINGS TO VIDEO EMBEDDINGS")
    print("="*60 + "\n")
    
    # Load metadata
    print("📂 Loading complete metadata...")
    with open(COMPLETE_METADATA, encoding="utf-8") as f:
        metadata = json.load(f)
    
    total = len(metadata)
    video_count = sum(1 for c in metadata if c.get("source_type") == "video")
    book_count = sum(1 for c in metadata if c.get("source_type") == "book")
    
    print(f"   Total: {total}")
    print(f"   🎥 Videos: {video_count}")
    print(f"   📚 Books: {book_count}\n")
    
    # Load existing video embeddings
    print("📂 Loading saved video embeddings...")
    video_embeddings = np.load(VIDEO_EMBEDDINGS)
    print(f"   Shape: {video_embeddings.shape}")
    print(f"   ✅ Your 5 hours of work loaded!\n")
    
    if len(video_embeddings) != video_count:
        print(f"⚠️  WARNING: Embedding count mismatch!")
        print(f"   Expected: {video_count}, Got: {len(video_embeddings)}")
        print(f"   Proceeding anyway...\n")
    
    dimension = video_embeddings.shape[1]
    
    # Generate embeddings for books only
    print(f"🔄 Generating embeddings for {book_count} book chunks...")
    print(f"   (Only books, videos already done!)")
    print("="*60)
    
    book_embeddings = []
    failed = 0
    start_time = time.time()
    
    # Start from video_count (skip videos)
    for i in range(video_count, total):
        chunk = metadata[i]
        text = chunk.get("text", "").strip() or chunk.get("title", "empty")
        
        emb = embed_text(text)
        
        if emb is not None and len(emb) == dimension:
            book_embeddings.append(emb)
        else:
            book_embeddings.append(np.zeros(dimension, dtype='float32'))
            failed += 1
        
        # Progress
        books_processed = i - video_count + 1
        if books_processed % 50 == 0:
            elapsed = time.time() - start_time
            rate = 50 / elapsed
            remaining = (book_count - books_processed) / rate
            
            print(f"[{books_processed:4d}/{book_count}] "
                  f"{100*books_processed/book_count:5.1f}% | "
                  f"{rate:.1f}/s | ETA: {remaining/60:.1f}m | Failed: {failed}")
            
            start_time = time.time()
    
    print("="*60)
    print(f"✅ Book embeddings generated!\n")
    
    # Combine video + book embeddings
    print("🔗 Combining video and book embeddings...")
    book_embeddings = np.array(book_embeddings, dtype='float32')
    all_embeddings = np.vstack([video_embeddings, book_embeddings])
    
    print(f"   Video embeddings: {video_embeddings.shape}")
    print(f"   Book embeddings: {book_embeddings.shape}")
    print(f"   Combined: {all_embeddings.shape}\n")
    
    # Normalize all
    print("📏 Normalizing combined embeddings...")
    faiss.normalize_L2(all_embeddings)
    print("✅ Normalized\n")
    
    # Build index
    print("🔨 Building complete FAISS index...")
    index = faiss.IndexFlatIP(dimension)
    index.add(all_embeddings)
    print(f"✅ Index built: {index.ntotal} vectors\n")
    
    # Save
    print(f"💾 Saving: {FINAL_INDEX}")
    faiss.write_index(index, FINAL_INDEX)
    
    # Test
    print("\n🧪 Testing complete index...\n")
    
    test_queries = [
        ("backpropagation", "Should find books + videos"),
        ("neural network implementation", "Should favor videos"),
        ("mathematical theory of deep learning", "Should favor books")
    ]
    
    for query, expected in test_queries:
        q_emb = embed_text(query)
        if q_emb is not None:
            faiss.normalize_L2(q_emb.reshape(1, -1))
            scores, indices = index.search(q_emb.reshape(1, -1), 5)
            
            print(f"Query: '{query}'")
            print(f"Expected: {expected}")
            for j, (score, idx) in enumerate(zip(scores[0], indices[0]), 1):
                if idx < len(metadata):
                    src = metadata[idx].get("source_type", "?")
                    title = metadata[idx].get("title", "")[:40]
                    print(f"  {j}. [{src:5s}] {score:.3f} | {title}")
            print()
    
    print("="*60)
    print("✅ COMPLETE INDEX READY!")
    print("="*60)
    print(f"\n⏱️  Time saved: ~4 hours (reused video embeddings)")
    print(f"\n📝 Next steps:")
    print(f"1. Backup: move faiss.index faiss.index.old")
    print(f"2. Use new: move faiss_complete.index faiss.index")
    print(f"3. Test: python final_rag.py")

if __name__ == "__main__":
    try:
        add_book_embeddings()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()