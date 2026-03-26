import json
import faiss
import numpy as np
import requests
import os
import time

# =========================
# CONFIG
# =========================
BASE_DIR = r"C:\Project- RAG Based Al Teaching"
METADATA_FILE = os.path.join(BASE_DIR, "faiss_metadata_final.json")
OUTPUT_INDEX = os.path.join(BASE_DIR, "faiss_rebuilt.index")
CHECKPOINT_FILE = os.path.join(BASE_DIR, "embeddings_checkpoint.npy")

EMBED_MODEL = "nomic-embed-text"
OLLAMA_EMBED_URL = "http://localhost:11434/api/embeddings"

# =========================
# EMBED TEXT (FIXED)
# =========================
def embed_text(text):
    """Generate embedding using FIXED API format"""
    try:
        r = requests.post(
            OLLAMA_EMBED_URL,
            json={
                "model": EMBED_MODEL,
                "prompt": text[:500]  # Limit length
            },
            timeout=30
        )
        r.raise_for_status()
        
        emb = r.json().get("embedding", [])
        if len(emb) == 0:
            return None
        
        return np.array(emb, dtype='float32')
    
    except Exception as e:
        print(f"⚠️  Error: {e}")
        return None

# =========================
# REBUILD INDEX WITH CHECKPOINTS
# =========================
def rebuild_index_with_checkpoints():
    print("="*60)
    print("🔧 REBUILDING FAISS INDEX")
    print("="*60 + "\n")
    
    # Load metadata
    print(f"📂 Loading metadata...")
    with open(METADATA_FILE, encoding="utf-8") as f:
        metadata = json.load(f)
    total_chunks = len(metadata)
    print(f"✅ {total_chunks} chunks to process\n")
    
    # Test embedding first
    print("🧪 Testing embedding...")
    test_emb = embed_text("neural networks and deep learning")
    if test_emb is None or len(test_emb) == 0:
        print("❌ Embedding test failed! Check Ollama.")
        return
    
    dimension = len(test_emb)
    print(f"✅ Embedding dimension: {dimension}\n")
    
    # Check for checkpoint
    start_idx = 0
    if os.path.exists(CHECKPOINT_FILE):
        print("📁 Found checkpoint, loading...")
        embeddings_list = list(np.load(CHECKPOINT_FILE))
        start_idx = len(embeddings_list)
        print(f"✅ Resuming from chunk {start_idx}\n")
    else:
        embeddings_list = []
    
    # Generate embeddings
    print(f"🔄 Generating embeddings ({start_idx}/{total_chunks} done)...")
    print("="*60)
    
    failed_count = 0
    start_time = time.time()
    
    for i in range(start_idx, total_chunks):
        # Get text
        chunk = metadata[i]
        text = chunk.get("text", "").strip()
        if not text:
            text = chunk.get("title", "empty chunk")
        
        # Generate embedding
        emb = embed_text(text)
        
        if emb is not None and len(emb) == dimension:
            embeddings_list.append(emb)
        else:
            embeddings_list.append(np.zeros(dimension, dtype='float32'))
            failed_count += 1
        
        # Progress update
        if (i + 1) % 100 == 0:
            elapsed = time.time() - start_time
            rate = 100 / elapsed
            remaining = (total_chunks - i - 1) / rate
            
            print(f"[{i+1:5d}/{total_chunks}] "
                  f"Progress: {100*(i+1)/total_chunks:5.1f}% | "
                  f"Rate: {rate:.1f} chunks/s | "
                  f"ETA: {remaining/60:.1f} min | "
                  f"Failed: {failed_count}")
            
            # Save checkpoint every 100
            np.save(CHECKPOINT_FILE, np.array(embeddings_list))
            start_time = time.time()
    
    print("="*60)
    print(f"✅ All embeddings generated!\n")
    
    # Convert to array
    embeddings = np.array(embeddings_list, dtype='float32')
    print(f"📊 Embeddings shape: {embeddings.shape}")
    print(f"⚠️  Failed embeddings: {failed_count}")
    
    # Normalize
    print("\n📏 Normalizing embeddings...")
    faiss.normalize_L2(embeddings)
    print(f"✅ Normalized (sample norm: {np.linalg.norm(embeddings[0]):.4f})")
    
    # Build index
    print("\n🔨 Building FAISS index...")
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)
    print(f"✅ Index: {index.ntotal} vectors, dimension {index.d}")
    
    # Save index
    print(f"\n💾 Saving index: {OUTPUT_INDEX}")
    faiss.write_index(index, OUTPUT_INDEX)
    
    # Test search
    print("\n🧪 Testing search...")
    test_queries = [
        "backpropagation",
        "neural networks",
        "gradient descent"
    ]
    
    for query in test_queries:
        q_emb = embed_text(query)
        if q_emb is not None:
            faiss.normalize_L2(q_emb.reshape(1, -1))
            scores, indices = index.search(q_emb.reshape(1, -1), 3)
            
            print(f"\n   Query: '{query}'")
            for j, (score, idx) in enumerate(zip(scores[0], indices[0]), 1):
                if idx < len(metadata):
                    title = metadata[idx].get("title", "")[:50]
                    print(f"   {j}. Score: {score:.4f} | {title}")
    
    # Cleanup checkpoint
    if os.path.exists(CHECKPOINT_FILE):
        os.remove(CHECKPOINT_FILE)
        print(f"\n🗑️  Removed checkpoint file")
    
    print("\n" + "="*60)
    print("✅ REBUILD COMPLETE!")
    print("="*60)
    print(f"\n📝 Next steps:")
    print(f"1. Backup old index:")
    print(f"   > ren faiss.index faiss.index.old")
    print(f"2. Use new index:")
    print(f"   > ren faiss_rebuilt.index faiss.index")
    print(f"3. Test your RAG system!")

# =========================
# MAIN
# =========================
if __name__ == "__main__":
    try:
        rebuild_index_with_checkpoints()
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted! Checkpoint saved.")
        print("   Run again to resume from checkpoint.")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()