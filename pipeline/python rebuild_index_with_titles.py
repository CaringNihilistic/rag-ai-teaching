import json
import faiss
import numpy as np
import requests
import os
import time
import re

# =========================
# CONFIG
# =========================
BASE_DIR = r"C:\Project- RAG Based Al Teaching"
METADATA_FILE = os.path.join(BASE_DIR, "faiss_metadata_clean.json")
OUTPUT_INDEX = os.path.join(BASE_DIR, "faiss_with_titles.index")
CHECKPOINT_FILE = os.path.join(BASE_DIR, "rebuild_checkpoint.npy")
CHECKPOINT_META = os.path.join(BASE_DIR, "rebuild_checkpoint_idx.txt")

EMBED_MODEL = "nomic-embed-text"
OLLAMA_EMBED_URL = "http://localhost:11434/api/embeddings"

# =========================
# EMBED TEXT WITH TITLE
# =========================
def embed_text_with_title(chunk):
    """Create embedding from title + text"""
    
    # Get title and text
    title = chunk.get("title", "")
    text = chunk.get("text", "")
    source_type = chunk.get("source_type", "")
    
    # Clean title
    if title:
        # Remove number prefixes and .wav extension
        title = re.sub(r'^\d+[\s_-]*', '', title)
        title = title.replace('.wav', '').replace('_', ' ').strip()
    
    # Build combined text based on source type
    if source_type == "video":
        # For videos: emphasize title heavily
        combined = f"Video Title: {title}. {title}. Content: {text[:300]}"
    elif source_type == "book":
        # For books: title is less important
        combined = f"Book: {title}. Content: {text[:400]}"
    else:
        combined = text[:500]
    
    # Embed
    try:
        r = requests.post(
            OLLAMA_EMBED_URL,
            json={"model": EMBED_MODEL, "prompt": combined},
            timeout=30
        )
        r.raise_for_status()
        emb = r.json().get("embedding", [])
        
        if len(emb) == 0:
            return None
        
        return np.array(emb, dtype='float32')
    
    except Exception as e:
        print(f"      ❌ Error: {e}")
        return None

# =========================
# REBUILD INDEX
# =========================
def rebuild_index():
    print("="*80)
    print("🔧 REBUILDING FAISS INDEX WITH TITLES")
    print("="*80 + "\n")
    
    # Load clean metadata
    print("📂 Loading clean metadata...")
    with open(METADATA_FILE, encoding="utf-8") as f:
        metadata = json.load(f)
    
    total = len(metadata)
    videos = sum(1 for c in metadata if c.get("source_type") == "video")
    books = sum(1 for c in metadata if c.get("source_type") == "book")
    
    print(f"   Total chunks: {total}")
    print(f"   🎥 Videos: {videos}")
    print(f"   📚 Books: {books}\n")
    
    # Test embedding
    print("🧪 Testing embedding with title...")
    test_chunk = {
        "title": "Tutorial 21- What is CNN",
        "text": "In this video we will learn about convolutional neural networks",
        "source_type": "video"
    }
    
    test_emb = embed_text_with_title(test_chunk)
    if test_emb is None or len(test_emb) == 0:
        print("❌ Embedding test failed! Check Ollama.")
        return
    
    dimension = len(test_emb)
    print(f"✅ Embedding successful!")
    print(f"   Dimension: {dimension}")
    print(f"   Norm: {np.linalg.norm(test_emb):.4f}\n")
    
    # Check for checkpoint
    start_idx = 0
    if os.path.exists(CHECKPOINT_FILE) and os.path.exists(CHECKPOINT_META):
        print("📁 Found checkpoint, loading...")
        embeddings_list = list(np.load(CHECKPOINT_FILE))
        with open(CHECKPOINT_META, 'r') as f:
            start_idx = int(f.read().strip())
        print(f"   Resuming from chunk {start_idx}/{total}\n")
    else:
        embeddings_list = []
    
    # Generate embeddings
    print("🔄 Generating embeddings with titles included...")
    print("   (This will take 1-2 hours)")
    print("="*80)
    
    failed = 0
    start_time = time.time()
    last_checkpoint_time = time.time()
    
    for i in range(start_idx, total):
        chunk = metadata[i]
        
        # Generate embedding with title
        emb = embed_text_with_title(chunk)
        
        if emb is not None and len(emb) == dimension:
            embeddings_list.append(emb)
        else:
            # Fallback: zero vector
            embeddings_list.append(np.zeros(dimension, dtype='float32'))
            failed += 1
        
        # Progress every 50 chunks
        if (i + 1) % 50 == 0:
            elapsed = time.time() - start_time
            rate = 50 / elapsed
            remaining = (total - i - 1) / rate
            progress = 100 * (i + 1) / total
            
            # Show sample of what we're embedding
            source = chunk.get("source_type", "?")[:5]
            title = chunk.get("title", "Unknown")[:40]
            
            print(f"[{i+1:5d}/{total}] {progress:5.1f}% | "
                  f"{rate:.1f}/s | ETA: {remaining/60:.1f}m | "
                  f"Failed: {failed} | Last: [{source}] {title}")
            
            start_time = time.time()
        
        # Save checkpoint every 5 minutes
        if time.time() - last_checkpoint_time > 300:
            np.save(CHECKPOINT_FILE, np.array(embeddings_list))
            with open(CHECKPOINT_META, 'w') as f:
                f.write(str(i + 1))
            print(f"      💾 Checkpoint saved at {i+1}/{total}")
            last_checkpoint_time = time.time()
    
    print("="*80)
    print("✅ All embeddings generated!\n")
    
    # Build index
    embeddings = np.array(embeddings_list, dtype='float32')
    print(f"📊 Embeddings shape: {embeddings.shape}")
    print(f"⚠️  Failed: {failed}\n")
    
    print("📏 Normalizing embeddings...")
    faiss.normalize_L2(embeddings)
    print(f"✅ Normalized (sample norm: {np.linalg.norm(embeddings[0]):.4f})\n")
    
    print("🔨 Building FAISS index...")
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)
    print(f"✅ Index built: {index.ntotal} vectors, {index.d}D\n")
    
    print(f"💾 Saving index: {OUTPUT_INDEX}")
    faiss.write_index(index, OUTPUT_INDEX)
    
    # Test the new index
    print("\n🧪 TESTING NEW INDEX")
    print("="*80 + "\n")
    
    test_queries = [
        "how do CNNs work",
        "what is backpropagation", 
        "explain RNN"
    ]
    
    for query in test_queries:
        print(f"Query: '{query}'")
        
        # Embed query
        q_emb = embed_text_with_title({"title": "", "text": query, "source_type": "video"})
        if q_emb is not None:
            faiss.normalize_L2(q_emb.reshape(1, -1))
            scores, indices = index.search(q_emb.reshape(1, -1), 5)
            
            print("Top 5 results:")
            for j, (score, idx) in enumerate(zip(scores[0], indices[0]), 1):
                if idx < len(metadata):
                    chunk = metadata[idx]
                    source = chunk.get("source_type", "?")
                    title = chunk.get("title", "Unknown")
                    title_clean = re.sub(r'^\d+[\s_-]*', '', title).replace('.wav', '')[:60]
                    
                    print(f"   {j}. [{source:5s}] {score:.3f} | {title_clean}")
            print()
    
    # Cleanup checkpoint
    if os.path.exists(CHECKPOINT_FILE):
        os.remove(CHECKPOINT_FILE)
    if os.path.exists(CHECKPOINT_META):
        os.remove(CHECKPOINT_META)
    
    print("="*80)
    print("✅ REBUILD COMPLETE!")
    print("="*80)
    print("\n📝 Next steps:")
    print("1. Backup old index:")
    print("   > move faiss.index faiss.index.backup")
    print("2. Use new index:")
    print("   > move faiss_with_titles.index faiss.index")
    print("3. Update app.py to use faiss_metadata_clean.json")
    print("4. Restart Flask app and test!")

# =========================
# MAIN
# =========================
if __name__ == "__main__":
    try:
        rebuild_index()
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted! Checkpoint saved.")
        print("   Run again to resume from checkpoint.")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()