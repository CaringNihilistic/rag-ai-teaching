import json
import os
import re
from difflib import SequenceMatcher

# =========================
# CONFIG
# =========================
BASE_DIR = r"C:\Project- RAG Based Al Teaching"
URL_FILE = os.path.join(BASE_DIR, "video_url_combined.json")
METADATA_FILE = os.path.join(BASE_DIR, "faiss_metadata_fixed.json")
OUTPUT_FILE = os.path.join(BASE_DIR, "faiss_metadata_final.json")

# =========================
# ADVANCED TITLE CLEANING
# =========================
def clean_title_advanced(title):
    """More aggressive title cleaning"""
    # Remove .wav
    if title.endswith(".wav"):
        title = title[:-4]
    
    # Remove leading numbers (039_, 01_, etc)
    title = re.sub(r'^\d+_', '', title)
    
    # Remove special separators
    title = title.replace('｜', ' ')
    title = title.replace('|', ' ')
    title = title.replace('-', ' ')
    title = title.replace('_', ' ')
    
    # Remove emojis and special chars, keep alphanumeric and spaces
    title = re.sub(r'[^\w\s]', ' ', title)
    
    # Normalize spaces
    title = ' '.join(title.split())
    
    return title.strip().lower()

# =========================
# SIMILARITY SCORE
# =========================
def similarity_score(a, b):
    """Calculate similarity between two strings"""
    return SequenceMatcher(None, a, b).ratio()

# =========================
# FIND BEST MATCH
# =========================
def find_best_match(chunk_title, video_list, threshold=0.6):
    """Find best matching video using fuzzy matching"""
    
    clean_chunk = clean_title_advanced(chunk_title)
    
    best_match = None
    best_score = 0
    
    for video in video_list:
        video_title = video.get("title", "")
        clean_video = clean_title_advanced(video_title)
        
        # Calculate similarity
        score = similarity_score(clean_chunk, clean_video)
        
        # Bonus points for exact word matches
        chunk_words = set(clean_chunk.split())
        video_words = set(clean_video.split())
        common_words = chunk_words & video_words
        
        if len(chunk_words) > 0:
            word_match_ratio = len(common_words) / len(chunk_words)
            score = (score * 0.7) + (word_match_ratio * 0.3)
        
        if score > best_score and score >= threshold:
            best_score = score
            best_match = video
    
    return best_match, best_score

# =========================
# FIX REMAINING URLS
# =========================
def fix_remaining_urls():
    print("="*60)
    print("🔧 FIXING REMAINING URLS WITH FUZZY MATCHING")
    print("="*60 + "\n")
    
    # Load metadata
    print(f"📂 Loading metadata: {METADATA_FILE}")
    with open(METADATA_FILE, encoding="utf-8") as f:
        metadata = json.load(f)
    
    # Load video list
    print(f"📂 Loading videos: {URL_FILE}")
    with open(URL_FILE, encoding="utf-8") as f:
        videos = json.load(f)
    
    # Find chunks without URLs
    chunks_to_fix = []
    for i, chunk in enumerate(metadata):
        if chunk.get("source_type") == "video":
            if not chunk.get("url_fixed") or not chunk.get("source_url"):
                chunks_to_fix.append((i, chunk))
    
    print(f"🔍 Found {len(chunks_to_fix)} chunks to fix\n")
    
    if len(chunks_to_fix) == 0:
        print("✅ All chunks already have URLs!")
        return
    
    # Try to match each chunk
    fixed_count = 0
    unmatched = []
    
    print("🔄 Matching with fuzzy matching...\n")
    
    for idx, (i, chunk) in enumerate(chunks_to_fix):
        title = chunk.get("title", "")
        
        # Find best match
        match, score = find_best_match(title, videos, threshold=0.5)
        
        if match:
            metadata[i]["source_url"] = match["url"]
            metadata[i]["url_fixed"] = True
            metadata[i]["match_score"] = score
            
            # Clean display title
            if title.endswith(".wav"):
                metadata[i]["display_title"] = title[:-4]
            
            fixed_count += 1
            
            if score < 0.7:  # Show low-confidence matches
                print(f"⚠️  Low confidence match (score: {score:.2f})")
                print(f"   Chunk: {title[:60]}")
                print(f"   Matched: {match['title'][:60]}\n")
        else:
            unmatched.append(title)
        
        # Progress
        if (idx + 1) % 50 == 0:
            print(f"  Processed {idx + 1}/{len(chunks_to_fix)}...")
    
    # Save
    print(f"\n💾 Saving to: {OUTPUT_FILE}")
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    
    # Report
    print("\n" + "="*60)
    print("✅ FUZZY MATCHING COMPLETE")
    print("="*60)
    print(f"Chunks attempted: {len(chunks_to_fix)}")
    print(f"Successfully matched: {fixed_count}")
    print(f"Still unmatched: {len(unmatched)}")
    
    if unmatched:
        print(f"\n⚠️  Unmatched chunks (first 20):")
        for title in unmatched[:20]:
            print(f"  - {title}")
        
        # Save unmatched for manual review
        unmatched_file = os.path.join(BASE_DIR, "unmatched_chunks.txt")
        with open(unmatched_file, "w", encoding="utf-8") as f:
            for title in unmatched:
                f.write(f"{title}\n")
        print(f"\n📝 Full list saved to: {unmatched_file}")
    
    print(f"\n📝 Next step:")
    print(f"   Replace your metadata with: {OUTPUT_FILE}")

# =========================
# VERIFY RESULTS
# =========================
def verify_results():
    """Verify the fixed metadata"""
    print("\n" + "="*60)
    print("🔍 VERIFYING RESULTS")
    print("="*60 + "\n")
    
    with open(OUTPUT_FILE, encoding="utf-8") as f:
        metadata = json.load(f)
    
    total_videos = 0
    with_urls = 0
    without_urls = 0
    
    for chunk in metadata:
        if chunk.get("source_type") == "video":
            total_videos += 1
            if chunk.get("source_url") and chunk.get("url_fixed"):
                with_urls += 1
            else:
                without_urls += 1
    
    print(f"Total video chunks: {total_videos}")
    print(f"With URLs: {with_urls} ({100*with_urls/total_videos:.1f}%)")
    print(f"Without URLs: {without_urls} ({100*without_urls/total_videos:.1f}%)")
    
    # Sample some fixed URLs
    print("\n📋 Sample of fixed URLs:")
    sample_count = 0
    for chunk in metadata:
        if chunk.get("source_type") == "video" and chunk.get("url_fixed"):
            title = chunk.get("display_title", chunk.get("title", ""))
            url = chunk.get("source_url", "")
            start = chunk.get("start", 0)
            
            print(f"\n▶ {title[:60]}")
            print(f"  {url}&t={int(start)}s")
            
            sample_count += 1
            if sample_count >= 5:
                break

# =========================
# MAIN
# =========================
if __name__ == "__main__":
    try:
        fix_remaining_urls()
        verify_results()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()