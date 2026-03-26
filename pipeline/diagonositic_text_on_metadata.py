import json
import os
import re
from collections import defaultdict

BASE_DIR = r"C:\Project- RAG Based Al Teaching"
METADATA_FILE = os.path.join(BASE_DIR, "faiss_metadata_complete.json")

print("="*80)
print("🔍 RAG DATA DIAGNOSTICS")
print("="*80 + "\n")

# Load metadata
print("📂 Loading metadata...")
with open(METADATA_FILE, encoding="utf-8") as f:
    metadata = json.load(f)

print(f"✅ Loaded {len(metadata)} chunks\n")

# =========================
# 1. BASIC STATISTICS
# =========================
print("="*80)
print("📊 BASIC STATISTICS")
print("="*80)

video_chunks = [c for c in metadata if c.get("source_type") == "video"]
book_chunks = [c for c in metadata if c.get("source_type") == "book"]

print(f"Total chunks: {len(metadata)}")
print(f"  📺 Video chunks: {len(video_chunks)}")
print(f"  📚 Book chunks: {len(book_chunks)}")

# Count unique videos
unique_videos = len(set(c.get("source_url") for c in video_chunks if c.get("source_url")))
print(f"  🎬 Unique videos: {unique_videos}\n")

# =========================
# 2. SEARCH FOR SPECIFIC KEYWORDS
# =========================
print("="*80)
print("🔎 KEYWORD SEARCH IN VIDEO TITLES")
print("="*80)

keywords_to_test = ['cnn', 'convolutional', 'rnn', 'recurrent', 'lstm', 
                    'gradient descent', 'backpropagation', 'neural network',
                    'deep learning', 'machine learning', 'regression', 'classification']

for keyword in keywords_to_test:
    matches = []
    for chunk in video_chunks:
        title = chunk.get("title", "").lower()
        if keyword.lower() in title:
            matches.append(chunk)
    
    if matches:
        print(f"\n🔍 '{keyword.upper()}' - Found {len(matches)} chunks from {len(set(c.get('source_url') for c in matches))} videos")
        
        # Show first 3 unique videos
        seen_urls = set()
        count = 0
        for chunk in matches:
            url = chunk.get("source_url")
            if url not in seen_urls and count < 3:
                title = chunk.get("title", "Unknown")
                title_clean = re.sub(r'^\d+_', '', title).replace('.wav', '')
                print(f"   ✓ {title_clean[:70]}")
                print(f"     URL: {url}")
                seen_urls.add(url)
                count += 1
    else:
        print(f"\n❌ '{keyword.upper()}' - No matches found!")

# =========================
# 3. SAMPLE VIDEO CHUNKS
# =========================
print("\n" + "="*80)
print("📹 SAMPLE VIDEO CHUNKS (First 5)")
print("="*80)

for i, chunk in enumerate(video_chunks[:5], 1):
    title = chunk.get("title", "Unknown")
    title_clean = re.sub(r'^\d+_', '', title).replace('.wav', '')
    text_preview = chunk.get("text", "")[:100]
    start = chunk.get("start", 0)
    url = chunk.get("source_url", "No URL")
    
    print(f"\n{i}. Title: {title_clean}")
    print(f"   Start: {start}s ({int(start//60)}:{int(start%60):02d})")
    print(f"   Text: {text_preview}...")
    print(f"   URL: {url}")

# =========================
# 4. CHECK FOR URL ISSUES
# =========================
print("\n" + "="*80)
print("🔗 VIDEO URL CHECK")
print("="*80)

videos_with_url = [c for c in video_chunks if c.get("source_url")]
videos_without_url = [c for c in video_chunks if not c.get("source_url")]

print(f"✅ Videos with URL: {len(videos_with_url)}")
print(f"❌ Videos without URL: {len(videos_without_url)}")

if videos_without_url:
    print(f"\n⚠️  Sample videos without URL:")
    for chunk in videos_without_url[:3]:
        title = chunk.get("title", "Unknown")
        print(f"   - {title}")

# =========================
# 5. ANALYZE VIDEO TITLE PATTERNS
# =========================
print("\n" + "="*80)
print("📝 VIDEO TITLE PATTERNS")
print("="*80)

# Get unique video titles
unique_titles = {}
for chunk in video_chunks:
    url = chunk.get("source_url", "unknown")
    title = chunk.get("title", "Unknown")
    if url not in unique_titles:
        unique_titles[url] = title

print(f"Total unique videos: {len(unique_titles)}\n")

# Show all unique titles (first 50)
print("All video titles in your collection (first 50):")
for i, (url, title) in enumerate(list(unique_titles.items())[:50], 1):
    title_clean = re.sub(r'^\d+_', '', title).replace('.wav', '')
    print(f"{i:3d}. {title_clean}")

if len(unique_titles) > 50:
    print(f"\n... and {len(unique_titles) - 50} more videos")

# =========================
# 6. TEST SPECIFIC QUERY
# =========================
print("\n" + "="*80)
print("🧪 TEST QUERY: 'how do CNNs work'")
print("="*80)

test_query = "how do cnns work"
test_keywords = ['cnn', 'cnns', 'convolutional']

print(f"Query: '{test_query}'")
print(f"Looking for keywords: {test_keywords}\n")

matching_videos = []
for chunk in video_chunks:
    title = chunk.get("title", "").lower()
    text = chunk.get("text", "").lower()
    
    # Check if any keyword in title or text
    title_match = any(kw in title for kw in test_keywords)
    text_match = any(kw in text for kw in test_keywords)
    
    if title_match or text_match:
        matching_videos.append({
            'chunk': chunk,
            'title_match': title_match,
            'text_match': text_match
        })

print(f"Found {len(matching_videos)} matching chunks")

# Separate by match type
title_matches = [v for v in matching_videos if v['title_match']]
text_only_matches = [v for v in matching_videos if v['text_match'] and not v['title_match']]

print(f"  ✅ Title matches: {len(title_matches)}")
print(f"  📄 Text-only matches: {len(text_only_matches)}")

if title_matches:
    print(f"\n🎯 Videos with CNN in TITLE (top 5):")
    seen = set()
    count = 0
    for match in title_matches:
        url = match['chunk'].get('source_url')
        if url not in seen and count < 5:
            title = match['chunk'].get('title', 'Unknown')
            title_clean = re.sub(r'^\d+_', '', title).replace('.wav', '')
            start = match['chunk'].get('start', 0)
            print(f"   {count+1}. {title_clean}")
            print(f"      Timestamp: {int(start//60)}:{int(start%60):02d}")
            print(f"      URL: {url}")
            seen.add(url)
            count += 1
else:
    print(f"\n❌ NO VIDEOS WITH 'CNN' IN TITLE!")
    print(f"   This is why your search isn't working!")

# =========================
# 7. RECOMMENDATIONS
# =========================
print("\n" + "="*80)
print("💡 DIAGNOSTIC SUMMARY & RECOMMENDATIONS")
print("="*80)

issues = []

# Check if CNN videos exist
cnn_in_titles = sum(1 for c in video_chunks if 'cnn' in c.get('title', '').lower() or 'convolutional' in c.get('title', '').lower())
if cnn_in_titles == 0:
    issues.append("❌ NO CNN VIDEOS FOUND - Your video collection may not have CNN tutorials")
else:
    print(f"✅ Found {cnn_in_titles} CNN-related video chunks")

# Check URL coverage
if videos_without_url:
    issues.append(f"⚠️  {len(videos_without_url)} video chunks missing URLs")
else:
    print(f"✅ All video chunks have URLs")

# Check for variety
if unique_videos < 10:
    issues.append(f"⚠️  Only {unique_videos} unique videos - collection may be too small")
else:
    print(f"✅ Good variety: {unique_videos} unique videos")

if issues:
    print("\n⚠️  ISSUES FOUND:")
    for issue in issues:
        print(f"   {issue}")
else:
    print("\n✅ No major issues detected!")

print("\n" + "="*80)
print("✅ DIAGNOSTIC COMPLETE")
print("="*80)