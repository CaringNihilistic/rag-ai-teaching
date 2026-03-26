import os
import shutil

BASE_DIR = r"C:\Project- RAG Based Al Teaching"

old_index = os.path.join(BASE_DIR, "faiss.index")
new_index = os.path.join(BASE_DIR, "faiss_complete.index")
backup_index = os.path.join(BASE_DIR, "faiss.index.old")

# Backup old index
if os.path.exists(old_index):
    print("📦 Backing up old index...")
    shutil.move(old_index, backup_index)
    print("✅ Backed up to faiss.index.old")

# Use new index
if os.path.exists(new_index):
    print("🔄 Activating complete index...")
    shutil.move(new_index, old_index)
    print("✅ faiss_complete.index → faiss.index")

print("\n✅ Index swap complete!")