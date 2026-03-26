import os
import json
import whisper
import torch

# =========================
# CONFIG
# =========================

AUDIO_FOLDERS = {
    "ml": "audios",      # Machine Learning playlist
    "dl": "audios_1"     # Deep Learning playlist
}

OUTPUT_DIR = "chunks"
MODEL_NAME = "base"     # BEST for RTX 3050 (4GB VRAM)
MAX_WORDS = 120         # chunk size

# =========================
# CUDA CHECK
# =========================

assert torch.cuda.is_available(), "❌ CUDA not available. Fix PyTorch CUDA first."

print("✅ CUDA available:", torch.cuda.get_device_name(0))

torch.backends.cuda.matmul.allow_tf32 = True

# =========================
# LOAD WHISPER (GPU)
# =========================

print("🚀 Loading Whisper model on GPU...")
model = whisper.load_model(MODEL_NAME, device="cuda")

# =========================
# HELPERS
# =========================

def chunk_segments(segments, max_words):
    chunks = []
    current_words = []
    start_time = None

    for seg in segments:
        words = seg["text"].strip().split()

        if not words:
            continue

        if start_time is None:
            start_time = seg["start"]

        current_words.extend(words)

        if len(current_words) >= max_words:
            chunks.append({
                "text": " ".join(current_words),
                "start": start_time,
                "end": seg["end"]
            })
            current_words = []
            start_time = None

    return chunks


def get_tutorial_number(filename):
    # expects: 01_Title.wav
    try:
        return int(filename.split("_")[0])
    except:
        return None

# =========================
# MAIN PIPELINE
# =========================

os.makedirs(OUTPUT_DIR, exist_ok=True)

for domain, audio_dir in AUDIO_FOLDERS.items():
    out_dir = os.path.join(OUTPUT_DIR, f"playlist_{domain}")
    os.makedirs(out_dir, exist_ok=True)

    print(f"\n📂 Processing folder: {audio_dir} ({domain.upper()})")

    for file in sorted(os.listdir(audio_dir)):
        if not file.lower().endswith((".wav", ".mp3")):
            continue

        input_path = os.path.join(audio_dir, file)
        output_path = os.path.join(out_dir, file.rsplit(".", 1)[0] + ".json")

        # ✅ Skip if already processed
        if os.path.exists(output_path):
            print(f"⏭️ Skipping already processed: {file}")
            continue

        print(f"🎧 Transcribing: {file}")

        result = model.transcribe(
            input_path,
            language="en",
            fp16=True,
            verbose=False
        )

        chunks = chunk_segments(result["segments"], MAX_WORDS)

        tutorial_number = get_tutorial_number(file)

        records = []
        for c in chunks:
            records.append({
                "text": c["text"],
                "source_type": "video",
                "domain": domain,
                "title": file,
                "playlist": audio_dir,
                "tutorial_number": tutorial_number,
                "start": c["start"],
                "end": c["end"]
            })

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(records, f, ensure_ascii=False, indent=2)

        print(f"✅ Saved chunks → {output_path}")

print("\n🎉 ALL AUDIO FILES PROCESSED SUCCESSFULLY")
