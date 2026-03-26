import subprocess
import json

PLAYLISTS = {
    "playlist_ml": "https://youtube.com/playlist?list=PLZoTAELRMXVPBTrWtJkn3wWQxZkmTXGwe",
    "playlist_dl": "https://youtube.com/playlist?list=PLZoTAELRMXVPGU70ZGsckrMdr0FteeRUi"
}

OUTPUT_FILE = "video_urls.json"

url_map = {}

for key, playlist_url in PLAYLISTS.items():
    print(f"\n🔍 Extracting URLs for {key}")

    cmd = [
        "python", "-m", "yt_dlp",
        "--flat-playlist",
        "--dump-json",
        playlist_url
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    if not result.stdout.strip():
        print("❌ No output from yt-dlp")
        continue

    lines = result.stdout.strip().split("\n")

    playlist_dict = {}
    index = 1

    for line in lines:
        try:
            data = json.loads(line)
            video_id = data.get("id")

            if video_id:
                playlist_dict[f"{index:02d}"] = f"https://www.youtube.com/watch?v={video_id}"
                index += 1

        except json.JSONDecodeError:
            continue

    url_map[key] = playlist_dict
    print(f"  ✅ Extracted {len(playlist_dict)} videos")

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(url_map, f, indent=2)

print("\n🎉 video_urls.json created successfully")
