
import json

output = []

with open("all_playlists_raw.json", "r", encoding="utf-16") as f:
    for line in f:
        data = json.loads(line)

        playlist_name = (
            "playlist_ml"
            if "PLZoTAELRMXVPBTrWtJkn3wWQxZkmTXGwe" in data.get("playlist_id", "")
            else "playlist_dl"
        )

        output.append({
            "playlist": playlist_name,
            "title": data.get("title"),
            "url": f"https://www.youtube.com/watch?v={data.get('id')}",
            "source": "youtube"
        })

with open("video_urls_combined.json", "w", encoding="utf-8") as f:
    json.dump(output, f, indent=2)

print("✅ video_urls_combined.json created successfully")
