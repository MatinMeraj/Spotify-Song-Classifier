import os
from spotipy import Spotify
from spotipy.oauth2 import SpotifyClientCredentials

os.environ["SPOTIPY_CLIENT_ID"] = "35185f0af35a4a7a9b94b84fd8570375"
os.environ["SPOTIPY_CLIENT_SECRET"] = "0f1d2ee2789d4a69bcb6cd2a25198716"
sp = Spotify(auth_manager=SpotifyClientCredentials())

MOOD_PLAYLISTS = {
    "happy": "37i9dQZF1DXdPec7aLTmlC",
    "chill": "37i9dQZF1DX4WYpdgoIcn6",
    "sad":   "37i9dQZF1DX7qK8ma5wgG1",
    "hyped": "37i9dQZF1DWY4xHQp97fN6",
}

import pandas as pd
def fetch_playlist_tracks(playlist_id):
    rows, results = [], sp.playlist_items(playlist_id, limit=100)
    while results:
        for item in results["items"]:
            t = item.get("track") or {}
            if t and t.get("id"):
                rows.append({
                    "track_id": t["id"],
                    "track_name": t.get("name"),
                    "artists": ", ".join(a["name"] for a in (t.get("artists") or [])),
                    "duration_ms": t.get("duration_ms"),
                    "popularity": t.get("popularity"),
                })
        results = sp.next(results) if results.get("next") else None
    return pd.DataFrame(rows)


dfs = []
for mood, pid in MOOD_PLAYLISTS.items():
    df = fetch_playlist_tracks(pid)
    df["mood"] = mood
    dfs.append(df)
tracks = pd.concat(dfs, ignore_index=True).drop_duplicates(subset=["track_id"])


def fetch_audio_features(track_ids):
    feats = []
    for i in range(0, len(track_ids), 100):
        batch = track_ids[i:i+100]
        resp = sp.audio_features(batch) or []
        feats.extend([f for f in resp if f and f.get("id")])
    return pd.DataFrame(feats)
feats = fetch_audio_features(tracks["track_id"].tolist())


df = tracks.merge(feats, left_on="track_id", right_on="id", how="left")

# keep the useful columns only
keep = [
    "track_id","track_name","artists","duration_ms","popularity","mood",
    "tempo","energy","valence","loudness","danceability","speechiness",
    "acousticness","instrumentalness","liveness","key","mode","time_signature"
]
df = df[[c for c in keep if c in df.columns]].dropna(subset=["tempo","energy","valence"])

df.to_csv("song_mood_dataset.csv", index=False)
print(df.shape, "→ saved to song_mood_dataset.csv")
