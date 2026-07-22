"""
make_small_model.py
Trains a COMPACT version of the audio mood model so the app can deploy on
Streamlit Cloud (GitHub caps files at 100 MB; the original model is ~390 MB).

This keeps the real, live "Try It" prediction working. It just uses a smaller
forest that loads instantly and behaves almost identically for the demo.

Run once:   python make_small_model.py
Output:     models/mood_model_small.joblib   (expected: a few MB)
"""

from pathlib import Path
import joblib
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

BASE = Path(__file__).resolve().parent
DATA = BASE / "data" / "processed" / "songs_mapped_20k_balanced.csv"
OUT = BASE / "models" / "mood_model_small.joblib"
OUT.parent.mkdir(parents=True, exist_ok=True)

# Same 8 features the original model used (loudness was dropped in training)
FEATURES = ["tempo", "energy", "valence", "danceability",
            "speechiness", "acousticness", "instrumentalness", "liveness"]
TARGETS = ["chill", "happy", "hyped", "sad"]

print(f"Loading {DATA} ...")
df = pd.read_csv(DATA)

# normalize column names in case they are capitalized
renamer = {"Tempo": "tempo", "Energy": "energy", "Positiveness": "valence",
           "Danceability": "danceability", "Speechiness": "speechiness",
           "Liveness": "liveness", "Acousticness": "acousticness",
           "Instrumentalness": "instrumentalness"}
df = df.rename(columns={k: v for k, v in renamer.items() if k in df.columns})

df = df[df["mood"].isin(TARGETS)].copy()
for c in FEATURES:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

have = [c for c in FEATURES if c in df.columns]
X = df[have]
y = df["mood"].astype(str)

print(f"Training compact Random Forest on {len(X)} rows, features: {have}")

# COMPACT settings: far fewer, shallower trees -> tiny file, fast load
pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("clf", RandomForestClassifier(
        n_estimators=60,      # was 400
        max_depth=12,         # capped so trees stay small
        min_samples_leaf=5,   # prevents huge, overfit trees
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1,
    )),
])
pipe.fit(X, y)

bundle = {
    "pipeline": pipe,
    "features": have,
    "labels": sorted(y.unique()),
    "version": "streamlit-compact-1.0",
}
joblib.dump(bundle, OUT, compress=3)  # compress shrinks it further

size_mb = OUT.stat().st_size / (1024 * 1024)
print(f"Saved {OUT}  ({size_mb:.1f} MB)")
if size_mb > 90:
    print("WARNING: still large. Tell Claude and we'll shrink further.")
else:
    print("Great, this is small enough for GitHub + Streamlit.")
