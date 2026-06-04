#!/usr/bin/env python
"""
train_fusion_model.py
---------------------------------------------------------------
Honest, apples-to-apples comparison of three models on ONE held-out test set:

    1. Audio-only      -> Random Forest on 9 Spotify audio features
    2. Lyrics-only     -> Random Forest on 4 VADER sentiment features
    3. Fusion          -> Random Forest on audio + lyrics features combined

Why this script exists
----------------------
The original pipeline reported ~35% audio accuracy and ~26.5% lyrics accuracy,
but the "confusion matrix vs true labels" figures were generated on data the
model had already seen, which makes accuracy look ~96% and is NOT trustworthy.

This script fixes that: it does a SINGLE stratified train/test split up front,
trains everything on train, and reports every number on the SAME untouched
test set. That gives you one clean headline metric you can defend in an
interview, and answers the natural follow-up: "does combining the two signals
beat either one alone?"

Usage
-----
    python src/train_fusion_model.py

Outputs
-------
    - Prints held-out accuracy for audio-only, lyrics-only, and fusion
    - Saves results/fusion_results.json with the numbers
    - Saves figures/fusion_accuracy_comparison.png
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
except ImportError:
    print("ERROR: vaderSentiment not installed. Run: pip install vaderSentiment")
    sys.exit(1)

# ---------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------
BASE = Path(__file__).resolve().parents[1]
DATA_DIR = BASE / "data" / "processed"
FIG_DIR = BASE / "figures"
RESULTS_DIR = BASE / "results"
FIG_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Prefer the balanced 20k file; fall back to the full mapped file
CANDIDATE_PATHS = [
    DATA_DIR / "songs_mapped_20k_balanced.csv",
    DATA_DIR / "songs_balanced_sample.csv",
    DATA_DIR / "songs_mapped.csv",
]

TARGETS = ["happy", "chill", "sad", "hyped"]
AUDIO_FEATURES = [
    "tempo", "energy", "valence", "loudness", "danceability",
    "speechiness", "acousticness", "instrumentalness", "liveness",
]
LYRICS_FEATURES = ["vader_neg", "vader_neu", "vader_pos", "vader_compound"]
RANDOM_STATE = 42


def load_dataset() -> pd.DataFrame:
    for path in CANDIDATE_PATHS:
        if path.exists():
            print(f"[INFO] Loading {path}")
            return pd.read_csv(path)
    raise FileNotFoundError(
        f"None of the candidate datasets were found: {CANDIDATE_PATHS}"
    )


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Match the lowercase column names used elsewhere in the project."""
    renamer = {
        "Tempo": "tempo", "Energy": "energy", "Positiveness": "valence",
        "Loudness (db)": "loudness", "Loudness (dB)": "loudness",
        "Danceability": "danceability", "Speechiness": "speechiness",
        "Liveness": "liveness", "Acousticness": "acousticness",
        "Instrumentalness": "instrumentalness", "song": "track_name",
        "Artist(s)": "artists",
    }
    present = {k: v for k, v in renamer.items() if k in df.columns}
    return df.rename(columns=present) if present else df


def add_vader_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute 4 VADER sentiment scores per row from the lyrics text column."""
    lyrics_col = None
    for cand in ["text", "lyrics", "Lyrics", "Text"]:
        if cand in df.columns:
            lyrics_col = cand
            break
    if lyrics_col is None:
        raise ValueError("No lyrics/text column found; cannot build lyrics model.")

    print(f"[INFO] Scoring lyrics with VADER (column: '{lyrics_col}')...")
    analyzer = SentimentIntensityAnalyzer()

    neg, neu, pos, comp = [], [], [], []
    for i, txt in enumerate(df[lyrics_col].fillna("")):
        s = analyzer.polarity_scores(str(txt))
        neg.append(s["neg"]); neu.append(s["neu"])
        pos.append(s["pos"]); comp.append(s["compound"])
        if (i + 1) % 2000 == 0:
            print(f"   scored {i + 1}/{len(df)}")

    df["vader_neg"] = neg
    df["vader_neu"] = neu
    df["vader_pos"] = pos
    df["vader_compound"] = comp
    return df


def make_rf() -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(
            n_estimators=400,
            class_weight="balanced_subsample",
            random_state=RANDOM_STATE,
        )),
    ])


def evaluate(name, feature_cols, X_train, X_test, y_train, y_test):
    pipe = make_rf()
    pipe.fit(X_train[feature_cols], y_train)
    y_pred = pipe.predict(X_test[feature_cols])
    acc = accuracy_score(y_test, y_pred)
    print(f"\n=== {name} ===")
    print(f"Held-out accuracy: {acc:.3f} ({acc * 100:.1f}%)")
    print(classification_report(y_test, y_pred, labels=TARGETS, zero_division=0))
    return acc


def main():
    df = load_dataset()
    df = normalize_columns(df)

    if "mood" not in df.columns:
        raise ValueError("Expected a 'mood' column.")
    df = df[df["mood"].isin(TARGETS)].copy()

    # Coerce audio features to numeric
    have_audio = [c for c in AUDIO_FEATURES if c in df.columns]
    for c in have_audio:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = add_vader_features(df)

    baseline = 1.0 / len(TARGETS)
    print(f"\n[INFO] Random baseline ({len(TARGETS)} classes): {baseline:.3f}")
    print(f"[INFO] Rows: {len(df)}  |  Audio features: {have_audio}")

    # ONE held-out split used by all three models
    X = df[have_audio + LYRICS_FEATURES]
    y = df["mood"].astype(str)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )
    print(f"[INFO] Train: {len(X_train)}  Test: {len(X_test)}")

    audio_acc = evaluate("AUDIO-ONLY", have_audio, X_train, X_test, y_train, y_test)
    lyrics_acc = evaluate("LYRICS-ONLY", LYRICS_FEATURES, X_train, X_test, y_train, y_test)
    fusion_acc = evaluate("FUSION (audio + lyrics)", have_audio + LYRICS_FEATURES,
                          X_train, X_test, y_train, y_test)

    results = {
        "n_rows": int(len(df)),
        "n_test": int(len(X_test)),
        "random_baseline": round(baseline, 4),
        "audio_only_acc": round(float(audio_acc), 4),
        "lyrics_only_acc": round(float(lyrics_acc), 4),
        "fusion_acc": round(float(fusion_acc), 4),
        "fusion_gain_over_best_single": round(
            float(fusion_acc - max(audio_acc, lyrics_acc)), 4
        ),
    }
    with open(RESULTS_DIR / "fusion_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[INFO] Saved {RESULTS_DIR / 'fusion_results.json'}")
    print(json.dumps(results, indent=2))

    # Plot
    names = ["Random\nbaseline", "Audio\nonly", "Lyrics\nonly", "Fusion"]
    vals = [baseline, audio_acc, lyrics_acc, fusion_acc]
    plt.figure(figsize=(7, 5))
    bars = plt.bar(names, [v * 100 for v in vals],
                   color=["#bbbbbb", "#45B7D1", "#96CEB4", "#FF6B6B"])
    for b, v in zip(bars, vals):
        plt.text(b.get_x() + b.get_width() / 2, v * 100 + 1,
                 f"{v * 100:.1f}%", ha="center", va="bottom")
    plt.ylabel("Held-out accuracy (%)")
    plt.title("Mood Classification: Audio vs Lyrics vs Fusion")
    plt.ylim(0, 100)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fusion_accuracy_comparison.png", dpi=200)
    print(f"[INFO] Saved {FIG_DIR / 'fusion_accuracy_comparison.png'}")


if __name__ == "__main__":
    main()
