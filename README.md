# 🎵 Song MoodMapper — Music Mood Classification

**Predicting a song's mood from how it _sounds_ vs. what it _says_.**

> A song can sound like a party but read like a breakup. This project builds two
> independent mood models — one on audio features, one on lyrics — and measures
> how often they disagree.

<!-- TODO: replace with a real screenshot or GIF of the running app -->
<!-- ![demo](docs/demo.gif) -->
<!-- 🔗 **Live demo:** https://your-app.vercel.app -->

---

## What it does

Given a song, the system predicts one of four moods — **happy, chill, sad, hyped** — two different ways, and tells you whether the two agree:

- **Audio model** — a Random Forest trained on 9 Spotify acoustic features (tempo, energy, valence, danceability, loudness, etc.)
- **Lyrics model** — VADER sentiment analysis over the song's lyrics
- **Agreement layer** — compares the two predictions and surfaces confidence + disagreement

Predictions are served through a **Flask REST API** and consumed by a **Next.js** frontend.

---

## Key results

| Model | Held-out accuracy | vs. random baseline (25%) |
|-------|------------------|---------------------------|
| Audio-only (Random Forest) | **~35%** | 1.4× |
| Lyrics-only (VADER) | **~26.5%** | ~1.1× |
| Audio + Lyrics fusion | _run `train_fusion_model.py` to fill in_ | — |

**Headline finding:** across ~20,000 songs, the audio and lyrics models agreed on only **~26%** of predictions (disagreed ~74% of the time). The way a song _sounds_ and what it _says_ are largely independent signals — which is exactly why surface-level audio tagging alone produces awkward playlist placements.

> ℹ️ **A note on honesty:** the "true" mood labels are derived from an emotion
> column via a semantic mapping (joy→happy, anger→hyped, etc.), so these models
> predict *that mapping* from each modality. The modest accuracy is a genuine
> finding about how weakly audio features alone encode emotional mood — not a
> bug. Earlier "confusion matrix vs. true labels" figures were computed on
> non-held-out data and overstate accuracy; the numbers above come from a proper
> train/test split.

---

## Architecture

```
Raw Spotify data (500K+ rows)
        │  prep_data.py        → clean + map emotions to 4 moods
        ▼
  songs_mapped.csv
        │  create_balanced_sample.py / audio_data.py → balance to 20K (5K/class)
        ▼
  songs_mapped_20k_balanced.csv
        │
        ├── train_audio_model.py   → RF / LogReg / KNN, 5-fold CV → best model.joblib
        ├── lyrics_classifier_free.py → VADER sentiment → mood
        ├── train_fusion_model.py  → honest audio vs lyrics vs fusion comparison
        │
        ▼
  compare_audio_lyrics.py + enhanced_visualizations.py → 18 evaluation figures
        │
        ▼
  api_server.py (Flask)  ←→  UI/ (Next.js frontend)
```

---

## Tech stack

**ML / data:** Python, pandas, NumPy, scikit-learn (Random Forest, pipelines, cross-validation), VADER sentiment
**Visualization:** matplotlib, seaborn (confusion matrices, PCA/t-SNE, confidence distributions)
**Serving:** Flask + flask-cors (REST API)
**Frontend:** Next.js / React

---

## Run it locally

### 1. Backend (model + API)

```bash
# from project root
pip install -r requirements.txt

# (optional) regenerate data + models from scratch
python src/prep_data.py
python src/create_balanced_sample.py
python src/train_audio_model.py

# honest audio vs lyrics vs fusion comparison
python src/train_fusion_model.py

# start the API
python src/api_server.py        # serves on http://localhost:8000
```

### 2. Frontend

```bash
cd UI
npm install
npm run dev                      # serves on http://localhost:3000
```

### 3. Try a prediction

```bash
curl -X POST http://localhost:8000/api/predict \
  -H "Content-Type: application/json" \
  -d '{"song": "Mr. Brightside", "artist": "The Killers"}'
```

---

## Project structure

```
src/
  prep_data.py                  # raw → mood-mapped dataset
  audio_data.py                 # feature normalization + balanced split
  create_balanced_sample.py     # even 5K-per-class sampling
  train_audio_model.py          # RF/LogReg/KNN with 5-fold CV
  lyrics_classifier_free.py     # VADER lyrics → mood
  train_fusion_model.py         # held-out audio vs lyrics vs fusion
  compare_audio_lyrics.py       # cross-modal agreement analysis
  enhanced_visualizations.py    # 18 evaluation figures
  api_server.py                 # Flask REST API
UI/                             # Next.js frontend
data/processed/                 # mapped + balanced datasets
models/                         # trained model (.joblib)
figures/                        # generated plots
```

---

## Limitations & honest notes

- Labels are a semantic mapping from an emotion column, not human-verified mood tags — accuracy ceilings reflect that.
- Audio features alone are weak predictors of lyrical mood (the central finding, not a defect).
- VADER is a lexicon-based sentiment tool; it has no music-specific tuning.
- Reported accuracy is from a single held-out split; figures labeled "vs. true labels" in earlier versions were not held out and should not be read as accuracy.

## Future work

- Train a single multimodal model on audio + lyrics features (see `train_fusion_model.py`) and report the lift.
- Deploy a live demo (Vercel frontend + Render/Railway API).
- Replace VADER with a fine-tuned text classifier for lyrics.
