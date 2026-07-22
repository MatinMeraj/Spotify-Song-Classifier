# Music Mood Classifier
*(originally **Song MoodMapper**, CMPT 310 group project)*

Predicting a song's mood from how it sounds versus what it says.

**Live app:** https://matinmeraj-musicmood.streamlit.app

A song can sound like a party but read like a breakup. This project builds two mood models. One listens only to the audio. The other reads only the lyrics. Then it measures what happens when you combine them.

## Key results

| Model | Held-out accuracy | vs. random baseline (25%) |
|-------|------------------|---------------------------|
| Audio only (Random Forest) | 34.8% | 1.4x |
| Lyrics only (VADER) | 36.1% | 1.4x |
| Audio + Lyrics (fusion) | 41.5% | 1.7x |

Each signal is weak on its own. Combined, they reach 41.5% on a held-out set of 4,000 songs, a 5.5-point gain over the best single model. Across roughly 20,000 songs, the audio mood and the lyrics mood disagree about 74% of the time. Sound and words carry different information, which is exactly why using both wins.

## What it does

- **Audio model.** A Random Forest trained on 8 Spotify audio features like tempo, energy, and valence.
- **Lyrics model.** VADER sentiment scores on the song text.
- **Fusion model.** One Random Forest trained on both feature sets together.
- **Live dashboard.** A Streamlit app that shows the results and lets you enter a song's features and lyrics to get a live prediction.

## Run it locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Honest notes

- The dataset ships with lyrics-based emotion labels (joy, sadness, anger, fear, love, surprise), which were mapped to 4 moods and balanced to 20,000 songs.
- Accuracy is modest because mood is a fuzzy 4-class target. The interesting result is the lift from fusion, not the raw number.
- The live demo runs a lightweight version of the model so it loads fast. The reported accuracy comes from the full model.

## Credits

This started as a group project for CMPT 310 (Introduction to Artificial Intelligence) at Simon Fraser University, Fall 2025.

**Original team (Song MoodMapper):**
- Nadine Gunawan
- Jim Saraza
- Matin Meraj Mohammadi
- Thanh Vinh Nguyen

**Later additions by Matin Meraj Mohammadi:** the fusion model, the honest held-out evaluation, the Streamlit dashboard, and the live deployment.

## Stack

Python, pandas, scikit-learn, VADER, Streamlit.
