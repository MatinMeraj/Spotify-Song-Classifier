"""
Music Mood Classifier - Interactive Dashboard
Predicting a song's mood from how it sounds versus what it says.

Run locally:   streamlit run app.py
"""

import json
from pathlib import Path

import pandas as pd
import streamlit as st

# ------------------------------------------------------------------
# PAGE CONFIG + THEME
# ------------------------------------------------------------------
st.set_page_config(
    page_title="Music Mood Classifier",
    page_icon="🎵",
    layout="wide",
)

CORAL = "#FF6B6B"
TEAL = "#4ECDC4"
BLUE = "#45B7D1"
GREEN = "#96CEB4"
NAVY = "#1d3a5f"

MOOD_COLORS = {"happy": CORAL, "chill": TEAL, "sad": BLUE, "hyped": GREEN}
MOOD_EMOJI = {"happy": "😊", "chill": "😌", "sad": "😢", "hyped": "🔥"}

st.markdown(
    f"""
    <style>
    .big-title {{ font-size:2.6rem; font-weight:800; color:{NAVY}; margin-bottom:0; }}
    .subtitle {{ font-size:1.2rem; color:#555; margin-top:0; }}
    .pill {{ display:inline-block; padding:4px 12px; border-radius:14px;
             background:{TEAL}22; color:{NAVY}; font-weight:600;
             font-size:0.85rem; margin-right:6px; margin-bottom:4px; }}
    div[data-testid="stMetricValue"] {{ font-size:2.1rem; }}
    </style>
    """,
    unsafe_allow_html=True,
)

BASE = Path(__file__).resolve().parent
FIG = BASE / "figures"
MODEL_PATH = BASE / "models" / "mood_model_small.joblib"
RESULTS_PATH = BASE / "results" / "fusion_results.json"

# ------------------------------------------------------------------
# LOADERS (cached so they run once)
# ------------------------------------------------------------------
DEFAULT_RESULTS = {
    "n_rows": 20000, "n_test": 4000, "random_baseline": 0.25,
    "audio_only_acc": 0.348, "lyrics_only_acc": 0.3608,
    "fusion_acc": 0.4155, "fusion_gain_over_best_single": 0.0547,
}


@st.cache_data
def load_results():
    if RESULTS_PATH.exists():
        with open(RESULTS_PATH) as f:
            return json.load(f)
    return DEFAULT_RESULTS


@st.cache_resource
def load_model():
    try:
        import joblib
        return joblib.load(MODEL_PATH)
    except Exception as e:
        return None


@st.cache_resource
def load_vader():
    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        return SentimentIntensityAnalyzer()
    except Exception:
        return None


def show_fig(name, caption=None):
    p = FIG / name
    if p.exists():
        st.image(str(p), caption=caption, use_container_width=True)
    else:
        st.info(f"(figure not found: {name})")


r = load_results()
acc_audio = r["audio_only_acc"] * 100
acc_lyrics = r["lyrics_only_acc"] * 100
acc_fusion = r["fusion_acc"] * 100
acc_baseline = r["random_baseline"] * 100
acc_gain = r["fusion_gain_over_best_single"] * 100

# ------------------------------------------------------------------
# SIDEBAR  (the "built by" credit + links)
# ------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 🎵 Music Mood Classifier")
    st.markdown("**Built by Matin Meraj**")
    st.markdown(
        "[GitHub](https://github.com/MatinMeraj) &nbsp;·&nbsp; "
        "[LinkedIn](https://www.linkedin.com/in/matinmeraj/) &nbsp;·&nbsp; "
        "[Portfolio](https://matinmeraj.netlify.app)"
    )
    st.divider()
    st.markdown("**The idea**")
    st.caption(
        "A song can sound like a party but read like a breakup. This project builds a "
        "mood model on audio, another on lyrics, and shows that combining them wins."
    )
    st.divider()
    st.markdown("**Stack**")
    st.caption("Python · scikit-learn · Random Forest · VADER · Streamlit")
    st.caption("Tip: use the ⋮ menu, top-right, to switch Light / Dark mode.")

# ------------------------------------------------------------------
# HEADER
# ------------------------------------------------------------------
st.markdown('<p class="big-title">🎵 Music Mood Classifier</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="subtitle">Predicting a song\'s mood from how it <i>sounds</i> versus what it <i>says</i></p>',
    unsafe_allow_html=True,
)
st.markdown(
    '<span class="pill">Python</span><span class="pill">scikit-learn</span>'
    '<span class="pill">VADER</span><span class="pill">Random Forest</span>'
    '<span class="pill">Streamlit</span>',
    unsafe_allow_html=True,
)
st.markdown(
    "> **A song can sound like a party but read like a breakup.** "
    "This project builds two mood models. One listens only to the audio. "
    "The other reads only the lyrics. Then it measures what happens when you combine them."
)

st.divider()

tab_try, tab_overview, tab_results, tab_finding, tab_business, tab_method = st.tabs(
    ["🎚️ Try It", "📊 Overview", "🔬 Results", "🎯 The Finding", "💼 Why It Matters", "📋 Method"]
)

# ---------- TAB: TRY IT (interactive) ----------
with tab_try:
    st.header("Predict a song's mood, live")
    st.markdown(
        "Set the audio features on the left and paste some lyrics on the right. "
        "The app runs a trained Random Forest on the audio, runs VADER on the lyrics, "
        "then tells you whether the two agree. Try to build a song that sounds happy "
        "but reads sad."
    )
    st.caption(
        "Note: this live demo runs a lightweight version of the model so it loads fast. "
        "The reported accuracy and charts in the other tabs come from the full model."
    )

    model = load_model()
    vader = load_vader()

    EXAMPLES = {
        "Upbeat sound, sad words (the classic mismatch)": {
            "tempo": 120, "energy": 83, "valence": 87, "danceability": 70,
            "speechiness": 5, "acousticness": 10, "instrumentalness": 0, "liveness": 15,
            "lyrics": "Friends told her she was better off at the bottom of a river. "
                      "I'll swim even when the water's cold. She lost his number and he never called.",
        },
        "Slow and heavy": {
            "tempo": 70, "energy": 25, "valence": 20, "danceability": 30,
            "speechiness": 4, "acousticness": 70, "instrumentalness": 10, "liveness": 10,
            "lyrics": "Tears fall down as I sit here alone, missing everything we used to be.",
        },
        "Party anthem": {
            "tempo": 145, "energy": 92, "valence": 80, "danceability": 88,
            "speechiness": 8, "acousticness": 5, "instrumentalness": 0, "liveness": 30,
            "lyrics": "Turn it up, hands in the air, we dance all night, feel the fire, let's go!",
        },
    }

    choice = st.selectbox("Load an example (optional)", ["Custom"] + list(EXAMPLES.keys()))
    ex = EXAMPLES.get(choice, {})

    left, right = st.columns(2)
    with left:
        st.subheader("🎧 Audio features")
        tempo = st.slider("Tempo (BPM)", 40, 220, ex.get("tempo", 120))
        energy = st.slider("Energy", 0, 100, ex.get("energy", 60))
        valence = st.slider("Valence (positivity)", 0, 100, ex.get("valence", 55))
        danceability = st.slider("Danceability", 0, 100, ex.get("danceability", 60))
        speechiness = st.slider("Speechiness", 0, 100, ex.get("speechiness", 6))
        acousticness = st.slider("Acousticness", 0, 100, ex.get("acousticness", 20))
        instrumentalness = st.slider("Instrumentalness", 0, 100, ex.get("instrumentalness", 0))
        liveness = st.slider("Liveness", 0, 100, ex.get("liveness", 15))

    with right:
        st.subheader("📝 Lyrics")
        lyrics = st.text_area(
            "Paste or type lyrics",
            value=ex.get("lyrics", ""),
            height=220,
            placeholder="Type a few lines of lyrics here...",
        )

    go = st.button("🔮 Predict mood", type="primary", use_container_width=True)

    if go:
        # ----- AUDIO PREDICTION (real model) -----
        audio_mood, audio_conf = None, None
        if model is not None:
            try:
                feats = model["features"]
                row = {
                    "tempo": tempo, "energy": energy, "valence": valence,
                    "danceability": danceability, "speechiness": speechiness,
                    "acousticness": acousticness, "instrumentalness": instrumentalness,
                    "liveness": liveness,
                }
                X = pd.DataFrame([[row.get(f, 0) for f in feats]], columns=feats)
                pipe = model["pipeline"]
                proba = pipe.predict_proba(X)[0]
                classes = list(pipe.classes_)
                best = proba.argmax()
                audio_mood = classes[best]
                audio_conf = float(proba[best])
            except Exception as e:
                st.error(f"Audio model error: {e}")
        else:
            st.warning("Audio model file not found, cannot run the live audio prediction.")

        # ----- LYRICS PREDICTION (VADER) -----
        lyric_mood, lyric_score = None, None
        if lyrics.strip() and vader is not None:
            s = vader.polarity_scores(lyrics)
            comp = s["compound"]
            lyric_score = comp
            if comp <= -0.3:
                lyric_mood = "sad"
            elif comp < 0.1:
                lyric_mood = "chill"
            elif comp < 0.6:
                lyric_mood = "happy"
            else:
                lyric_mood = "hyped"

        # ----- SHOW RESULTS -----
        st.divider()
        rc1, rc2, rc3 = st.columns([1, 1, 1])
        with rc1:
            st.markdown("### 🎧 Audio says")
            if audio_mood:
                st.markdown(f"# {MOOD_EMOJI.get(audio_mood,'')} {audio_mood.title()}")
                st.caption(f"confidence {audio_conf*100:.0f}%")
            else:
                st.write("—")
        with rc2:
            st.markdown("### 📝 Lyrics say")
            if lyric_mood:
                st.markdown(f"# {MOOD_EMOJI.get(lyric_mood,'')} {lyric_mood.title()}")
                st.caption(f"sentiment score {lyric_score:+.2f}")
            elif not lyrics.strip():
                st.info("Add lyrics to get a reading.")
            else:
                st.write("—")
        with rc3:
            st.markdown("### 🤝 Verdict")
            if audio_mood and lyric_mood:
                if audio_mood == lyric_mood:
                    st.success("They **agree**. Sound and words point the same way.")
                else:
                    st.warning(
                        f"They **disagree**. The audio reads *{audio_mood}* while the "
                        f"lyrics read *{lyric_mood}*. This is the party-versus-breakup gap, "
                        "and it is the whole point of the project."
                    )
            else:
                st.write("—")

# ---------- TAB: OVERVIEW ----------
with tab_overview:
    st.header("The headline result")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Random baseline", f"{acc_baseline:.1f}%", help="Four mood classes, pure guessing")
    c2.metric("Audio only", f"{acc_audio:.1f}%", help="Random Forest on audio features")
    c3.metric("Lyrics only", f"{acc_lyrics:.1f}%", help="VADER sentiment on the words")
    c4.metric("Audio + Lyrics (fusion)", f"{acc_fusion:.1f}%", delta=f"+{acc_gain:.1f} pts")

    st.success(
        f"**Fusion wins.** Each signal is weak on its own. Audio reaches {acc_audio:.1f}% and "
        f"lyrics reach {acc_lyrics:.1f}%. Combine them, however, and accuracy jumps to "
        f"**{acc_fusion:.1f}%** on a held-out test set of {r['n_test']:,} songs. "
        f"That is a {acc_gain:.1f}-point gain over the best single model, and {acc_fusion/acc_baseline:.1f} times "
        "the random baseline. The lesson is simple. Sound and words carry different information, "
        "so using both beats using either one alone."
    )

    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("Audio vs Lyrics vs Fusion")
        show_fig("fusion_accuracy_comparison.png")
    with col_b:
        st.subheader("How often do the two agree?")
        show_fig("audio_lyrics_agreement_pie.png",
                 "The audio mood and the lyrics mood agree only about a quarter of the time.")

# ---------- TAB: RESULTS ----------
with tab_results:
    st.header("Model evaluation")
    st.markdown(
        "Both models were scored against the true mood labels. The confusion matrices below "
        "show where each one gets it right, along the diagonal, and where it mixes moods up. "
        "Read them together and a pattern appears. Each model is strong on different moods."
    )
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Audio model")
        show_fig("audio_confusion_matrix_vs_true.png")
    with col2:
        st.subheader("Lyrics model")
        show_fig("lyrics_confusion_matrix_vs_true.png")

    st.divider()
    st.subheader("What each model predicts")
    show_fig("audio_lyrics_distribution_comparison.png",
             "The lyrics model leans hard toward sad. The audio model spreads its guesses more evenly.")

    st.divider()
    st.subheader("Where the two models land differently")
    show_fig("audio_lyrics_confusion_matrix.png")

# ---------- TAB: THE FINDING ----------
with tab_finding:
    st.header("Why fusion helps: the two signals disagree")
    left, right = st.columns([1, 1])
    with left:
        st.markdown(
            f"""
Across roughly {r['n_rows']:,} songs, the audio mood and the lyrics mood
agree only about **a quarter of the time**. In other words, they disagree
on nearly **three out of four songs**.

That disagreement is the discovery, not a bug. How a song sounds and what it
says are largely independent. A track can hit a fast tempo, high energy, and a
bright, hyped sound while the words underneath tell a sad story.

A recommender that listens to audio alone will never catch that gap. One that
reads both can.
            """
        )
        st.info(
            "**\"Even When the Water's Cold\" by !!!**  \n"
            "The lyrics are about heartbreak and drowning sorrow, so the song is labeled **sad**.  \n"
            "The audio, however, is bright and driving, with energy 83 and valence 87. Those are "
            "party numbers.  \n\n"
            "The sound says party. The words say breakup. It is one song with two moods, and an "
            "audio-only recommender would miss it completely."
        )
    with right:
        show_fig("audio_lyrics_agreement_pie.png")

    st.divider()
    st.subheader("The mood map: audio features in two dimensions (t-SNE)")
    show_fig("mood_map_tsne.png",
             "Every dot is a song, colored by its true mood. The colors blend together, "
             "which is exactly why audio alone struggles and why lyrics add real signal.")

# ---------- TAB: WHY IT MATTERS ----------
with tab_business:
    st.header("Why it matters for music platforms")
    st.markdown(
        "Streaming services already know a lot about how songs sound. What they mostly ignore "
        "is what songs say. That is the opening this project points to."
    )
    b1, b2, b3 = st.columns(3)
    with b1:
        st.subheader("Better playlists")
        st.markdown(
            "Spotify and Apple Music lean heavily on audio features and listening behavior. "
            "Lyrical mood catches the songs that sound one way but feel another. As a result, "
            "playlists get fewer jarring transitions."
        )
    with b2:
        st.subheader("Insight for stakeholders")
        st.markdown(
            "The gap between audio mood and lyric mood is a brand new signal, and it is "
            "measurable. It could feed recommendation, retention, and content-tagging decisions."
        )
    with b3:
        st.subheader("The distinction")
        st.markdown(
            "The novelty is fusion. This project uses audio and lyrics together. "
            "Mainstream recommenders largely do not."
        )

# ---------- TAB: METHOD ----------
with tab_method:
    st.header("Method and honest limitations")
    st.markdown(
        f"""
**The data.** The project starts from roughly 551,000 Spotify tracks on Kaggle. Each track
already carries a lyrics-based emotion label: joy, sadness, anger, fear, love, or surprise.
Those six emotions were mapped down to four moods, then balanced to a **{r['n_rows']:,}-song**
training set with 5,000 songs per mood.

**The models.**
- Audio uses a Random Forest on tempo, energy, valence, danceability, and other sound features.
- Lyrics uses VADER sentiment scores on the song text.
- Fusion uses a single Random Forest on both feature sets combined.
- All three are scored on the **same held-out {r['n_test']:,}-song test set**, which keeps the comparison fair.

**The honest caveats.** These are worth stating plainly, because owning them is what makes the
result trustworthy.
- The labels went through two lossy steps. First lyrics became an emotion, then that emotion
  became one of four moods. Anger and fear, for example, were both folded into "hyped."
- The accuracy is modest on purpose. Mood is a genuinely fuzzy four-class target. The interesting
  result is the lift from fusion, not the raw number.
- An earlier version of this project reported accuracy in the 80 to 96 percent range. Those figures
  came from scoring the model on data it had already seen, so they are not trustworthy. They are
  excluded here.
        """
    )
    with st.expander("See the raw dataset class balance"):
        show_fig("mood_class_distribution.png")

st.divider()
st.caption(
    "Built by Matin Meraj with Python, scikit-learn, VADER, and Streamlit. "
    "The numbers load live from results/fusion_results.json. The figures come from the model pipeline."
)
