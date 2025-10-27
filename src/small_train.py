""" import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report

BASE = Path(__file__).resolve().parents[1]   
IN_FILE  = BASE / "data" / "processed" / "songs.csv" 

#read the processed data
df = pd.read_csv(IN_FILE)

#balance the small sample set with all the moods having 500 records each

from sklearn.utils import resample
target_n = 500
df_balanced = pd.concat([
    resample(df[df['mood']=='happy'], n_samples=target_n, random_state=42),
    resample(df[df['mood']=='hyped'], n_samples=target_n, random_state=42),
    resample(df[df['mood']=='chill'], n_samples=target_n, random_state=42, replace=True),
    resample(df[df['mood']=='sad'], n_samples=target_n, random_state=42, replace=True)
])

print(df_balanced["mood"].value_counts())


# 3. Separate features and label
y = df_balanced["mood"]

# Drop the target column from features
X_raw = df_balanced.drop(columns=["mood"])

# 4. Keep ONLY numeric features for training
X = X_raw.select_dtypes(include=["number"])

print("Columns going into the model:", list(X.columns))
print("Example rows:\n", X.head())

# 5. Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=42,
    stratify=y
)

# 6. Train a simple KNN classifier
model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train, y_train)

# 7. Predict
y_pred = model.predict(X_test)

# 8. Evaluate
print(classification_report(y_test, y_pred)) """


import os
import time
import pandas as pd
from pathlib import Path

from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from openai import OpenAI

########################################
# CONFIG
########################################

BASE = Path(__file__).resolve().parents[1]
IN_FILE = BASE / "data" / "processed" / "songs.csv"

LYRICS_COL = "text"   # <-- change if your lyrics col is called something else
TARGET_COL = "mood"

OPENAI_MODEL = "gpt-4.1-mini"  # model for lyrics classification

########################################
# OPENAI CLIENT
########################################

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

def classify_mood_from_lyrics(lyrics_text: str) -> str:
    """
    Return one of: happy, chill, sad, hype
    based ONLY on the lyrical content.
    """
    system_instruction = (
        "You are a music mood classifier. "
        "You ONLY answer with exactly one word: happy, chill, sad, or hype.\n"
        "- happy = positive / joyful / uplifting / romantic / grateful\n"
        "- chill = calm / relaxed / introspective / mellow / laid-back\n"
        "- sad   = heartbreak / loss / loneliness / regret / pain / crying\n"
        "- hype  = energetic / flex / confident / party / aggressive / pump-up\n"
        "Do not explain. Do not add punctuation. Just the label."
    )

    # Trim to control tokens/cost
    lyrics_text = str(lyrics_text)[:2000]

    # Handle missing/short lyrics
    if len(lyrics_text.strip()) < 20:
        return "unknown"

    user_prompt = (
        "Classify the mood of this song based on the LYRICS ONLY.\n\n"
        f"LYRICS:\n{lyrics_text}\n\n"
        "Mood?"
    )

    response = client.responses.create(
        model=OPENAI_MODEL,
        instructions=system_instruction,
        input=user_prompt,
    )

    mood_label = response.output_text.strip().lower()

    allowed = {"happy", "chill", "sad", "hype"}
    if mood_label not in allowed:
        # fallback normalization, just in case
        for m in allowed:
            if m in mood_label:
                return m
        return "chill"

    return mood_label

########################################
# LOAD DATA
########################################

df = pd.read_csv(IN_FILE)

# balance each mood to target_n rows
target_n = 500

df_balanced = pd.concat([
    resample(df[df[TARGET_COL] == "happy"], n_samples=target_n, random_state=42, replace=True),
    resample(df[df[TARGET_COL] == "hyped"], n_samples=target_n, random_state=42, replace=True),
    resample(df[df[TARGET_COL] == "chill"], n_samples=target_n, random_state=42, replace=True),
    resample(df[df[TARGET_COL] == "sad"],   n_samples=target_n, random_state=42, replace=True),
], ignore_index=False)

print("Balanced counts:")
print(df_balanced[TARGET_COL].value_counts())

########################################
# FEATURE / LABEL SPLIT
########################################

y = df_balanced[TARGET_COL]

# Drop the label from features
X_raw = df_balanced.drop(columns=[TARGET_COL])

# Keep only numeric audio-type features for KNN (things like bpm, energy, loudness,...)
X = X_raw.select_dtypes(include=["number"])

print("Columns going into KNN model:", list(X.columns))
print("Example numeric feature rows:")
print(X.head())

########################################
# TRAIN / TEST SPLIT
########################################

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=42,
    stratify=y
)

# We'll also grab the corresponding lyrics for JUST the test rows
lyrics_test = df_balanced.loc[X_test.index, LYRICS_COL]

########################################
# TRAIN KNN
########################################

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

y_pred_knn = knn.predict(X_test)

print("=== KNN classification report (audio features only) ===")
print(classification_report(y_test, y_pred_knn))
print("KNN accuracy:", accuracy_score(y_test, y_pred_knn))
print("KNN confusion matrix:")
print(confusion_matrix(y_test, y_pred_knn, labels=["happy","chill","sad","hype"]))

########################################
# GPT LYRICS PREDICTIONS ON SAME TEST ROWS
########################################

gpt_preds = []
for lyrics in lyrics_test:
    mood_label = classify_mood_from_lyrics(lyrics)
    gpt_preds.append(mood_label)
    time.sleep(0.5)  # throttle a bit so you don't spam requests too fast

# align to X_test index
gpt_series = pd.Series(gpt_preds, index=X_test.index, name="mood_lyrics_pred")

########################################
# MERGE PREDICTIONS FOR ANALYSIS
########################################

results_df = pd.DataFrame({
    "true_mood": y_test,
    "pred_knn": y_pred_knn,
    "pred_lyrics": gpt_series,
})

print("\n=== Sample comparison rows ===")
print(results_df.head())

print("\n=== GPT (lyrics-only) classification report ===")
# filter out rows where GPT said "unknown"
mask_known = results_df["pred_lyrics"] != "unknown"
print(classification_report(
    results_df.loc[mask_known, "true_mood"],
    results_df.loc[mask_known, "pred_lyrics"],
    labels=["happy","chill","sad","hype"]
))
print("GPT accuracy (ignoring 'unknown'):",
      accuracy_score(results_df.loc[mask_known, "true_mood"],
                     results_df.loc[mask_known, "pred_lyrics"]))
print("GPT confusion matrix:")
print(confusion_matrix(
    results_df.loc[mask_known, "true_mood"],
    results_df.loc[mask_known, "pred_lyrics"],
    labels=["happy","chill","sad","hype"]
))

########################################
# OPTIONAL: SAVE FOR YOUR REPORT
########################################

# attach both predictions back to test subset.
export_df = df_balanced.loc[X_test.index].copy()
export_df["true_mood"] = y_test
export_df["pred_knn"] = y_pred_knn
export_df["pred_lyrics"] = gpt_series

export_df.to_csv("test_predictions_comparison.csv", index=False)
print("\nWrote test_predictions_comparison.csv")
