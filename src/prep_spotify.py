from pathlib import Path
import pandas as pd
import numpy as np
import re

#loading the file input and setting a path for the output
BASE = Path(__file__).resolve().parents[1]   
IN_FILE  = BASE / "data" / "spotify_dataset.csv"      #raw CSV input
OUT_DIR  = BASE / "data" / "processed"       #the output directory
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_FILE = OUT_DIR / "songs.csv"             #cleaned output CSV file

#to check if the file is actually there or not
if IN_FILE.is_dir():
    raise ValueError(f"IN_FILE points to a directory, not a file: {IN_FILE}")
if not IN_FILE.exists():
    raise FileNotFoundError(f"IN_FILE does not exist: {IN_FILE}")


MIN_YEAR = 2010

#define regex for the columns that we wanted to remove
GOODFOR_PATTERN = re.compile(r'^Good for ', re.I)            #columns starting with "Good for "
SIMILAR_PATTERN = re.compile(r'^Similar ', re.I)             #columns starting with "Similar "
SIM_SCORE_PATTERN = re.compile(r'^Similarity Score', re.I)   #columns starting with "Similarity Score"

#important columns 
REQUIRED = [
    'tempo', 'energy', 'valence', 'loudness'
    ]

#final columns we want in our data
FINAL_COLS = [
    'artists', 'track_name', 'text', 'Length', 'mood', 'Genre',
    'tempo', 'loudness', 'energy', 'valence', 'release_year'
]

#map the raw data columns to the cleaned one with renaming them
COL_MAP = {
    'Tempo' : 'tempo', 
    'Energy': 'energy', 
    'Positiveness' : 'valence',
    'Loudness (db)' :'loudness',
    'emotion': 'mood',
    'Artist(s)': 'artists',
    'song':'track_name',
}

OPTIONAL_TO_KEEP = ['artists','track_name','release_year']

#extracting the year only from 2000-max(year)
def extract_year_series(s):
    s = s.astype(str).str.strip()
    last4 = s.str[-4:]
    year = pd.to_numeric(last4, errors='coerce')
    mask = year.isna()
    if mask.any():
        yr2 = s.str.extract(r'((?:19|20)\d{2})', expand=False)
        year = year.fillna(pd.to_numeric(yr2, errors='coerce'))
    return year.astype('Int64')


#based on the tempo, energy, and valence we will decide on the bpm and energy 
#to categorize the four different types

# making the tempo energy and valence numeric


def rule_based_mood(row):
    bpm = row.get('tempo', None)
    energy = row.get('energy', None)
    valence = row.get('valence', None)

    # safety if something is missing
    if pd.isna(bpm) or pd.isna(energy) or pd.isna(valence):
        return None

    #sad: slow, low energy, low valence (negative)
    if bpm < 90 and energy < 50 and valence < 50:
        return 'sad'

    #chill: mid tempo, okay-low energy, neutral valence
    if 90 <= bpm <= 115 and energy < 65 and 45 <= valence <= 70:
        return 'chill'

    #hyped: fast, very energetic, excitement
    if bpm > 125 and energy > 80 and valence >= 50:
        return 'hyped'

    #happy: upbeat, positive valence, but not super-aggressive hype
    if 100 <= bpm <= 130 and 60 <= energy <= 80 and valence > 60:
        return 'happy'

    #fallback bucket: call it 'happy' so nothing is unlabeled
    return 'happy'



def main():
    #loading the data
    df = pd.read_csv(IN_FILE)

    #rename to standard names
    df = df.rename(columns={k:v for k,v in COL_MAP.items() if k in df.columns and k != v})

    #clean loudness data properly 
    if 'loudness' in df.columns:
        s = df['loudness'].astype(str).str.strip()
        #fixing unicode and removing the 'db' from the loudness column 
        s = s.str.replace('\u2212', '-', regex=False)
        s = s.str.replace('[dD][bB]', '', regex=True)
        
        #keeping only digits, sign and decimal
        s = s.str.replace(r'[^0-9\.\-\+]', '', regex=True)
        df['loudness'] = pd.to_numeric(s, errors='coerce').clip(-60, 0)


    #filtering year only from 2010 to 2024
    if 'Release Date' in df.columns:
        df['release_year'] = extract_year_series(df['Release Date'])
        before = len(df)
        df = df[df['release_year'].notna() & (df['release_year'] >= MIN_YEAR)].copy()
        print(f"Year filter >= {MIN_YEAR}: kept {len(df)}/{before}")
    else:
        print("Warning: 'Release Date' not found; skipping year filter.")

    #drop unwanted columns
    drop_cols = []
    for c in df.columns:
        if c == 'Album' or GOODFOR_PATTERN.match(c) or SIMILAR_PATTERN.match(c) or SIM_SCORE_PATTERN.match(c):
            drop_cols.append(c)
    if drop_cols:
        df = df.drop(columns=drop_cols)

    #making the tempo energy and valence for training the model to numeric
    for c in ['tempo','energy','valence']:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')

    #assign mood using our own rules from tempo/energy/valence
    df['mood'] = df.apply(rule_based_mood, axis=1)
    # drop rows where we totally failed to assign a mood (should be rare, but good practice)
    before = len(df)
    df = df[df['mood'].notna()].copy()


    if df.empty:
        raise RuntimeError("No rows left after preprocessing")

    #saving it into our output file
    df.to_csv(OUT_FILE, index=False)
    print(f"Saved cleaned dataset to {OUT_FILE}")
    src_name = 'Loudness (db)' if 'Loudness (db)' in df.columns else ('Loudness (dB)' if 'Loudness (dB)' in df.columns else 'loudness')


if __name__ == "__main__":
    main()