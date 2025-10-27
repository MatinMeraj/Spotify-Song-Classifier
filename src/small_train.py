import pandas as pd
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


#saving the target variable to y
y = df_balanced["mood"]

#dropping mood
X_raw = df_balanced.drop(columns=["mood"])

#X only has numeric for training
X = X_raw.select_dtypes(include=["number"])

print("Columns going into the model:", list(X.columns))
print("Example rows:\n", X.head())

#train test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=42,
    stratify=y
)

#train with a simple kkn
model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train, y_train)

#prediction
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred)) 



