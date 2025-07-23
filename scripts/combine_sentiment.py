import pandas as pd
from glob import glob
import os

# Eingabe- und Ausgabeordner
SENTIMENT_DIR = "data/sentiment"
OUTPUT_FILE = os.path.join(SENTIMENT_DIR, "sentiment_combined.csv")

# Alle Subreddit-Dateien laden
files = glob(os.path.join(SENTIMENT_DIR, "sentiment_*.csv"))
dfs = []

for f in files:
    df = pd.read_csv(f, parse_dates=["date"])
    dfs.append(df)

# Zusammenführen
all_sentiment = pd.concat(dfs, ignore_index=True)

# Kombinierter Index: gewichtetes Mittel pro Monat
combined = (
    all_sentiment.groupby("date")
    .apply(lambda g: pd.Series({
        "n_total_posts": g["n_posts"].sum(),
        "avg_sentiment": (g["avg_sentiment"] * g["n_posts"]).sum() / g["n_posts"].sum()
    }))
    .reset_index()
)

# Z-Transformation
combined["sentiment_z"] = (
    (combined["avg_sentiment"] - combined["avg_sentiment"].mean()) /
    combined["avg_sentiment"].std()
)

# Speichern
combined.to_csv(OUTPUT_FILE, index=False)
print(f"✅ Kombinierter Sentiment-Index gespeichert unter: {OUTPUT_FILE}")