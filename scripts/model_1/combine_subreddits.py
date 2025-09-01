import pandas as pd
from glob import glob
import os

# configuration
BASE_DIR = "data/output/model_1_vader/full_set"
INPUT_PATTERN = os.path.join(BASE_DIR, "vader_full_*.csv")
OUTPUT_FILE = os.path.join(BASE_DIR, "vader_full_combined.csv")

# ensure output directory exists
os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

# load all sentiment files
files = sorted(glob(INPUT_PATTERN))
if not files:
    raise FileNotFoundError(f"No input files found matching: {INPUT_PATTERN}")

dfs = []
for f in files:
    df = pd.read_csv(f, parse_dates=["date"])
    missing = {"date", "n_posts", "avg_sentiment"} - set(df.columns)
    if missing:
        raise ValueError(f"{f} is missing columns: {missing}")
    dfs.append(df)

# combine all
all_sentiment = pd.concat(dfs, ignore_index=True)

# compute weighted average sentiment per date
tmp = all_sentiment.copy()
tmp["weighted"] = tmp["avg_sentiment"] * tmp["n_posts"]
combined = (
    tmp.groupby("date", as_index=False)
      .agg(n_total_posts=("n_posts", "sum"), weighted=("weighted", "sum"))
)
combined["avg_sentiment"] = combined["weighted"] / combined["n_total_posts"]
combined = combined.drop(columns=["weighted"]).sort_values("date")

# z transform
std = combined["avg_sentiment"].std()
combined["sentiment_z"] = (combined["avg_sentiment"] - combined["avg_sentiment"].mean()) / (std if std != 0 else 1)

# save
combined.to_csv(OUTPUT_FILE, index=False)
print(f"Saved: {OUTPUT_FILE}")
