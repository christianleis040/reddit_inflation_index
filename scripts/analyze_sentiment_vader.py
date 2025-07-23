import os
import argparse
import json
from datetime import datetime
import pandas as pd
from tqdm import tqdm
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Argument parsing
parser = argparse.ArgumentParser()
parser.add_argument("--subreddit", type=str, required=True, help="Name of the subreddit folder (e.g. 'personalfinance')")
parser.add_argument("--data_dir", type=str, default="data/raw", help="Path to the raw data folder")
parser.add_argument("--output_file", type=str, default="vader_sentiment_summary.csv", help="Where to save the aggregated output")
args = parser.parse_args()

# Setup
analyzer = SentimentIntensityAnalyzer()
subreddit_path = os.path.join(args.data_dir, args.subreddit)
results = []

# Check folder
if not os.path.exists(subreddit_path):
    raise FileNotFoundError(f"Subreddit path not found: {subreddit_path}")

# Process all jsonl files
for filename in sorted(os.listdir(subreddit_path)):
    if not filename.endswith(".jsonl"):
        continue

    full_path = os.path.join(subreddit_path, filename)
    date = filename.replace(".jsonl", "")
    sentiments = []

    with open(full_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                post = json.loads(line)
                text = f"{post.get('title', '')} {post.get('selftext', '')}".strip()
                if text:
                    score = analyzer.polarity_scores(text)["compound"]
                    sentiments.append(score)
            except Exception:
                continue

    if sentiments:
        avg_sent = sum(sentiments) / len(sentiments)
        results.append({"subreddit": args.subreddit, "date": date, "n_posts": len(sentiments), "avg_sentiment": avg_sent})

# Create output folder
output_dir = "data/sentiment"
os.makedirs(output_dir, exist_ok=True)

# Save to CSV
df = pd.DataFrame(results)
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date")

output_path = os.path.join(output_dir, f"sentiment_{args.subreddit}.csv")
df.to_csv(output_path, index=False)
print(f"Saved sentiment index to: {output_path}")