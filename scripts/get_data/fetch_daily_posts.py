# scripts/fetch_daily_posts.py

import os
import json
import praw
from datetime import datetime, timezone, timedelta, date
from dotenv import load_dotenv
from collections import defaultdict


load_dotenv()

reddit = praw.Reddit(
    client_id="c5x092Xt4vBTo_YCo5EZpw",
    client_secret="L5193Z4FSRkFECo9CvkaRHqJgoTOVA",
    user_agent="inflation_index by /u/christianleis"
)

SUBREDDIT = ["povertyfinance", "food"]
KEYWORDS = [
    "prices", "expensive", "too expensive", "went up", "cost", "costs", "more expensive",
    "inflation", "price increase", "groceries", "grocery", "shopping", "bills", "budget",
    "rent", "housing", "landlord", "utilities", "electricity", "gas bill", "heating", "water bill",
    "food", "eggs", "milk", "bread", "meat", "vegetables", "fruit", "toilet paper",
    "gas", "gasoline", "fuel", "afford", "broke", "struggling", "unaffordable", "cheaper", 
    "saving money", "cutting back", "walmart", "povertyfinance", "personalfinance", "frugal", 
    "budgeting", "cost of living", "price hike", "price gouging", "consumer prices", 
    "inflation rate", "economic hardship"
]
MIN_SCORE = 1
MIN_COMMENTS = 0

def fetch_and_save_today():
    for sub in SUBREDDIT:
        posts = reddit.subreddit(sub).new(limit=1000)
        daily_collected = defaultdict(list)

        for post in posts:
            full_text = f"{post.title} {post.selftext}".lower()
            if any(kw in full_text for kw in KEYWORDS):
                post_date = datetime.fromtimestamp(post.created_utc, tz=timezone.utc).date()
                daily_collected[str(post_date)].append({
                    "created_utc": post.created_utc,
                    "title": post.title,
                    "selftext": post.selftext,
                    "score": post.score,
                    "num_comments": post.num_comments,
                    "subreddit": sub,
                    "url": post.url
                })

        for date_str, posts in daily_collected.items():
            output_dir = f"data/live_set/{sub}"
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, f"{date_str}.jsonl")

            N_DAYS_OVERWRITE = 14  # Number of days to overwrite
            overwrite_cutoff = date.today() - timedelta(days=N_DAYS_OVERWRITE)

            if os.path.exists(output_path) and datetime.strptime(date_str, "%Y-%m-%d").date() < overwrite_cutoff:
                print(f"Already exists and too old → Skipping: {output_path}")
                continue
            elif os.path.exists(output_path):
                print(f"Overwriting recent file: {output_path}")

            with open(output_path, "w", encoding="utf-8") as f:
                for p in posts:
                    f.write(json.dumps(p) + "\n")
            print(f"Saved {len(posts)} posts → {output_path}")

if __name__ == "__main__":
    fetch_and_save_today()