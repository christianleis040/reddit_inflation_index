import requests
import pandas as pd
import time
import os
from datetime import datetime, timedelta
import argparse

# === Argument parser ===
parser = argparse.ArgumentParser(description="Fetch Reddit posts via Pushshift API by date")
parser.add_argument("--subreddit", type=str, required=True, help="Subreddit name")
parser.add_argument("--start_date", type=str, required=True, help="Start date YYYY-MM-DD")
parser.add_argument("--end_date", type=str, required=True, help="End date YYYY-MM-DD")
parser.add_argument("--keywords", nargs="*", help="Filter: at least one keyword must be in post")
parser.add_argument("--min_score", type=int, default=0, help="Minimum post score")
parser.add_argument("--min_comments", type=int, default=0, help="Minimum number of comments")
parser.add_argument("--max_comments", type=int, default=10000, help="Maximum number of comments")
args = parser.parse_args()

# === Helper: convert date to epoch ===
def date_to_epoch(date_str):
    return int(datetime.strptime(date_str, "%Y-%m-%d").timestamp())

start = datetime.strptime(args.start_date, "%Y-%m-%d")
end = datetime.strptime(args.end_date, "%Y-%m-%d")
date_list = [(start + timedelta(days=i)).strftime("%Y-%m-%d") for i in range((end - start).days + 1)]

# === Output directory ===
base_dir = f"../data/raw/{args.subreddit}"
os.makedirs(base_dir, exist_ok=True)

# === Loop over days ===
for date_str in date_list:
    after = date_to_epoch(date_str)
    before = after + 86400  # add one day in seconds

    url = "https://api.pushshift.io/reddit/search/submission/"
    params = {
        "subreddit": args.subreddit,
        "after": after,
        "before": before,
        "size": 500,
        "selftext:not": "[removed]"
    }

    try:
        res = requests.get(url, params=params)
        res.raise_for_status()
        data = res.json()["data"]
    except Exception as e:
        print(f"Failed to fetch {date_str}: {e}")
        continue

    posts = []
    for post in data:
        score = post.get("score", 0)
        num_comments = post.get("num_comments", 0)
        if score < args.min_score or num_comments < args.min_comments or num_comments > args.max_comments:
            continue

        title = post.get("title", "")
        selftext = post.get("selftext", "")
        full_text = f"{title} {selftext}".lower()

        if args.keywords and not any(kw.lower() in full_text for kw in args.keywords):
            continue

        posts.append({
            "created_utc": datetime.utcfromtimestamp(post["created_utc"]).strftime('%Y-%m-%d %H:%M:%S'),
            "title": title,
            "selftext": selftext,
            "score": score,
            "num_comments": num_comments,
            "author": post.get("author"),
            "subreddit": args.subreddit,
            "url": post.get("full_link")
        })

    if posts:
        df = pd.DataFrame(posts)
        day_dir = os.path.join(base_dir, date_str)
        os.makedirs(day_dir, exist_ok=True)
        df.to_csv(os.path.join(day_dir, "posts.csv"), index=False)
        print(f"{len(df)} posts from r/{args.subreddit} saved to {day_dir}/posts.csv")
    else:
        print(f"No matching posts for {date_str}")

    time.sleep(1)  # respectful rate limit