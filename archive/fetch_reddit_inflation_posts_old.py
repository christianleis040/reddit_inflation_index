import praw
import pandas as pd
from datetime import datetime
import os
import argparse

# === Argument parser ===
parser = argparse.ArgumentParser(description="Reddit post fetcher with filters")
parser.add_argument("--subreddit", type=str, required=True, help="Subreddit name (without r/)")
parser.add_argument("--limit", type=int, default=100, help="Maximum number of posts to fetch")
parser.add_argument("--interval", type=int, default=1, help="Take every n-th post only")
parser.add_argument("--min_score", type=int, default=0, help="Minimum post score to include")
parser.add_argument("--min_comments", type=int, default=0, help="Minimum number of comments")
parser.add_argument("--keywords", nargs="*", help="Filter: at least one keyword must be in post text/title")
args = parser.parse_args()

# === Reddit API credentials ===
reddit = praw.Reddit(
    client_id="c5x092Xt4vBTo_YCo5EZpw",
    client_secret="L5193Z4FSRkFECo9CvkaRHqJgoTOVA",
    user_agent="inflation_index by /u/christianleis"
)

# === Prepare output ===
now = datetime.utcnow()
date_str = now.strftime("%Y-%m-%d")
output_dir = f"../data/raw/{args.subreddit}/{date_str}"
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, "posts.csv")

# === Fetch posts ===
posts = []
count = 0

for i, post in enumerate(reddit.subreddit(args.subreddit).new(limit=args.limit * args.interval)):
    if i % args.interval != 0:
        continue  # skip posts depending on interval

    if post.score < args.min_score or post.num_comments < args.min_comments:
        continue

    combined_text = f"{post.title} {post.selftext}".lower()
    if args.keywords and not any(kw.lower() in combined_text for kw in args.keywords):
        continue

    posts.append({
        "created_utc": datetime.utcfromtimestamp(post.created_utc).strftime('%Y-%m-%d %H:%M:%S'),
        "title": post.title,
        "selftext": post.selftext,
        "score": post.score,
        "num_comments": post.num_comments,
        "author": str(post.author),
        "subreddit": args.subreddit,
        "url": post.url
    })
    count += 1

# === Save ===
df = pd.DataFrame(posts)
df.to_csv(output_path, index=False)
print(f"{count} filtered posts from r/{args.subreddit} saved to {output_path}")