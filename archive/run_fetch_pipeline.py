import json
import subprocess

# Load configuration
with open("fetch_config.json") as f:
    config = json.load(f)

# Define time range
start_date = config.get("start_date", "2012-01-01")
end_date = config.get("end_date", "2024-12-31")
max_comments = config.get("max_comments", 1000)

# Iterate over subreddits
for sub in config["subreddits"]:
    cmd = [
        "python", "scripts/fetch_reddit_inflation_posts.py",
        "--subreddit", sub,
        "--start_date", start_date,
        "--end_date", end_date,
        "--min_score", str(config["min_score"]),
        "--min_comments", str(config["min_comments"]),
        "--max_comments", str(max_comments),
        "--keywords"
    ] + config["keywords"]

    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd)