import os
import json
import zstandard as zstd
import orjson
from datetime import datetime, timezone
import io
import argparse
import re

# Args
parser = argparse.ArgumentParser()
parser.add_argument("--month", type=str, required=True, help="YYYY-MM of the RS_ dump, e.g. 2023-06")
parser.add_argument("--subreddit", type=str, help="Single subreddit to extract (overrides fetch_config)")
args = parser.parse_args()

# Config
with open("fetch_config.json", "r") as f:
    config = json.load(f)

# Paths
DUMP_DIR = "dumps"
OUTPUT_DIR = "data/raw"

# Date window
if "start_date" in config and "end_date" in config:
    start_date = datetime.strptime(config.get("start_date", "2005-01-01"), "%Y-%m-%d").replace(tzinfo=timezone.utc)
    end_date = datetime.strptime(config.get("end_date", "2025-12-31"), "%Y-%m-%d").replace(tzinfo=timezone.utc)
else:
    m = datetime.strptime(f"{args.month}-01", "%Y-%m-%d").replace(tzinfo=timezone.utc)
    end_date = datetime(m.year + (m.month == 12), (m.month % 12) + 1, 1, tzinfo=timezone.utc)
    start_date = m

# Filters
if args.subreddit:
    subreddits = {args.subreddit.lower()}
else:
    subreddits = {s.lower() for s in config.get("subreddits", [])}

keywords = {w.lower() for w in config.get("keywords", [])}
min_score = int(config.get("min_score", 0))
min_comments = int(config.get("min_comments", 0))
limit = config.get("limit", None)
if isinstance(limit, int) and limit <= 0:
    limit = None

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Keyword matching
if keywords:
    keyword_pattern = re.compile("|".join(re.escape(k) for k in keywords), re.IGNORECASE)
    def text_matches(text: str) -> bool:
        return keyword_pattern.search(text) is not None
else:
    def text_matches(text: str) -> bool:
        return True

def match(post) -> bool:
    created = post.get("created_utc")
    if created is None:
        return False
    try:
        created_dt = datetime.fromtimestamp(int(created), tz=timezone.utc)
    except Exception:
        return False
    if not (start_date <= created_dt < end_date):
        return False
    sr = post.get("subreddit", "")
    if not sr or sr.lower() not in subreddits:
        return False
    if post.get("score", 0) < min_score:
        return False
    if post.get("num_comments", 0) < min_comments:
        return False
    text = f"{post.get('title', '')} {post.get('selftext', '')}"
    return text_matches(text)

def extract_month():
    target = os.path.join(DUMP_DIR, f"RS_{args.month}.zst")
    if not os.path.exists(target):
        print(f"Dump file not found: {target}")
        return

    open_files = {}
    written = 0
    try:
        with open(target, "rb") as fh:
            dctx = zstd.ZstdDecompressor(max_window_size=2**31)
            with dctx.stream_reader(fh) as reader:
                stream = io.TextIOWrapper(reader, encoding="utf-8")
                for line in stream:
                    try:
                        post = orjson.loads(line)
                    except Exception:
                        continue
                    if not match(post):
                        continue

                    sr = post["subreddit"]
                    ts = int(post["created_utc"])
                    month_str = datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m")

                    out_dir = os.path.join(OUTPUT_DIR, sr)
                    if (sr, month_str) not in open_files:
                        os.makedirs(out_dir, exist_ok=True)
                        path = os.path.join(out_dir, f"{month_str}.jsonl")
                        open_files[(sr, month_str)] = open(path, "ab")

                    fh_out = open_files[(sr, month_str)]
                    fh_out.write(orjson.dumps(post))
                    fh_out.write(b"\n")

                    written += 1
                    if isinstance(limit, int) and written >= limit:
                        break
    finally:
        for f in open_files.values():
            try:
                f.close()
            except Exception:
                pass

    print(f"Done. Written posts: {written}")

if __name__ == "__main__":
    extract_month()
