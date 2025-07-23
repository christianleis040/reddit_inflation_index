import os
import json
import zstandard as zstd
import orjson
from datetime import datetime
import io
import argparse
from datetime import timezone


# Parse optional --month argument (e.g. "2023-06")
parser = argparse.ArgumentParser()
parser.add_argument("--month", type=str, help="Month to filter in format YYYY-MM")
parser.add_argument("--subreddit", type=str, help="Single subreddit to extract (overrides fetch_config)")
args = parser.parse_args()

# Load config
with open("fetch_config.json", "r") as f:
    config = json.load(f)

# Paths
DUMP_DIR = "dumps"
OUTPUT_DIR = "data/raw"

# Date range handling
if args.month:
    start_date = datetime.strptime(f"{args.month}-01", "%Y-%m-%d").replace(tzinfo=timezone.utc)
    if start_date.month == 12:
        end_date = datetime(start_date.year + 1, 1, 1, tzinfo=timezone.utc)
    else:
        end_date = datetime(start_date.year, start_date.month + 1, 1, tzinfo=timezone.utc)
else:
    start_date = datetime.strptime(config.get("start_date", "2005-01-01"), "%Y-%m-%d")
    end_date = datetime.strptime(config.get("end_date", "2025-12-31"), "%Y-%m-%d")

# Other filters
if args.subreddit:
    subreddits = {args.subreddit.lower()}
else:
    subreddits = set(s.lower() for s in config["subreddits"])
keywords = set(w.lower() for w in config["keywords"])
min_score = config.get("min_score", 0)
min_comments = config.get("min_comments", 0)
limit = config.get("limit", None)

# Ensure output folder exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

def match(post):
    created = post.get("created_utc")
    if not created:
        return False

    try:
        created_dt = datetime.fromtimestamp(int(created), tz=timezone.utc)
    except (ValueError, TypeError):
        return False
    if not (start_date <= created_dt < end_date):
        return False

    if post.get("subreddit", "").lower() not in subreddits:
        return False
    if post.get("score", 0) < min_score:
        return False
    if post.get("num_comments", 0) < min_comments:
        return False

    text = f"{post.get('title', '')} {post.get('selftext', '')}".lower()
    return any(k in text for k in keywords)

def process_zst_file(filepath):
    results = []
    try:
        with open(filepath, 'rb') as f:
            dctx = zstd.ZstdDecompressor(max_window_size=2**31)
            with dctx.stream_reader(f) as reader:
                text_stream = io.BufferedReader(reader)  # robuster als TextIOWrapper bei Encoding-Issues
                while True:
                    try:
                        line = text_stream.readline()
                        if not line:
                            break
                        post = orjson.loads(line)
                        if match(post):
                            print(f"Matched post: {post['subreddit']} {post['title'][:80]}")
                            results.append(post)
                            if limit and len(results) >= limit:
                                break
                    except orjson.JSONDecodeError:
                        continue  # Skip corrupt lines
                    except Exception as e:
                        print(f"⚠️ Fehler beim Verarbeiten einer Zeile: {e}")
                        continue
    except Exception as e:
        print(f"❌ Fehler beim Öffnen oder Verarbeiten der Datei {filepath}: {e}")
    return results

def extract_all():
    if not args.month:
        print("Error: --month argument is required to select specific dump file.")
        return

    target_filename = f"RS_{args.month}.zst"
    filepath = os.path.join(DUMP_DIR, target_filename)

    if not os.path.exists(filepath):
        print(f"Dump file not found: {filepath}")
        return

    print(f"Processing: {target_filename}")
    results = process_zst_file(filepath)

    if results:
        out_subreddit = results[0]["subreddit"]

        try:
            ts = int(results[0]["created_utc"])
            out_date = datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d")
        except (ValueError, TypeError) as e:
            print(f"❌ Fehler beim Konvertieren von created_utc: {e}")
            return

        out_dir = os.path.join(OUTPUT_DIR, out_subreddit)
        os.makedirs(out_dir, exist_ok=True)
        out_file = os.path.join(out_dir, f"{out_date}.jsonl")

        with open(out_file, "wb") as f_out:
            for r in results:
                f_out.write(orjson.dumps(r))
                f_out.write(b"\n")

if __name__ == "__main__":
    extract_all()


### NEW

import os
import json
import zstandard as zstd
import orjson
from datetime import datetime
import io
import argparse
from datetime import timezone


# Parse optional --month argument (e.g. "2023-06")
parser = argparse.ArgumentParser()
parser.add_argument("--month", type=str, help="Month to filter in format YYYY-MM")
parser.add_argument("--subreddit", type=str, help="Single subreddit to extract (overrides fetch_config)")
args = parser.parse_args()

# Load config
with open("fetch_config.json", "r") as f:
    config = json.load(f)

# Paths
DUMP_DIR = "dumps"
OUTPUT_DIR = "data/raw"

# Date range handling
if args.month:
    start_date = datetime.strptime(f"{args.month}-01", "%Y-%m-%d").replace(tzinfo=timezone.utc)
    if start_date.month == 12:
        end_date = datetime(start_date.year + 1, 1, 1, tzinfo=timezone.utc)
    else:
        end_date = datetime(start_date.year, start_date.month + 1, 1, tzinfo=timezone.utc)
else:
    start_date = datetime.strptime(config.get("start_date", "2005-01-01"), "%Y-%m-%d")
    end_date = datetime.strptime(config.get("end_date", "2025-12-31"), "%Y-%m-%d")

# Other filters
if args.subreddit:
    subreddits = {args.subreddit.lower()}
else:
    subreddits = set(s.lower() for s in config["subreddits"])
keywords = set(w.lower() for w in config["keywords"])
min_score = config.get("min_score", 0)
min_comments = config.get("min_comments", 0)
limit = config.get("limit", None)

# Ensure output folder exists
os.makedirs(OUTPUT_DIR, exist_ok=True)


import re

# Am Anfang des Skripts, nach dem Laden von keywords
keyword_pattern = re.compile("|".join(re.escape(k) for k in keywords), re.IGNORECASE)

def match(post):
    created = post.get("created_utc")
    if not created:
        return False
    try:
        created_dt = datetime.fromtimestamp(int(created), tz=timezone.utc)
    except (ValueError, TypeError):
        return False
    if not (start_date <= created_dt < end_date):
        return False
    if post.get("subreddit", "").lower() not in subreddits:
        return False
    if post.get("score", 0) < min_score:
        return False
    if post.get("num_comments", 0) < min_comments:
        return False
    text = f"{post.get('title', '')} {post.get('selftext', '')}"
    return keyword_pattern.search(text) is not None


from concurrent.futures import ProcessPoolExecutor
import multiprocessing

def process_chunk(chunk):
    results = []
    for line in chunk:
        try:
            post = orjson.loads(line)
            if match(post):
                results.append(post)
        except:
            continue
    return results

def process_zst_file(filepath):
    results = []
    try:
        with open(filepath, 'rb') as f:
            dctx = zstd.ZstdDecompressor(max_window_size=2**31)
            with dctx.stream_reader(f) as reader:
                text_stream = io.BufferedReader(reader)
                lines = text_stream.readlines()
                chunk_size = len(lines) // multiprocessing.cpu_count()
                chunks = [lines[i:i + chunk_size] for i in range(0, len(lines), chunk_size)]
                with ProcessPoolExecutor() as executor:
                    chunk_results = executor.map(process_chunk, chunks)
                    for chunk_result in chunk_results:
                        results.extend(chunk_result)
                        if limit and len(results) >= limit:
                            break
                if limit and len(results) >= limit:
                    results = results[:limit]
    except Exception as e:
        print(f"❌ Fehler beim Öffnen oder Verarbeiten der Datei {filepath}: {e}")
    return results

def extract_all():
    if not args.month:
        print("Error: --month argument is required to select specific dump file.")
        return

    target_filename = f"RS_{args.month}.zst"
    filepath = os.path.join(DUMP_DIR, target_filename)

    if not os.path.exists(filepath):
        print(f"Dump file not found: {filepath}")
        return

    print(f"Processing: {target_filename}")
    results = process_zst_file(filepath)

    if results:
        out_subreddit = results[0]["subreddit"]

        try:
            ts = int(results[0]["created_utc"])
            out_date = datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d")
        except (ValueError, TypeError) as e:
            print(f"❌ Fehler beim Konvertieren von created_utc: {e}")
            return

        out_dir = os.path.join(OUTPUT_DIR, out_subreddit)
        os.makedirs(out_dir, exist_ok=True)
        out_file = os.path.join(out_dir, f"{out_date}.jsonl")

        with open(out_file, "wb") as f_out:
            for r in results:
                f_out.write(orjson.dumps(r))
                f_out.write(b"\n")

if __name__ == "__main__":
    extract_all()