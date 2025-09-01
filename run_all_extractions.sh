#!/bin/bash
# subreddits=("food" "economy" "walmart" "povertyfinance")

subreddits=("food" "economy" "walmart" "povertyfinance") 
years=(2023)
month="09"

for year in "${years[@]}"; do
    for subreddit in "${subreddits[@]}"; do
        snapshot="${year}-${month}"
        echo ">> Running for $subreddit - $snapshot"
        python scripts/extract_reddit_from_dumps.py --month $snapshot --subreddit $subreddit
    done
done