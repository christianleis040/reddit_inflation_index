import os
import json
import argparse
import torch
import pandas as pd
from tqdm import tqdm
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig
from datasets import Dataset

# Args
parser = argparse.ArgumentParser()
parser.add_argument("--subreddits", nargs="+", required=True, help="List of subreddits (e.g., povertyfinance food)")
parser.add_argument("--model_dir", default="data/bert_pipeline/model", help="Path to trained models")
parser.add_argument("--model_name", default="distilbert-base-uncased", help="Tokenizer model name")
args = parser.parse_args()


print('Start')
device = "cuda" if torch.cuda.is_available() else "cpu"

# Extract text from post
def extract_text(post):
    title = post.get("title", "")
    body = post.get("selftext", "")
    return (title + "\n\n" + body).strip()

# Predict sentiment for a list of texts
def predict_sentiment(texts, model, tokenizer):
    dataset = Dataset.from_dict({"text": texts})
    dataset = dataset.map(lambda x: tokenizer(x["text"], truncation=True, padding="max_length", max_length=256), batched=True)
    dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32)

    preds = []
    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            preds.extend(torch.argmax(logits, dim=1).cpu().tolist())
    return preds

# Main loop over subreddits
for subreddit in args.subreddits:
    RAW_DIR = f"data/live_set/{subreddit}"
    OUTPUT_PATH = f"data/output/model_2_bert/live_set/bert_full_1_{subreddit}.csv"
    MODEL_DIR = f"models/model_2/{subreddit}_model"
    model_path = os.path.abspath(MODEL_DIR)

    if not os.path.isdir(model_path):
        print(f"Model directory not found for r/{subreddit}: {model_path}")
        continue

    # Load model + tokenizer
    config = AutoConfig.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path, config=config, local_files_only=True).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)

    results = []
    for filename in sorted(os.listdir(RAW_DIR)):
        if not filename.endswith(".jsonl"):
            continue
        date = filename.replace(".jsonl", "")
        try:
            datetime.strptime(date, "%Y-%m-%d")
        except ValueError:
            continue

        with open(os.path.join(RAW_DIR, filename), "r", encoding="utf-8") as f:
            texts = []
            for line in f:
                try:
                    post = json.loads(line)
                    text = extract_text(post)
                    if text and len(text.split()) >= 3:
                        texts.append(text)
                except:
                    continue
            if not texts:
                continue
            sentiments = predict_sentiment(texts, model, tokenizer)
            avg_sentiment = sum(sentiments) / len(sentiments)
            results.append({
                "subreddit": subreddit,
                "date": date,
                "n_posts": len(sentiments),
                "avg_sentiment": avg_sentiment
            })

    # Save results
    df = pd.DataFrame(results)
    if df.empty:
        print(f"No sentiment data for r/{subreddit}")
        continue

    df["date"] = pd.to_datetime(df["date"])
    df = df[["subreddit", "date", "n_posts", "avg_sentiment"]]
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved sentiment data for r/{subreddit} → {OUTPUT_PATH}")