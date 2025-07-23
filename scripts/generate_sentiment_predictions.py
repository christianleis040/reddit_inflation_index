import os
import json
import argparse
import torch
import pandas as pd
from tqdm import tqdm
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import Dataset

# === Argumente ===
parser = argparse.ArgumentParser()
parser.add_argument("--subreddit", required=True, help="Name des Subreddits (z. B. economy)")
parser.add_argument("--model_dir", default="data/bert_pipeline/model", help="Pfad zum trainierten Modell")
parser.add_argument("--model_name", default="distilbert-base-uncased", help="Tokenizer-Modellname")
args = parser.parse_args()

SUBREDDIT = args.subreddit
MODEL_DIR = args.model_dir
MODEL_NAME = args.model_name

RAW_DIR = f"data/raw/{SUBREDDIT}"
OUTPUT_PATH = f"data/sentiment/sentiment_bert_{SUBREDDIT}.csv"

# === SETUP ===
device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(
    pretrained_model_name_or_path=os.path.abspath(MODEL_DIR)
).to(device)

# === HELPER ===
def extract_text(post):
    title = post.get("title", "")
    body = post.get("selftext", "")
    return (title + "\n\n" + body).strip()

def predict_sentiment(texts):
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

# === INFER SENTIMENT ===
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
        sentiments = predict_sentiment(texts)
        avg_sentiment = sum(sentiments) / len(sentiments)
        results.append({"month": date, "avg_sentiment": avg_sentiment, "n_posts": len(sentiments)})

# === SPEICHERN ===
for r in results:
    r["subreddit"] = SUBREDDIT
    r["date"] = r.pop("month")

df = pd.DataFrame(results)[["subreddit", "date", "n_posts", "avg_sentiment"]]
os.makedirs("data/sentiment", exist_ok=True)
df.to_csv(OUTPUT_PATH, index=False)
print("✅ Fertig:", OUTPUT_PATH)