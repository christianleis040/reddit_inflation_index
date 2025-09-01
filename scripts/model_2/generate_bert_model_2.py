import os
import argparse
import json
import torch
import pandas as pd
from datetime import datetime
from sklearn.model_selection import train_test_split
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments

# Args
parser = argparse.ArgumentParser()
parser.add_argument("--subreddit", required=True, help="Name of the subreddit folder (e.g. 'personalfinance')")
parser.add_argument("--invert", action="store_true", help="Invert labels (for subreddits where low score = positive sentiment)")
args = parser.parse_args()

SUBREDDIT = args.subreddit
INVERT_LABELS = args.invert

# Paths
RAW_DIR = "data/raw"
MODEL_NAME = "distilbert-base-uncased"

# Basis für Model 2
MODEL2_BASE = "models/model_2"
CKPT_DIR = os.path.join(MODEL2_BASE, f"{SUBREDDIT}_model")
SPLIT_DIR = os.path.join(MODEL2_BASE, "splits", SUBREDDIT)

os.makedirs(CKPT_DIR, exist_ok=True)
os.makedirs(SPLIT_DIR, exist_ok=True)

TRAIN_JSONL = os.path.join(SPLIT_DIR, "train.jsonl")
TEST_JSONL  = os.path.join(SPLIT_DIR, "test.jsonl")


# Extract text from post
def extract_text(post):
    title = post.get("title", "")
    body = post.get("selftext", "")
    return (title + "\n\n" + body).strip()


# Check if filename is a valid date
def _is_valid_date_filename(name: str) -> bool:
    base = name.replace(".jsonl", "")
    for fmt in ("%Y-%m-%d", "%Y-%m"):
        try:
            datetime.strptime(base, fmt)
            return True
        except ValueError:
            pass
    return False


# Load posts and create balanced dataset
def load_posts(subreddit):
    folder = os.path.join(RAW_DIR, subreddit)
    positive_records = []
    negative_records = []

    for filename in sorted(os.listdir(folder)):
        if not filename.endswith(".jsonl") or not _is_valid_date_filename(filename):
            continue

        with open(os.path.join(folder, filename), "r", encoding="utf-8") as f:
            for line in f:
                try:
                    post = json.loads(line)
                    text = extract_text(post)
                    if not text or len(text.split()) < 3:
                        continue

                    score = post.get("score", 0)

                    if score >= 3:
                        label = 0 if INVERT_LABELS else 1
                        positive_records.append({"text": text, "label": label})
                    elif score <= 0:
                        label = 1 if INVERT_LABELS else 0
                        negative_records.append({"text": text, "label": label})

                except:
                    continue

    # Fallback when no negative samples found
    if len(negative_records) == 0:
        print(f"No negative samples found for '{subreddit}'. Using score <= 1 as negative.")
        for filename in sorted(os.listdir(folder)):
            if not filename.endswith(".jsonl"):
                continue
            with open(os.path.join(folder, filename), "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        post = json.loads(line)
                        text = extract_text(post)
                        if not text or len(text.split()) < 3:
                            continue
                        score = post.get("score", 0)
                        if score >= 2:
                            label = 0 if INVERT_LABELS else 1
                            positive_records.append({"text": text, "label": label})
                        elif score <= 1:
                            label = 1 if INVERT_LABELS else 0
                            negative_records.append({"text": text, "label": label})
                    except:
                        continue

    n = min(len(positive_records), len(negative_records))
    if n == 0:
        print(f"Not balanced '{subreddit}'. Quitting.")
        return []

    import random
    balanced = positive_records[:n] + negative_records[:n]
    random.shuffle(balanced)
    return balanced

def save_jsonl(data, path):
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")

# Load data
print("Load data...")
MAX_SAMPLES = 10000 # save memory
all_data = load_posts(SUBREDDIT)[:MAX_SAMPLES]
if not all_data:
    raise SystemExit(f"No data available for subreddit '{SUBREDDIT}'.")

train_data, test_data = train_test_split(all_data, test_size=0.2, random_state=42)
print(f"Train: {len(train_data)} | Test: {len(test_data)}")

save_jsonl(train_data, TRAIN_JSONL)
save_jsonl(test_data, TEST_JSONL)

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize(example):
    return tokenizer(example["text"], truncation=True, padding="max_length", max_length=256)


df_train = pd.read_json(TRAIN_JSONL, lines=True).rename(columns={"label": "labels"})
df_test  = pd.read_json(TEST_JSONL,  lines=True).rename(columns={"label": "labels"})


dataset_train = Dataset.from_pandas(df_train).map(tokenize, batched=True)
dataset_test = Dataset.from_pandas(df_test).map(tokenize, batched=True)

# Model and Trainer
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

# Force CPU usage if no CUDA available (especially for M1/M2 or no GPU)
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Divice: ", device)

training_args = TrainingArguments(
    output_dir=CKPT_DIR,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=2,
    learning_rate=2e-5,
    logging_dir=os.path.join(CKPT_DIR, "logs"),
    logging_steps=50,
    save_total_limit=1,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    no_cuda=(device == "cpu")  # statt use_cpu
)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = torch.argmax(torch.tensor(logits), dim=1)
    acc = (preds == torch.tensor(labels)).float().mean().item()
    return {"accuracy": acc}

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset_train,
    eval_dataset=dataset_test,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
    callbacks=[]

)

trainer.train()

# Save final model and tokenizer manually
model.save_pretrained(CKPT_DIR)
tokenizer.save_pretrained(CKPT_DIR)
print("Training done, output saved in: ", CKPT_DIR)