import os
import argparse
import json
import random
import torch
import pandas as pd
from tqdm import tqdm
from datetime import datetime
from sklearn.model_selection import train_test_split
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments

# === Argumente ===
parser = argparse.ArgumentParser()
parser.add_argument("--subreddit", required=True, help="Name des Subreddits (z. B. economy)")
parser.add_argument("--invert", action="store_true", help="Labels invertieren (für negative Korrelationen)")
args = parser.parse_args()

SUBREDDIT = args.subreddit
INVERT_LABELS = args.invert

RAW_DIR = "data/raw"
MODEL_NAME = "distilbert-base-uncased"
OUTPUT_DIR = f"data/bert_pipeline/{SUBREDDIT}_model"
TRAIN_JSONL = os.path.join(OUTPUT_DIR, "train.jsonl")
TEST_JSONL = os.path.join(OUTPUT_DIR, "test.jsonl")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# === HILFSFUNKTIONEN ===
def extract_text(post):
    title = post.get("title", "")
    body = post.get("selftext", "")
    return (title + "\n\n" + body).strip()

def load_posts(subreddit):
    folder = os.path.join(RAW_DIR, subreddit)
    positive_records = []
    negative_records = []

    for filename in sorted(os.listdir(folder)):
        if not filename.endswith(".jsonl"):
            continue
        date = filename.replace(".jsonl", "")
        try:
            datetime.strptime(date, "%Y-%m-%d")
        except ValueError:
            continue
        with open(os.path.join(folder, filename), "r", encoding="utf-8") as f:
            for line in f:
                try:
                    post = json.loads(line)
                    text = extract_text(post)
                    if not text or len(text.split()) < 3:
                        continue

                    score = post.get("score", 0)  # ✅ MUSS HIER HIN

                    if score >= 3:
                        label = 0 if INVERT_LABELS else 1
                        positive_records.append({"text": text, "label": label})
                    elif score <= 0:
                        label = 1 if INVERT_LABELS else 0
                        negative_records.append({"text": text, "label": label})

                except:
                    continue

    # ➤ fallback: wenn keine negative Beispiele → alternative Schwellen (z. B. score ≥ 2 vs. ≤ 1)
    if len(negative_records) == 0:
        print(f"⚠️ Kein negatives Sample gefunden – versuche mit lockerer Schwelle (score ≥ 2 / ≤ 1)")
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
        print(f"❌ Immer noch keine ausgewogenen Daten für '{subreddit}'. Abbruch.")
        return []

    import random
    balanced = positive_records[:n] + negative_records[:n]
    random.shuffle(balanced)
    return balanced

def save_jsonl(data, path):
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")

# === DATEN VORBEREITEN ===
print("📥 Lade Daten...")
all_data = load_posts(SUBREDDIT)

#train_data, test_data = train_test_split(all_data, test_size=0.2, random_state=42)
train_data, test_data = train_test_split(all_data[:1000], test_size=0.2, random_state=42)
print(f"✅ Train: {len(train_data)} | ✅ Test: {len(test_data)}")

save_jsonl(train_data, TRAIN_JSONL)
save_jsonl(test_data, TEST_JSONL)

# === TOKENIZER UND DATASETS ===
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize(example):
    return tokenizer(example["text"], truncation=True, padding="max_length", max_length=256)

df_train = pd.read_json(TRAIN_JSONL, lines=True)
df_test = pd.read_json(TEST_JSONL, lines=True)

dataset_train = Dataset.from_pandas(df_train).map(tokenize, batched=True)
dataset_test = Dataset.from_pandas(df_test).map(tokenize, batched=True)

# === MODELL UND TRAINING ===
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)


import torch

# Force CPU usage if no CUDA available (especially for M1/M2 or no GPU)
device = "cuda" if torch.cuda.is_available() else "cpu"
print("📟 Verwende Gerät:", device)


training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=3, #erhöhen
    per_device_eval_batch_size=3, #erhöhen
    num_train_epochs=2, #erhöhen
    logging_dir=os.path.join(OUTPUT_DIR, "logs"),
    logging_steps=10,
    save_total_limit=1,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    use_cpu=(device == "cpu")  # transformers ≥ 4.53
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
    callbacks=[]  # ⛔ NeptuneCallback explizit deaktivieren

)

trainer.train()


# Save final model and tokenizer manually
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print("🎉 Training abgeschlossen. Modell gespeichert unter:", OUTPUT_DIR)