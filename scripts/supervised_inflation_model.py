import os
import json
import argparse
import torch
import pandas as pd
from tqdm import tqdm
from datetime import datetime
from transformers import AutoTokenizer, BertModel, BertPreTrainedModel, Trainer, TrainingArguments
from transformers.modeling_outputs import SequenceClassifierOutput
from datasets import Dataset
import torch.nn as nn

# === Argumente ===
parser = argparse.ArgumentParser()
parser.add_argument("--subreddit", required=True, help="Subreddit-Name (z. B. povertyfinance)")
args = parser.parse_args()
SUBREDDIT = args.subreddit

# === Pfade ===
RAW_DIR = f"data/raw/{SUBREDDIT}"
INFLATION_CSV = "data/inlation/usa_inflation.csv"
OUTPUT_DIR = f"data/bert_regression/{SUBREDDIT}"
MODEL_NAME = "distilbert-base-uncased"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# === Modellklasse ===
class BertForRegression(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.bert = BertModel(config)
        self.regressor = nn.Linear(config.hidden_size, 1)
        self.loss_fn = nn.MSELoss()
        self.init_weights()

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        prediction = self.regressor(pooled_output).squeeze()
        loss = None
        if labels is not None:
            loss = self.loss_fn(prediction, labels)
        return SequenceClassifierOutput(loss=loss, logits=prediction)

# === Daten vorbereiten ===
def extract_text(post):
    return (post.get("title", "") + "\n\n" + post.get("selftext", "")).strip()

def load_monthly_data():
    texts, dates = [], []
    for file in sorted(os.listdir(RAW_DIR)):
        if not file.endswith(".jsonl"): continue
        try:
            dt = datetime.strptime(file.replace(".jsonl", ""), "%Y-%m-%d")
        except ValueError:
            continue
        with open(os.path.join(RAW_DIR, file), "r") as f:
            monthly_texts = []
            for line in f:
                try:
                    post = json.loads(line)
                    txt = extract_text(post)
                    if txt and len(txt.split()) >= 3:
                        monthly_texts.append(txt)
                except:
                    continue
        if monthly_texts:
            texts.append(" ".join(monthly_texts))
            dates.append(dt)
    return pd.DataFrame({"date": dates, "text": texts})

def load_inflation():
    return pd.read_csv(INFLATION_CSV, names=["date", "cpi"], header=None, parse_dates=["date"])

print("📥 Lade Daten...")

df_reddit = load_monthly_data()
df_cpi = load_inflation()

# doppelt sicherstellen, dass beide datetime64 sind
df_reddit["date"] = pd.to_datetime(df_reddit["date"])
df_cpi["date"] = pd.to_datetime(df_cpi["date"])

df = pd.merge(df_reddit, df_cpi, on="date")


df["cpi"] = df["cpi"].astype("float32")  # ggf. sicherstellen

# === Tokenisieren & Dataset bauen ===
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
def tokenize(example):
    return tokenizer(example["text"], truncation=True, padding="max_length", max_length=256)

dataset = Dataset.from_pandas(df[["text", "cpi"]].rename(columns={"cpi": "labels"}))
dataset = dataset.map(tokenize, batched=True)
dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

# === Split
train_test = dataset.train_test_split(test_size=0.2)
train_dataset, eval_dataset = train_test["train"], train_test["test"]

# === Training
model = BertForRegression.from_pretrained(MODEL_NAME)
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    num_train_epochs=4,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_dir=os.path.join(OUTPUT_DIR, "logs"),
    load_best_model_at_end=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)

trainer.train()
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print("✅ Training abgeschlossen. Modell gespeichert in:", OUTPUT_DIR)