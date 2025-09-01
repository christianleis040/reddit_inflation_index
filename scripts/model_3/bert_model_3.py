import os
import json
import argparse
import pandas as pd
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.preprocessing import StandardScaler

# Args
parser = argparse.ArgumentParser()
parser.add_argument("--subreddit", required=True, help="Subreddit name (e.g. food)")
parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
parser.add_argument("--num_epochs", type=int, default=10, help="Number of epochs")
args = parser.parse_args()
SUBREDDIT = args.subreddit
BATCH_SIZE = args.batch_size
NUM_EPOCHS = args.num_epochs

# Paths
RAW_DIR = f"data/raw/{SUBREDDIT}"
INFLATION_CSV = "data/inflation/usa_inflation.csv"

# Model storage (as requested)
MODEL_DIR = os.path.join("models", "model_3", SUBREDDIT)  # e.g., models/model_3/povertyfinance
os.makedirs(MODEL_DIR, exist_ok=True)

# CSV output (as requested)
MODEL3_BASE = "data/output/model_3_bert"
FULL_SET_DIR = os.path.join(MODEL3_BASE, "full_set")
EVAL_DIR = os.path.join(MODEL3_BASE, "eval", SUBREDDIT)
os.makedirs(FULL_SET_DIR, exist_ok=True)
os.makedirs(EVAL_DIR, exist_ok=True)

MODEL_NAME = "distilbert-base-uncased"

# Helper to parse YYYY-MM or YYYY-MM-DD
def parse_date_from_filename(name: str):
    base = name.replace(".jsonl", "")
    for fmt in ("%Y-%m-%d", "%Y-%m"):
        try:
            return datetime.strptime(base, fmt)
        except ValueError:
            continue
    return None

# Extract text
def extract_text(post):
    return (post.get("title", "") + "\n\n" + post.get("selftext", "")).strip()

# VADER
analyzer = SentimentIntensityAnalyzer()

# Load Reddit data
def load_reddit_data():
    texts, dates, sentiments, n_posts_list = [], [], [], []
    for file in sorted(os.listdir(RAW_DIR)):
        if not file.endswith(".jsonl"):
            continue
        dt = parse_date_from_filename(file)
        if dt is None:
            continue
        with open(os.path.join(RAW_DIR, file), "r", encoding="utf-8") as f:
            monthly_texts = []
            monthly_sentiments = []
            n_posts = 0
            for line in f:
                try:
                    post = json.loads(line)
                    txt = extract_text(post)
                    if txt and len(txt.split()) >= 3:
                        monthly_texts.append(txt)
                        sentiment = analyzer.polarity_scores(txt)["compound"]
                        monthly_sentiments.append(sentiment)
                        n_posts += 1
                except Exception:
                    continue
            if monthly_texts:
                texts.append(" ".join(monthly_texts))
                dates.append(dt)
                sentiments.append(np.mean(monthly_sentiments) if monthly_sentiments else 0.0)
                n_posts_list.append(n_posts)
    return pd.DataFrame({"date": dates, "text": texts, "vader_sentiment": sentiments, "n_posts": n_posts_list})

# Load inflation data
def load_inflation():
    print("Checking inflation CSV header preview:")
    with open(INFLATION_CSV, "r", encoding="utf-8") as f:
        print(f.read().splitlines()[:5])
    df = pd.read_csv(INFLATION_CSV, skiprows=1, names=["date", "cpi"])
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df

def main():
    print("Loading data...")
    df_reddit = load_reddit_data()
    df_cpi = load_inflation()

    df_reddit["date"] = pd.to_datetime(df_reddit["date"])
    df_cpi["date"] = pd.to_datetime(df_cpi["date"], errors="coerce")

    df = pd.merge(df_reddit, df_cpi, on="date", how="inner")
    if df.empty:
        print("No matching rows after merge. Check date alignment.")
        return

    print(f"Merged rows: {len(df)}")

    # Normalize CPI
    scaler = StandardScaler()
    df["cpi_normalized"] = scaler.fit_transform(df[["cpi"]])

    # Sort
    df = df.sort_values("date")

    # Train-test split (chronological)
    split_index = int(len(df) * 0.8)
    train_df = df.iloc[:split_index]
    test_df = df.iloc[split_index:]

    # Baseline: Ridge on TF-IDF + VADER
    print("Training baseline Ridge...")
    vectorizer = TfidfVectorizer(max_features=1000)
    X_train_tfidf = vectorizer.fit_transform(train_df["text"])
    X_test_tfidf = vectorizer.transform(test_df["text"])
    X_train = np.hstack([X_train_tfidf.toarray(), train_df[["vader_sentiment"]].values])
    X_test = np.hstack([X_test_tfidf.toarray(), test_df[["vader_sentiment"]].values])
    y_train = train_df["cpi_normalized"]
    y_test = test_df["cpi_normalized"]

    ridge = Ridge(alpha=1.0)
    ridge.fit(X_train, y_train)
    ridge_test_pred = ridge.predict(X_test)
    ridge_test_rmse = np.sqrt(mean_squared_error(y_test, ridge_test_pred))
    print(f"Baseline Ridge RMSE: {ridge_test_rmse:.4f}")

    # Full predictions for baseline (not saved, but available if needed)
    X_full = np.hstack([vectorizer.transform(df["text"]).toarray(), df[["vader_sentiment"]].values])
    ridge_full_pred = scaler.inverse_transform(ridge.predict(X_full).reshape(-1, 1)).flatten()

    # Tokenizer and datasets
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=512)

    cols = ["text", "cpi_normalized", "date", "n_posts", "vader_sentiment"]
    train_dataset = Dataset.from_pandas(train_df[cols].rename(columns={"cpi_normalized": "labels"}))
    test_dataset = Dataset.from_pandas(test_df[cols].rename(columns={"cpi_normalized": "labels"}))
    full_dataset = Dataset.from_pandas(df[cols].rename(columns={"cpi_normalized": "labels"}))

    train_dataset = train_dataset.map(tokenize_function, batched=True)
    test_dataset = test_dataset.map(tokenize_function, batched=True)
    full_dataset = full_dataset.map(tokenize_function, batched=True)

    # Using numpy format is OK for regression with Trainer
    train_dataset.set_format(type="numpy")
    test_dataset.set_format(type="numpy")
    full_dataset.set_format(type="numpy")

    # Model
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=1)

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        mse = mean_squared_error(labels, predictions)
        rmse = np.sqrt(mse)
        return {"mse": mse, "rmse": rmse}

    # Trainer args: write runs to MODEL_DIR
    training_args = TrainingArguments(
        output_dir=MODEL_DIR,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=NUM_EPOCHS,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="rmse",
        greater_is_better=False,
        logging_dir=os.path.join(MODEL_DIR, "logs"),
        logging_steps=10,
        learning_rate=2e-5,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
    )

    # Manual early stopping
    print("Start BERT training...")
    best_rmse = float("inf")
    patience = 3
    patience_counter = 0
    for epoch in range(NUM_EPOCHS):
        trainer.train(resume_from_checkpoint=(epoch > 0))
        eval_results = trainer.evaluate()
        current_rmse = eval_results["eval_rmse"]
        print(f"Epoch {epoch+1} RMSE: {current_rmse:.4f}")
        if current_rmse < best_rmse - 0.01:
            best_rmse = current_rmse
            patience_counter = 0
            trainer.save_model(os.path.join(MODEL_DIR, "best_model"))
        else:
            patience_counter += 1
        if patience_counter >= patience:
            print("Early stopping")
            break

    # Reload best model
    best_path = os.path.join(MODEL_DIR, "best_model")
    model = AutoModelForSequenceClassification.from_pretrained(best_path)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
    )

    print("Evaluate BERT model...")
    eval_results = trainer.evaluate()
    print("BERT eval:", eval_results)

    # Test predictions
    test_predictions = trainer.predict(test_dataset)
    test_pred_df = pd.DataFrame({
        "subreddit": SUBREDDIT,
        "date": list(test_dataset["date"]),
        "n_posts": list(test_dataset["n_posts"]),
        "avg_sentiment": scaler.inverse_transform(test_predictions.predictions).flatten(),
        "true_cpi": scaler.inverse_transform(np.array(test_dataset["labels"]).reshape(-1, 1)).flatten(),
        "vader_sentiment": list(test_dataset["vader_sentiment"]),
    })
    test_pred_path = os.path.join(EVAL_DIR, "test_predictions.csv")
    test_pred_df.to_csv(test_pred_path, index=False)
    print("Saved:", test_pred_path)

    # Full predictions
    print("Predict full dataset...")
    full_predictions = trainer.predict(full_dataset)
    pred_df = pd.DataFrame({
        "subreddit": SUBREDDIT,
        "date": list(full_dataset["date"]),
        "n_posts": list(full_dataset["n_posts"]),
        "avg_sentiment": scaler.inverse_transform(np.array(full_predictions.predictions).reshape(-1, 1)).flatten(),
        "vader_sentiment": list(full_dataset["vader_sentiment"]),
    })

    # Save final full-set CSV with requested naming
    final_csv = os.path.join(FULL_SET_DIR, f"bert_full_2_{SUBREDDIT}.csv")
    pred_df.to_csv(final_csv, index=False)
    print("Saved:", final_csv)
    print("Training completed. Model stored in:", MODEL_DIR)

if __name__ == "__main__":
    main()
