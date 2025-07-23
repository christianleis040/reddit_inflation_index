## How to run
- pip install -r requirements.txt
- python scripts/extract_reddit_from_dumps.py --month 2024-12
- ./run_all_extractions.sh
- python scripts/analyze_sentiment_vader.py --subreddit personalfinance
- python scripts/combine_sentiment.py
- python scripts/generate_bert_sentiment_data.py
- python scripts/predict_bert_sentiment.py --subreddit economy
- python scripts/generate_sentiment_predictions.py  # falls du `SUBREDDIT` dynamisch per argparse machen willst, sag Bescheid
- python scripts/generate_sentiment_predictions.py




## Ergenisse 1st VADER (13.07)
Korrelation für povertyfinance (z-transformiert): -0.872 ; 0.850
Korrelation für food (z-transformiert): -0.781, 0.885 (nicht invert)
Korrelation für economy (z-transformiert): 0.778; 0.880
Korrelation für walmart (z-transformiert): -0.630; 0.791

Korrelation für frugal (z-transformiert): -0.554; 0.249
Korrelation für AskAnAmerican (z-transformiert): 0.450; 0.497


Korrelation für personalfinance (z-transformiert): -0.325
Korrelation für Costco (z-transformiert): 0.123
Korrelation für combined (z-transformiert): 0.079






python scripts/generate_bert_sentiment_data.py --subreddit povertyfinance --invert
python scripts/generate_bert_sentiment_data.py --subreddit food
python scripts/generate_bert_sentiment_data.py --subreddit walmart --invert
python scripts/generate_bert_sentiment_data.py --subreddit frugal --invert
python scripts/generate_bert_sentiment_data.py --subreddit AskAnAmerican --invert



python scripts/generate_sentiment_predictions.py --subreddit walmart --model_dir data/bert_pipeline/povertyfinance_model
python scripts/generate_sentiment_predictions.py --subreddit frugal --model_dir data/bert_pipeline/food_model
python scripts/generate_sentiment_predictions.py --subreddit AskAnAmerican --model_dir data/bert_pipeline/walmart_model



python scripts/supervised_inflation_model.py --subreddit povertyfinance
python scripts/supervised_inflation_model.py --subreddit food
python scripts/supervised_inflation_model.py --subreddit economy


## Comments ergebnisse:
- data/sentiment/sentiment_1_economy.csv: SUBREDDITS = ["economy", "AskAnAmerican", "personalfinance"] und SUBREDDIT = "economy" 

## Project Structure
reddit_inflation_index/
│
├── data/
│   ├── raw/                  ← Raw Reddit data by subreddit
│   │   └── personalfinance/
│   │       └── 2025-04-10.csv
│   └── processed/            ← Cleaned or enriched data (e.g., with sentiment)
│
├── scripts/
│   └── fetch_reddit_posts.py ← Scraper script
│
├── notebooks/               ← Jupyter notebooks (EDA, visualization)
│
├── output/                  ← Final outputs, charts, exportable files
│
└── requirements.txt         ← Python dependencies



## Subreddits:
- r/personalfinance	Sehr aktiv, viele Posts über Preise, Lebenshaltungskosten, Budget, Löhne. (21.2M)
- r/frugal	Nutzer teilen Spar-Tipps und klagen über gestiegene Preise → sehr oft produktbezogen. (6.6M)
- r/povertyfinance	Konkrete Probleme mit Lebenshaltungskosten (Miete, Lebensmittel, Gas, etc.). (2.3M)
- r/AskAnAmerican	Fragen zum Alltag in den USA – häufig Konsum-/Preisthemen enthalten. (1.1M)
- r/Costco	Konkrete Produktpreisbeobachtungen in US-Ketten – oft Preisvergleiche & Beschwerden. (1.3M)
- r/walmart	Diskussionen über Preisveränderungen bei konkreten Produkten in US-Supermärkten. (350k)
- r/economy	Makroökonomischer Fokus – weniger alltagsbezogen, aber nützlich für Kontext. (1M)
- r/food	Klagen über Restaurantpreise, Zutatenkosten, Lieferdienste. (24.4M)

get the data
https://academictorrents.com/details/1614740ac8c94505e4ecb9d88be8bed7b6afddd4