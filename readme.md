## 1. Introduction
This project constructs a **Reddit-based Inflation Index** using public Reddit discussions and machine learning models. The goal is to analyze how user-generated content on subreddits like *r/food* and *r/povertyfinance* reflects and anticipates inflation trends in the United States.

To achieve this, the pipeline implements **three modeling approaches**:

- **Model 1**: A rule-based baseline using VADER sentiment analysis and simple aggregation.
- **Model 2**: A weakly supervised BERT classifier, trained using Reddit post scores as proxy labels for sentiment.
- **Model 3**: A regression model that directly predicts monthly CPI values based on Reddit content, using fine-tuned BERT and VADER as features.

A **live dashboard** built with Streamlit provides an interactive and continuously updated view of sentiment and inflation indicators extracted from Reddit. Historical analysis and model outputs are available for reproduction and extension. For viewing the current state of the dashboard with existing data on GitHub (without the update function), access the following link: https://reddit-inflation-index.streamlit.app

## 2. Technical Setup

This section provides instructions on how to set up the project locally. You will need Python (≥3.9) and `pip`.

### 2.1 Virtual Environment and Dependencies

We recommend using a virtual environment to keep dependencies isolated.

#### Step 1: Create virtual environment

```bash
python3 -m venv venv
```

#### Step 2: Activate the environment
##### macOS / Linux:
```bash
source venv/bin/activate
```

##### Windows (CMD):
```bash
source venv\Scripts\activate.bat
```

##### Windows (PowerShell):
```bash
venv\Scripts\Activate.ps1
```

#### Step 3: Install required dependencies
```bash
pip install -r requirements.txt
```

### 2.2 Get Data

#### 2.2.1 Get Reddit Data

You need to extract relevant Reddit posts from the public Pushshift dumps. This involves two main steps: downloading the `.zst` files and extracting posts using a provided script.

##### 2.2.1.1 Download Reddit Dumps

Visit the following source to download monthly Reddit submission dumps (`RS_*.zst` files):

**Academic Torrents:**  
https://academictorrents.com/details/ba051999301b109eab37d16f027b3f49ade2de13

Download all required `.zst` files (e.g. `RS_2023-09.zst`) and place them in the following folder:

```bash
dumps/
```

---

##### 2.2.1.2 Extract Reddit Posts

To extract relevant posts (based on subreddit, date range, score, etc.), use the script:


```bash
./run_all_extractions.sh
```
Before running the script, edit the following variables at the top of the file. Example:

```bash
subreddits=("food" "povertyfinance")
years=(2012, 2013, ..., 2024)
month="09"
```

Then execute:
```bash
scripts/get_data/run_all_extractions.sh
```
Example output:
```bash
data/raw/food/2023-09.jsonl
```
You can modify filters like minimum score, keywords, or date range in the *fetch_config.json* file before running the script.




#### 2.1.3 Update Inflation Data (CPI)

To include official U.S. inflation data, you can fetch the **Consumer Price Index (CPI)** from the FRED API using:

```bash
python scripts/get_data/cpi_data.py
```

This will download the latest CPI values and store them in:

```
data/inflation/usa_inflation.csv
```

> **Note:** When using the dashboard (`app.py`), this script runs automatically if the CPI file is missing.  
However, for reproducibility or analysis of updated data, you can run it manually at any time.



### 2.3 Model/Results Generation

This section explains how to train the sentiment classification models used in the project.

#### 2.2.1 Model 2 (Dashboard)

To enable the live dashboard, two fine-tuned BERT models are required — one for each of the following subreddits:

- `food`
- `povertyfinance`

These models predict sentiment scores for Reddit posts and are used to visualize sentiment-based inflation trends over time.

To train and save these models, run the following script with the corresponding `--subreddit` argument:

```bash
python scripts/model_2/generate_bert_model_2.py --subreddit food
python scripts/model_2/generate_bert_model_2.py --subreddit povertyfinance
```
This script loads the preprocessed Reddit posts from:
```bash
data/raw/<subreddit>/
```

and saves the trained models to:
```bash
models/model_2/<subreddit>_model/
```
These models are required to run the Streamlit dashboard.

If the sentiment polarity is inverted (i.e., negative correlation), you can optionally pass the --invert flag:

```bash
python scripts/model_2/generate_bert_model_2.py --subreddit food --invert
```


The following steps are only required if you want to **reproduce sentiment models** or **extend the project**, for example by including:

- additional time periods  
- different subreddits  
- adjusted thresholds or preprocessing logic
- ...

If you only want to use the precomputed dashboard (`app.py`), you can skip this section — the required models for `food` and `povertyfinance` are expected in the default paths.

#### 2.3.2 Model 2

If the dashboard models have not yet been trained, run the following commands to generate them:

```bash
python scripts/model_2/generate_bert_model_2.py --subreddit <subreddit>
```

Once the models are trained, you can apply them to all available Reddit posts and generate a monthly sentiment time series using:

```bash
python scripts/model_2/generate_bert_model_2_predictions.py --subreddit <subreddit> --model_dir models/model_2/<subreddit>_model
```

The script processes all `.jsonl` files in:

```
data/raw/<subreddit>/
```

and creates a `.csv` file with average sentiment scores per day:

```
data/output/model_2_bert/full_set/bert_full_<subreddit>.csv
```

This CSV is required to visualize the sentiment index.


#### 2.3.3 Model 3

To train a BERT regression model that predicts the official U.S. **Consumer Price Index (CPI)** from Reddit posts, run:

```bash
python scripts/model_3/bert_model_3.py --subreddit <subreddit>
```

This model uses monthly Reddit posts as input and learns to approximate normalized CPI values based on textual content and sentiment signals.

The model is saved to:

```
models/model_3/<subreddit>/best_model/
```

and full CPI predictions are stored under:

```
data/output/model_3_bert/full_set/
```

This step is optional and only necessary for reproducing or extending the CPI regression analysis (Model 3).



### 2.4 Dashboard
To launch the interactive dashboard locally, run:

```bash
streamlit run app.py
```

This will open the dashboard in your default browser at:

```
http://localhost:8501/
```

The dashboard visualizes sentiment trends over time for selected subreddits and overlays them with official U.S. inflation data (CPI).

## 3. Project Structure

```text
.
├── analysis/                         # Jupyter Notebooks for exploratory analysis
│   └── analysis_notebook.ipynb
├── app.py                            # Streamlit dashboard entry point
├── data/                             # All data folders (raw, processed, output)
│   ├── analytics/                    # Post count summaries per subreddit/month
│   │   ├── post_counts_per_subreddit.csv
│   │   └── post_counts_total.csv
│   ├── inflation/                    # Official CPI data
│   │   └── usa_inflation.csv
│   ├── live_set/                     # Daily live subreddit data for dashboard
│   │   ├── food/
│   │   └── povertyfinance/
│   ├── output/                       # Model outputs (per approach)
│   │   ├── model_1_vader/
│   │   ├── model_2_bert/
│   │   └── model_3_bert/
│   └── raw/                          # Filtered Reddit posts (.jsonl by month)
│       ├── AskAnAmerican/
│       ├── Costco/
│       ├── economy/
│       ├── food/
│       ├── Frugal/
│       ├── personalfinance/
│       ├── povertyfinance/
│       └── walmart/
├── dumps/                            # Downloaded .zst Reddit dumps (from Pushshift)
├── fetch_config.json                 # Filters for Reddit post extraction
├── models/                           # Trained model weights
│   ├── model_2/
│   │   ├── food_model/
│   │   └── povertyfinance_model/
│   └── model_3/
│       ├── food/
│       └── povertyfinance/
├── readme.md                         # This file
├── requirements.txt                  # Python dependencies
├── run_all_extractions.sh            # Bash script to extract Reddit posts from .zst
├── scripts/                          # All data processing and model scripts
│   ├── get_data/
│   │   ├── cpi_data.py
│   │   ├── fetch_daily_posts.py
│   │   └── reddit_data_from_dumps.py
│   ├── model_1/
│   │   ├── combine_subreddits.py
│   │   └── vader_model_1.py
│   ├── model_2/
│   │   ├── generate_bert_model_2_predictions_dashboard.py
│   │   ├── generate_bert_model_2_predictions.py
│   │   └── generate_bert_model_2.py
│   └── model_3/
│       └── bert_model_3.py
└── venv/                             # Virtual environment (optional, gitignored)
```

## 4. Results 
This project evaluates three modeling approaches for constructing a Reddit-based sentiment index aligned with official U.S. inflation data (CPI). The models were trained and tested on historical Reddit data (2012–2024), with evaluation at multiple aggregation levels (quarterly, semi-annual, annual).

### Summary of Findings

- **Model 2 (BERT with score supervision)** delivered the strongest results:
  - For *r/povertyfinance*, correlations with CPI exceeded **r = 0.95** across all aggregation levels, with **MAE ≤ 0.22** and **RMSE ≤ 0.32**.
  - Performance improved further with moderate smoothing (semi-annual average: **r = 0.97**, **MAE = 0.15**, **RMSE = 0.24**).
  - For *r/food*, correlations reached **r ≈ 0.82–0.84**, significantly outperforming the other models.

- **Model 1 (VADER rule-based)** performed reasonably well:
  - On *povertyfinance*, it achieved **r = 0.84–0.94**, with moderate error values.
  - On *food*, performance was weaker (**r ≈ 0.57–0.64**), indicating that more heterogeneous subreddits are harder to model.

- **Model 3 (BERT regression on CPI)** showed the weakest correlation:
  - Only **r ≈ 0.29–0.35** on *povertyfinance*, with large errors (e.g., **MAE ≈ 0.95**, **RMSE ≈ 1.17**).
  - *Food* performed slightly better but still underwhelming compared to Model 2.

### Best Performing Configuration

The combination of:
- **Subreddit**: *r/povertyfinance*
- **Model**: BERT (Model 2, score supervision)

showed the highest predictive power, with **r = 0.97** (Semianual) and minimal error. This configuration was also used in the live dashboard.

> For full evaluation details, visualizations, and error metrics, refer to the [project report](./projekt_report_020925.pdf).