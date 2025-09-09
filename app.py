import streamlit as st
import pandas as pd
import os
import subprocess
import altair as alt
import json
import base64

# ---- SETTINGS ---- #
SENTIMENT_PATH = "data/output/model_2_bert/live_set"
INFLATION_PATH = "data/inflation/usa_inflation.csv"
DEFAULT_SUBREDDITS = ["povertyfinance", "food"]

st.set_page_config(page_title="Reddit Inflation Index", layout="wide")

# ---- TITLE ---- #
st.title("Reddit Inflation Index Dashboard")

st.markdown("""
This dashboard visualizes normalized sentiment trends from Reddit subreddits related to consumer prices.
""")



# ---- NOTICE BAR ---- # (Added notice for current state without update function)
st.warning("""
**Important Notice**: This dashboard reflects the current state of data from the GitHub repository and does not include the update function. 
For the full update functionality (e.g., fetching new Reddit posts and running predictions), please follow the instructions in the GitHub README: 
[https://github.com/christianleis040/reddit_inflation_index/](https://github.com/christianleis040/reddit_inflation_index/).
""")


# ---- SIDEBAR ---- #
with st.sidebar:
    selected_subreddits = st.multiselect("Select subreddits", DEFAULT_SUBREDDITS, default=DEFAULT_SUBREDDITS)
    show_inflation = st.checkbox("Show official US CPI", value=True)
    agg_level = st.selectbox("Aggregation Level", ["daily", "weekly", "monthly"], index=0)

    if st.button("Update Data (CPI & Reddit Sentiment)", key="update_button"):
        with st.spinner("Updating data..."):
            subprocess.run(["python3", "scripts/get_data/fetch_daily_posts.py"])
            subprocess.run(["python3", "scripts/get_data/cpi_data.py"])
            subprocess.run([
                "python3", "scripts/model_2/generate_bert_model_2_predictions_dashboard.py",
                "--subreddits", *selected_subreddits
            ])
        st.success("Data updated successfully!")
        st.rerun()

    st.markdown("### Settings")
    st.markdown("""
    The dashboard uses the two most reliable subreddits identified in extensive experiments: `povertyfinance` (primary) and `food` (as a backup to detect anomalies).

    Official U.S. CPI is included as a benchmark.  
                
    ### Note:
    - To ensure data completeness, posts from the **last 3 days are automatically refreshed** on each update.  
    - Current API limits up to **1000 posts** per call.
    -  **Reddit days are aligned with UTC time**, which may differ from your local time zone.

    <a href="https://github.com/christianleis040/reddit_inflation_index/" target="_blank">
    GitHub Repository: Technical details, full report, and experiments
    </a>
""", unsafe_allow_html=True)

# ---- LOAD DATA ---- #
def load_sentiment(subreddit):
    path = os.path.join(SENTIMENT_PATH, f"bert_full_1_{subreddit}.csv")
    if os.path.exists(path):
        df = pd.read_csv(path)
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date")
        df = df[["date", "avg_sentiment"]].rename(columns={"avg_sentiment": "value"})
        df["source"] = subreddit
        return df
    return pd.DataFrame()

def load_post_counts(subreddit):
    path = os.path.join(SENTIMENT_PATH, f"bert_full_1_{subreddit}.csv")
    if os.path.exists(path):
        df = pd.read_csv(path)
        df["date"] = pd.to_datetime(df["date"])
        df = df[["date", "n_posts"]].rename(columns={"n_posts": "count"})
        df["source"] = subreddit
        return df
    return pd.DataFrame()

def get_latest_reddit_date():
    latest_date = None
    for sub in selected_subreddits:
        path = os.path.join(SENTIMENT_PATH, f"bert_full_1_{sub}.csv")
        if os.path.exists(path):
            df = pd.read_csv(path)
            df["date"] = pd.to_datetime(df["date"])
            max_date = df["date"].max()
            if latest_date is None or max_date > latest_date:
                latest_date = max_date
    return latest_date if latest_date else pd.to_datetime("today")

def load_cpi():
    if os.path.exists(INFLATION_PATH):
        df = pd.read_csv(INFLATION_PATH)
        if "cpi_value" in df.columns:
            df = df.rename(columns={"cpi_value": "value"})
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date")
        df = df.set_index("date").resample("D").ffill()
        last_reddit_date = get_latest_reddit_date()
        if last_reddit_date > df.index[-1]:
            last_value = df.iloc[-1]["value"]
            extension_dates = pd.date_range(df.index[-1] + pd.Timedelta(days=1), last_reddit_date, freq="D")
            extension_df = pd.DataFrame({"value": last_value}, index=extension_dates)
            df = pd.concat([df, extension_df])
        df = df.reset_index(names="date")
        df["source"] = "CPI"
        return df[["date", "value", "source"]]
    return pd.DataFrame()

def aggregate_data(df, agg_level, value_col="value"):
    if agg_level == "daily":
        return df
    df = df.copy()
    if agg_level == "weekly":
        df["date"] = df["date"].dt.to_period("W").apply(lambda r: r.start_time)
    elif agg_level == "monthly":
        df["date"] = df["date"].dt.to_period("M").apply(lambda r: r.start_time)
    agg_df = df.groupby(["date", "source"]).agg({value_col: "mean"}).reset_index()
    return agg_df

# ---- COLLECT & NORMALIZE ---- #
sentiment_dfs = [load_sentiment(sub) for sub in selected_subreddits]
df_sentiment = pd.concat(sentiment_dfs) if sentiment_dfs else pd.DataFrame()
df_cpi = load_cpi() if show_inflation else pd.DataFrame()

if not df_sentiment.empty:
    combined = sentiment_dfs + ([df_cpi] if show_inflation else [])
    anchor_date = max(df["date"].min() for df in combined if not df.empty)
    all_data = pd.concat([df_sentiment, df_cpi]) if not df_cpi.empty else df_sentiment.copy()
    all_data = all_data[all_data["date"] >= anchor_date].copy()

    # Normalize to 100 at anchor_date per source
    def normalize(df):
        base = df[df["date"] == anchor_date]["value"].mean()
        df["normalized"] = df["value"] / base * 100 if base != 0 else df["value"]
        return df

    all_data = all_data.groupby("source", group_keys=False).apply(normalize).reset_index(drop=True)
    all_data = aggregate_data(all_data, agg_level, value_col="normalized")

    chart_trend = alt.Chart(all_data).mark_line(point=True).encode(
        x=alt.X("date:T", title="Date", axis=alt.Axis(format="%Y-%m-%d")),
        y=alt.Y("normalized:Q", title="Normalized Value (Base 100)"),
        color=alt.Color("source:N", title="Source"),
        tooltip=["source", "date:T", alt.Tooltip("normalized", format=".1f")]
    ).properties(
        width=900,
        height=500,
        title=f"{agg_level.capitalize()} Normalized Sentiment & CPI Trends (Base = first Reddit date)"
    )
    st.altair_chart(chart_trend, use_container_width=True)

# ---- SECOND GRAPH: POST COUNTS ---- #
post_count_dfs = [load_post_counts(sub) for sub in selected_subreddits]
df_counts = pd.concat(post_count_dfs) if post_count_dfs else pd.DataFrame()

if not df_counts.empty:
    df_counts = df_counts[df_counts["date"] >= anchor_date].copy()
    df_counts = aggregate_data(df_counts.rename(columns={"count": "value"}), agg_level, value_col="value").rename(columns={"value": "count"})

    chart_counts = alt.Chart(df_counts).mark_line(point=True).encode(
        x=alt.X("date:T", title="Date", axis=alt.Axis(format="%Y-%m-%d")),
        y=alt.Y("count:Q", title="Number of Posts"),
        color=alt.Color("source:N", title="Subreddit"),
        tooltip=["source", "date:T", "count"]
    ).properties(
        width=900,
        height=300,
        title=f"{agg_level.capitalize()} Reddit Post Counts (Data Volume)"
    )
    st.altair_chart(chart_counts, use_container_width=True)

    # ---- POST PREVIEW TABLES ---- #
    st.markdown("### Reddit Post Preview")

    def load_post_json(subreddit):
        path = f"data/live_set/{subreddit}"
        if not os.path.exists(path):
            return pd.DataFrame()

        all_rows = []
        for filename in sorted(os.listdir(path), reverse=True):
            if filename.endswith(".jsonl"):
                with open(os.path.join(path, filename), "r", encoding="utf-8") as f:
                    for line in f:
                        row = json.loads(line)
                        row["date"] = filename.replace(".jsonl", "")
                        all_rows.append(row)
        return pd.DataFrame(all_rows)

    for sub in selected_subreddits:
        st.markdown(f"##### Subreddit: `r/{sub}`")

        df_posts = load_post_json(sub)
        if df_posts.empty:
            st.markdown("_No posts available._")
            continue

        # Create summary column
        def summarize(text):
            return " ".join(text.split()[:30]) + "..." if text else ""

        df_posts["summary"] = df_posts["title"].fillna("") + " — " + df_posts["selftext"].fillna("").apply(summarize)
        df_posts["date"] = pd.to_datetime(df_posts["date"]).dt.date
        
        df_posts = df_posts[df_posts["score"] >= 1]


        # Most recent posts
        st.markdown("**Most Recent Posts**")
        df_recent = df_posts.sort_values("date", ascending=False).head(3)
        st.dataframe(df_recent[["date", "title", "summary", "score", "num_comments"]].rename(
            columns={"score": "upvotes", "num_comments": "comments"}
        ))

        # Top upvoted posts
        st.markdown("**Top Upvoted Posts**")
        df_top = df_posts.sort_values("score", ascending=False).head(3)
        st.dataframe(df_top[["date", "title", "summary", "score", "num_comments"]].rename(
            columns={"score": "upvotes", "num_comments": "comments"}
        ))


