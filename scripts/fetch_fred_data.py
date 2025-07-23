import pandas as pd
import os
import requests

FRED_API_KEY = "a4663ff5b8d4b3022ff51995cdcef6d3"
SERIES_ID = "CPIAUCSL"  # All Urban Consumers, Seasonally Adjusted
OUTPUT_DIR = "../data/raw/fred/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

url = f"https://api.stlouisfed.org/fred/series/observations"
params = {
    "series_id": SERIES_ID,
    "api_key": FRED_API_KEY,
    "file_type": "json"
}

response = requests.get(url, params=params)
data = response.json()["observations"]

# Build DataFrame
df = pd.DataFrame(data)[["date", "value"]]
df["value"] = pd.to_numeric(df["value"], errors="coerce")
df.columns = ["date", "cpi_value"]

# Save as CSV
output_path = os.path.join(OUTPUT_DIR, f"{SERIES_ID}.csv")
df.to_csv(output_path, index=False)

print(f"CPI data saved to {output_path}")