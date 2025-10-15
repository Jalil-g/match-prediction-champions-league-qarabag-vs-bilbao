from fbref_data_downloader import FBRefDownloader, clean_matchlog
import pandas as pd

downloader = FBRefDownloader(delay_range=(3, 7))

teams = {
    'Qarabag-FK': '44b65410',
    'Athletic-Club-Bilbao': '2b390eca'
}

seasons = ['2025-2026']

all_dfs = []

for team, tid in teams.items():
    df = downloader.download_matchlogs(team, tid, seasons)
    if df is not None:
        cleaned = clean_matchlog(df)
        all_dfs.append(cleaned)

if all_dfs:
    combined = pd.concat(all_dfs, ignore_index=True)
    combined.to_csv("data/qarabag_bilbao_2025_2026.csv", index=False)
    print(f"✅ Saved data/qarabag_bilbao_2025_2026.csv with {len(combined)} rows")
else:
    print("⚠️ No new data downloaded.")
