"""
Downloads match logs for multiple teams from FBRef and
creates a combined dataset ready for ML model training.
"""

import pandas as pd
import requests
from bs4 import BeautifulSoup
import time, os
import random

class FBRefDownloader:
    def __init__(self, delay_range=(2, 6)):
        self.headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
        self.delay_range = delay_range
        os.makedirs("data", exist_ok=True)

    def safe_get(self, url):
        """Make a polite request with rate-limit handling."""
        try:
            response = requests.get(url, headers=self.headers)
            if response.status_code == 429:
                print("⚠️ Rate limited (429). Waiting 60 seconds...")
                time.sleep(60)
                return self.safe_get(url)
            elif response.status_code != 200:
                print(f"❌ Error {response.status_code} fetching {url}")
                return None
            else:
                # polite random delay to avoid hitting rate limits
                time.sleep(random.uniform(*self.delay_range))
                return response
        except Exception as e:
            print(f"⚠️ Request failed: {e}")
            time.sleep(10)
            return None

    def download_matchlogs(self, team_name, team_id, seasons):
        """Download match logs for each season unless already cached."""
        file_path = f"data/{team_name.lower().replace(' ', '-')}_matchlogs.csv"
        if os.path.exists(file_path):
            print(f"📂 Cached file found for {team_name}, skipping download.")
            try:
                return pd.read_csv(file_path)
            except Exception as e:
                print(f"⚠️ Could not read {file_path}: {e}")
                return None

        print(f"\n📥 Fetching {team_name} match logs ...")
        all_logs = []

        for season in seasons:
            url = f"https://fbref.com/en/squads/{team_id}/{season}/matchlogs/all_comps/schedule/{team_name}-Scores-and-Fixtures-All-Competitions"
            resp = self.safe_get(url)
            if not resp:
                continue

            try:
                tables = pd.read_html(resp.content)
                if not tables:
                    print(f"⚠️ No table found for {team_name} {season}")
                    continue

                df = tables[0]
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = ["_".join(col).strip() for col in df.columns]

                df['Season'] = season
                df['Team'] = team_name
                all_logs.append(df)

                print(f"✅ {team_name} {season}: {len(df)} matches")

            except Exception as e:
                print(f"❌ Error parsing {team_name} {season}: {e}")

        if all_logs:
            combined = pd.concat(all_logs, ignore_index=True)
            combined.to_csv(file_path, index=False)
            print(f"💾 Saved {file_path}")
            return combined
        else:
            print(f"⚠️ No data for {team_name}")
            return None


def clean_matchlog(df):
    """Standardize numeric columns and outcomes."""
    useful_cols = ['Date', 'Comp', 'Venue', 'Result', 'GF', 'GA', 'xG', 'xGA', 'Poss', 'Opponent', 'Team', 'Season']
    df = df[[c for c in useful_cols if c in df.columns]].copy()
    for col in ['GF', 'GA', 'xG', 'xGA', 'Poss']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df['GoalDiff'] = df['GF'] - df['GA']
    df['xGDiff'] = df['xG'] - df['xGA']
    df['Outcome'] = df.apply(lambda r: 'Win' if r['GF'] > r['GA'] else ('Draw' if r['GF']==r['GA'] else 'Loss'), axis=1)
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    return df.dropna(subset=['Date'])

def merge_teams(dfs):
    """Combine all team datasets into one CSV."""
    combined = pd.concat(dfs, ignore_index=True)
    combined.to_csv("data/all_teams_training_data.csv", index=False)
    print(f"\n💾 Saved combined dataset: data/all_teams_training_data.csv ({len(combined)} matches)")
    return combined

# ----------------------------------------------------------------

if __name__ == "__main__":
    downloader = FBRefDownloader(delay_range=(2, 6))

    teams = {
    # 🇪🇸 Spain – La Liga
    'Real-Madrid': '53a2f082',
    'Barcelona': '206d90db',
    'Atletico-Madrid': 'db3b9613',
    'Athletic-Club-Bilbao': '2b390eca',
    'Real-Sociedad': 'e4a775cb',
    'Villarreal': '2a8183b3',
    'Real-Betis': 'f25da7fb',
    'Sevilla': 'ad2be733',
    'Osasuna': '03c57e2b',
    'Getafe': '7848bd64',
    'Las-Palmas': 'efcc0f5b',

    # 🇩🇪 Germany – Bundesliga
    'Bayern-Munich': '054efa67',
    'Borussia-Dortmund': 'add600ae',
    'Bayer-Leverkusen': 'c7a9f859',
    'Eintracht-Frankfurt': 'f0ac8ee6',

    # 🇮🇹 Italy – Serie A
    'Inter-Milan': 'd609edc0',
    'Juventus': 'e0652b02',
    'Atalanta': '922493f3',
    'Napoli': 'd48ad4ff',

    # 🏴 England – Premier League
    'Arsenal': '18bb7c10',
    'Manchester-City': 'b8fd03ef',
    'Tottenham-Hotspur': '361ca564',
    'Chelsea': 'cff3d9bb',
    'Liverpool': '822bd0ba',
    'Newcastle-United': 'b2b47a98',

    # 🇫🇷 France – Ligue 1
    'PSG': 'e2d8892c',
    'Marseille': '5725cc7b',
    'Monaco': 'c802d753',

    # 🇵🇹 Portugal
    'Benfica': 'e8e6e29f',
    'Sporting-CP': '2a609ed9',

    # 🇳🇱 Netherlands
    'Ajax': '19c3f8c4',
    'PSV': 'e65c8bee',

    # 🇧🇪 Belgium
    'Club-Brugge': '07549d0d',
    'Union-Saint-Gilloise': 'a75d5f0b',

    # 🇬🇷 Greece
    'Olympiacos': '29a9c5f6',

    # 🇩🇰 / 🇨🇭 / 🇨🇿 / 🇨🇾
    'Copenhagen': '6135e56d',
    'Slavia-Praha': 'd80d4b29',
    'Pafos': 'f6d9e1b8',

    # 🇦🇿 Azerbaijan
    'Qarabag-FK': '44b65410',

    # 🇹🇷 Turkey
    'Galatasaray': 'ecd11ca2',
    'Fenerbahce': '4723d4b7',
    'Besiktas': '26790c6c',

    # 🇰🇿 Kazakhstan
    'Kairat': '768fb565',

    # 🇭🇺 Hungary
    'Ferencvaros': 'b301d8d5',

    # 🇧🇬 Bulgaria
    'Ludogorets': 'c8669c51',

    # 🇸🇰 Slovakia
    'Slovan-Bratislava': '2f0b111a',

    # 🇲🇩 Moldova
    'Sheriff-Tiraspol': '5dbe9b87'
}


    seasons = ['2023-2024', '2024-2025']
    all_dfs = []

    for team, tid in teams.items():
        df = downloader.download_matchlogs(team, tid, seasons)
        if df is not None:
            all_dfs.append(clean_matchlog(df))

    if all_dfs:
        combined = merge_teams(all_dfs)
        print("✅ All data successfully downloaded and merged.")
    else:
        print("⚠️ No valid datasets were downloaded.")
