"""Game-day inactives from ESPN's public game feed (free, no key).

About 90 minutes before kickoff, ESPN flips injured players who won't dress to "Inactive" in each
game's injury list. Those players are added to the injury report's Out list, so the roster features
see them exactly like a Friday "Out". Only games that haven't kicked off are fetched.

    python src/gameday.py      # writes data/raw/gameday_inactives.csv for the upcoming week
"""
import pandas as pd
import requests

from common import RAW, kickoff_utc, load_games, norm_team, upcoming_week

URL = "https://site.api.espn.com/apis/site/v2/sports/football/nfl/summary?event={}"
OUT = RAW / "gameday_inactives.csv"
COLS = ["season", "week", "team", "gsis_id", "name", "fetched_utc"]


def fetch(espn_id):
    s = requests.get(URL.format(int(espn_id)), timeout=30).json()
    rows = []
    for t in s.get("injuries", []):
        team = norm_team(t["team"]["abbreviation"].replace("WSH", "WAS"))
        for i in t.get("injuries", []):
            fs = (i.get("details", {}).get("fantasyStatus") or {}).get("abbreviation", "")
            if fs == "INACTIVE" or i.get("status") == "Out":
                rows.append((team, str(i["athlete"].get("id")), i["athlete"].get("displayName")))
    return rows


def main(now=None):
    games = load_games()
    season, week = upcoming_week(games, now)
    wk = games[(games["season"] == season) & (games["week"] == week)]
    now = pd.Timestamp.now(tz="UTC") if now is None else now
    ids = pd.read_csv(RAW / "players.csv", usecols=["gsis_id", "espn_id"]).dropna()
    ids["espn_id"] = ids["espn_id"].astype(float).astype(int).astype(str)
    espn2gsis = ids.drop_duplicates("espn_id").set_index("espn_id")["gsis_id"]
    old = pd.read_csv(OUT) if OUT.exists() else pd.DataFrame(columns=COLS)
    new = []
    for g in wk.itertuples():
        if kickoff_utc(g.gameday, g.gametime) <= now or pd.isna(g.espn):
            continue  # started games keep whatever was known before kickoff
        try:
            for team, eid, name in fetch(g.espn):
                if eid in espn2gsis.index:
                    new.append((season, week, team, espn2gsis[eid], name, f"{now:%Y-%m-%d %H:%M}"))
        except (requests.RequestException, ValueError, KeyError) as e:
            print(f"  {g.game_id}: ESPN fetch failed ({e})")
    new = pd.DataFrame(new, columns=COLS)
    keep = old[~((old["season"] == season) & (old["week"] == week) & old["team"].isin(new["team"]))]
    out = pd.concat([keep, new], ignore_index=True).drop_duplicates(["season", "week", "team", "gsis_id"])
    out.to_csv(OUT, index=False)
    print(f"gameday inactives: {len(new)} players for {season} week {week}")
    return out


if __name__ == "__main__":
    main()
