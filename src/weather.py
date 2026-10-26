"""Kickoff weather for every game from Open-Meteo (free, no key).

Past games use the historical archive; games in the next 16 days use the forecast (refreshed each
run, replaced by archive values once the game is old enough). Indoor games get neutral values.
Cache: data/raw/weather_games.csv (game_id, wx_temp F, wx_wind mph, wx_precip mm, wx_indoor, source).
"""
import time

import pandas as pd
import requests

from common import RAW, kickoff_utc, load_games, venue_geo

CACHE = RAW / "weather_games.csv"
ARCHIVE = "https://archive-api.open-meteo.com/v1/archive"
FORECAST = "https://api.open-meteo.com/v1/forecast"
HOURS = 4  # kickoff hour + next 3
NEUTRAL = {"wx_temp": 70.0, "wx_wind": 0.0, "wx_precip": 0.0}


def indoor_flags(games):
    """Indoor if the roof is dome/closed; unknown roofs (upcoming games) use that stadium's usual roof."""
    usual = (games.dropna(subset=["roof"]).groupby("stadium_id")["roof"]
             .agg(lambda r: r.value_counts().index[0]).to_dict())
    roof = games["roof"].fillna(games["stadium_id"].map(usual))
    return roof.isin(["dome", "closed"])


def _fetch(url, lat, lon, start, end):
    params = {"latitude": round(lat, 3), "longitude": round(lon, 3), "start_date": start, "end_date": end,
              "hourly": "temperature_2m,wind_speed_10m,precipitation", "temperature_unit": "fahrenheit",
              "wind_speed_unit": "mph", "timezone": "UTC"}
    for attempt in range(4):
        r = requests.get(url, params=params, timeout=60)
        if r.status_code == 200:
            h = r.json()["hourly"]
            return pd.DataFrame({"time": pd.to_datetime(h["time"], utc=True), "t": h["temperature_2m"],
                                 "w": h["wind_speed_10m"], "p": h["precipitation"]}).set_index("time")
        time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"open-meteo {r.status_code}: {r.text[:200]}")


def build(games=None, verbose=True):
    games = load_games() if games is None else games
    g = games.copy()
    g["kick"] = [kickoff_utc(d, t) for d, t in zip(g["gameday"], g["gametime"])]
    g["indoor"] = indoor_flags(g).to_numpy()
    geo = [venue_geo(r) for _, r in g.iterrows()]
    g["lat"], g["lon"] = [x[0] for x in geo], [x[1] for x in geo]

    cache = pd.read_csv(CACHE) if CACHE.exists() else pd.DataFrame(columns=["game_id", "source"])
    now = pd.Timestamp.now(tz="UTC")
    archive_ok = now - pd.Timedelta(days=6)  # archive lags a few days
    keep = cache[cache["source"].isin(["archive", "indoor"])]
    done = set(keep["game_id"])
    todo = g[~g["game_id"].isin(done) & (g["kick"] < now + pd.Timedelta(days=15))]

    rows = []
    for gid, ind in zip(todo["game_id"], todo["indoor"]):
        if ind:
            rows.append({"game_id": gid, **NEUTRAL, "wx_indoor": 1, "source": "indoor"})
    out = todo[~todo["indoor"]]
    for (lat, lon), grp in out.groupby([out["lat"].round(3), out["lon"].round(3)]):
        for use_archive, sub in ((True, grp[grp["kick"] < archive_ok]), (False, grp[grp["kick"] >= archive_ok])):
            if sub.empty:
                continue
            # one request per contiguous block of a season keeps payloads small
            for season, ss in sub.groupby(sub["kick"].dt.year):
                start = (ss["kick"].min() - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
                end = (ss["kick"].max() + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
                if use_archive:
                    hw = _fetch(ARCHIVE, lat, lon, start, end)
                else:
                    end = min(pd.Timestamp(end, tz="UTC"), now + pd.Timedelta(days=15)).strftime("%Y-%m-%d")
                    start = max(pd.Timestamp(start, tz="UTC"), now - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
                    hw = _fetch(FORECAST, lat, lon, start, end)
                for gid, k in zip(ss["game_id"], ss["kick"]):
                    win = hw.loc[k.floor("h"): k.floor("h") + pd.Timedelta(hours=HOURS - 1)]
                    if win.empty or win["t"].isna().all():
                        continue
                    rows.append({"game_id": gid, "wx_temp": float(win["t"].mean()), "wx_wind": float(win["w"].mean()),
                                 "wx_precip": float(win["p"].sum()), "wx_indoor": 0,
                                 "source": "archive" if use_archive else "forecast"})
        if verbose:
            print(f"  weather {lat},{lon}: {len(grp)} games", flush=True)
    new = pd.DataFrame(rows)
    res = pd.concat([keep, new], ignore_index=True).drop_duplicates("game_id", keep="last")
    res.to_csv(CACHE, index=False)
    print(f"weather: {len(res)} games cached ({len(new)} fetched)")
    return res


if __name__ == "__main__":
    build()
