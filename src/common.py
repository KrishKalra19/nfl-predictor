"""Shared paths, team normalization, geography and raw-data loaders."""
import math
import os
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
# NFL_PROCESSED / NFL_FIRST_SEASON redirect experiment builds (e.g. pre-2015 history) away from the live tables
PROCESSED = Path(os.environ.get("NFL_PROCESSED", ROOT / "data" / "processed"))
OUTPUTS = ROOT / "outputs"

FIRST_SEASON = int(os.environ.get("NFL_FIRST_SEASON", 2014))  # pulled only to seed priors / player ratings
FIRST_MODEL_SEASON = int(os.environ.get("NFL_FIRST_MODEL_SEASON", 2015))  # first season in the modeling table
CURRENT_SEASON = 2026

# Relocated franchises -> one continuous code so team history carries over.
FRANCHISE = {"STL": "LA", "SD": "LAC", "OAK": "LV", "LAR": "LA",
             # legacy codes used in 2014-2015 roster files
             "ARZ": "ARI", "BLT": "BAL", "CLV": "CLE", "HST": "HOU", "SL": "LA"}


def norm_team(code):
    return FRANCHISE.get(code, code)


# (lat, lon, timezone offset from US Eastern). Arizona doesn't observe DST, so -2.5 averages the season.
_TEAM_GEO = {
    "ARI": (33.528, -112.263, -2.5), "ATL": (33.755, -84.401, 0), "BAL": (39.278, -76.623, 0),
    "BUF": (42.774, -78.787, 0), "CAR": (35.226, -80.853, 0), "CHI": (41.862, -87.617, -1),
    "CIN": (39.095, -84.516, 0), "CLE": (41.506, -81.700, 0), "DAL": (32.748, -97.093, -1),
    "DEN": (39.744, -105.020, -2), "DET": (42.340, -83.046, 0), "GB": (44.501, -88.062, -1),
    "HOU": (29.685, -95.411, -1), "IND": (39.760, -86.164, 0), "JAX": (30.324, -81.637, 0),
    "KC": (39.049, -94.484, -1), "LA": (33.953, -118.339, -3), "LAC": (33.953, -118.339, -3),
    "LV": (36.091, -115.184, -3), "MIA": (25.958, -80.239, 0), "MIN": (44.974, -93.258, -1),
    "NE": (42.091, -71.264, 0), "NO": (29.951, -90.081, -1), "NYG": (40.813, -74.074, 0),
    "NYJ": (40.813, -74.074, 0), "PHI": (39.901, -75.168, 0), "PIT": (40.447, -80.016, 0),
    "SEA": (47.595, -122.332, -3), "SF": (37.403, -121.970, -3), "TB": (27.976, -82.503, 0),
    "TEN": (36.166, -86.771, -1), "WAS": (38.908, -76.864, 0),
}


def team_geo(team, season):
    """Home-stadium (lat, lon, tz) for a franchise in a given season."""
    if team == "LA" and season <= 2015:
        return (38.633, -90.188, -1)      # St. Louis
    if team == "LAC" and season <= 2016:
        return (32.783, -117.120, -3)     # San Diego
    if team == "LV" and season <= 2019:
        return (37.751, -122.201, -3)     # Oakland
    return _TEAM_GEO[team]


# Neutral-site venues by stadium_id prefix (international + domestic neutral games).
_VENUE_GEO = {
    "LON": (51.53, -0.17, 5), "FRA": (50.07, 8.65, 6), "GER": (48.22, 11.62, 6), "MUN": (48.22, 11.62, 6),
    "MAD": (40.45, -3.69, 6), "PAR": (48.92, 2.36, 6), "MEL": (-37.82, 144.98, 15), "MEX": (19.30, -99.15, -1),
    "SAO": (-23.55, -46.47, 1), "RIO": (-22.91, -43.23, 1), "DUB": (53.36, -6.25, 5), "BER": (52.51, 13.24, 6),
    "ATL": _TEAM_GEO["ATL"], "CLE": _TEAM_GEO["CLE"], "DET": _TEAM_GEO["DET"], "HOU": _TEAM_GEO["HOU"],
    "IND": _TEAM_GEO["IND"], "JAX": _TEAM_GEO["JAX"], "LAX": _TEAM_GEO["LA"], "MIA": _TEAM_GEO["MIA"],
    "MIN": _TEAM_GEO["MIN"], "NOR": _TEAM_GEO["NO"], "NYC": _TEAM_GEO["NYG"], "PHO": _TEAM_GEO["ARI"],
    "PIT": _TEAM_GEO["PIT"], "SFO": _TEAM_GEO["SF"], "TAM": _TEAM_GEO["TB"], "VEG": _TEAM_GEO["LV"],
}


def venue_geo(row):
    """Location of the game itself."""
    if row["location"] == "Home" or not isinstance(row.get("stadium_id"), str):
        return team_geo(row["home_team"], row["season"])
    return _VENUE_GEO.get(row["stadium_id"][:3], team_geo(row["home_team"], row["season"]))


def haversine_miles(a, b):
    lat1, lon1, lat2, lon2 = map(math.radians, (a[0], a[1], b[0], b[1]))
    h = math.sin((lat2 - lat1) / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin((lon2 - lon1) / 2) ** 2
    return 3958.8 * 2 * math.asin(math.sqrt(h))


# ---------------------------------------------------------------- loaders

def _season_file(folder, stem, season):
    for ext in (".csv.gz", ".csv"):
        f = RAW / folder / f"{stem}_{season}{ext}"
        if f.exists():
            return f
    return None


def load_per_season(folder, stem, seasons, usecols=None):
    frames = []
    for s in seasons:
        f = _season_file(folder, stem, s)
        if f is not None:
            frames.append(pd.read_csv(f, usecols=usecols, low_memory=False))
    return pd.concat(frames, ignore_index=True)


def load_games():
    g = pd.read_csv(RAW / "games.csv")
    g = g[g.season >= FIRST_SEASON].copy()
    for c in ("home_team", "away_team"):
        g[c] = g[c].map(norm_team)
    g["gameday"] = pd.to_datetime(g["gameday"])
    return g.sort_values(["gameday", "gametime", "game_id"]).reset_index(drop=True)


# ---------------------------------------------------------------- kickoff times

def kickoff_utc(gameday, gametime):
    """Kickoff as a timezone-aware UTC timestamp (schedule times are US Eastern)."""
    from zoneinfo import ZoneInfo
    t = gametime if isinstance(gametime, str) and ":" in gametime else "13:00"
    local = pd.Timestamp(f"{pd.Timestamp(gameday):%Y-%m-%d} {t}").tz_localize(ZoneInfo("America/New_York"))
    return local.tz_convert("UTC")


def upcoming_week(games, now=None):
    """(season, week) of the next game that hasn't kicked off yet, by the clock (not by missing scores,
    since upstream scores can lag a day or two)."""
    now = pd.Timestamp.now(tz="UTC") if now is None else now
    ko = [kickoff_utc(d, t) for d, t in zip(games["gameday"], games["gametime"])]
    g = games.assign(kickoff=ko)
    g = g[g["kickoff"] > now].sort_values("kickoff")
    r = g.iloc[0]
    return int(r["season"]), int(r["week"])
