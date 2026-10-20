# NFL Predictor

A neural-network ensemble that outputs **win probabilities** for NFL games, built only on
information available before kickoff. Vegas lines are used as a benchmark, never as an input.

See [REPORT.md](REPORT.md) for results and findings, and [PLAN.md](PLAN.md) for the original design decisions.

## Weekly use

```bash
python src/run_weekly.py
```

This command:

1. Re-downloads the current season from nflverse.
2. Rebuilds the stats, roster ratings and features.
3. Retrains the five models on every completed game.
4. Writes `outputs/picks_<season>_wk<NN>.csv` and `.json`.

Run it after Friday injury reports. Re-run it Sunday morning if a starting QB changes.

To predict a specific week:

```bash
python src/run_weekly.py --no-download --season 2026 --week 6
```

## Site and pick sheet

- **`docs/`** is the GitHub Pages site: this week's model picks and Krish's graded picks.
  - It's rebuilt by `python src/build_site.py`.
  - Krish's picks appear only after kickoff, and notes are never published.
- **The Pick Sheet** is a claude.ai artifact where Krish enters weekly probabilities.
  - `src/pick_sheet.py` builds each week's game context for it.
  - Scheduled Claude tasks refresh it on Tue/Thu/Sun mornings and Friday evening.
- **Model predictions freeze at kickoff.** `predict.py` trains only on weeks before the target week.

## Setup

```bash
pip install -r requirements.txt
```

The parquet/pyarrow route is blocked on this PC by Windows Application Control, so the code reads CSV files instead.

## Pipeline (`src/`)

| File | What it does |
|---|---|
| `download_data.py` | nflverse schedules, play-by-play, weekly rosters, snap counts, depth charts, injuries, player IDs → `data/raw/` |
| `team_games.py` | Per-team per-game stats from play-by-play (EPA, success rate, pass/rush splits, explosives, turnovers, sacks, special teams), plus QB game logs |
| `players.py` | Player ratings (recency-weighted snap share + draft-capital prior) and pregame roster strength by unit, including missing regulars |
| `build_features.py` | Game-level table: short- and long-window form, opponent-adjusted EPA, ridge power ratings, Elo, QB rating and change flags, playoff stakes, rest/travel/body clock, Vegas benchmark |
| `dataset.py` | Turns games into team-1 vs team-2 matrices; mirrors every game so models stay symmetric |
| `models.py` | Logistic regression, ridge-margin, gradient boosting, MLP and Siamese networks (multi-task: win + margin) |
| `seq_model.py` | GRU over each team's last 12 games (tested; not in the final ensemble) |
| `config.py` | The final model lineup |
| `backtest.py`, `experiments.py`, `evaluate.py` | Walk-forward evaluation, the experiment grid (`outputs/experiments.csv`), and the final dev/test evaluation |
| `predict.py`, `run_weekly.py` | Weekly predictions |

## Leakage rules

- Every feature for a game uses only data from before that game's kickoff. Power ratings, standings and Elo are computed as of the first kickoff of that week.
- A player counts as "available" if he is on the 53-man roster and not listed Out or Doubtful. This is the same Friday view for historical games and live picks.
- Models were tuned on the 2018–2021 seasons and scored once on 2022–2026.
