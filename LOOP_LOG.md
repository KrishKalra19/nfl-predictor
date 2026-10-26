# Improvement loop log

Started 2026-10-26. Plan: LOOP_PLAN.md. One entry per item.

## 1. Live prediction log — done
- Added `src/live_log.py`. It's append-only: frozen pre-kickoff predictions for every game since 2026 week 6, plus results. It's stored at `docs/data/live_log.csv`, so it's published and committed by the existing scheduled tasks with no task changes needed.
- `build_site.py` refreshes it on every run.
- Excluded `2026_06_SEA_DEN`: its last stored prediction was recomputed after the game ended, before kickoff freezing existed. All week 7 predictions were confirmed frozen pre-kickoff (the Sunday refresh ran at 13:08 ET and kept earlier numbers for the 1 PM games).
- First reading: 26 graded live games. Model 15–11 (log-loss 0.715), Vegas 17–9 (0.701) on the same games. That's far too few games to mean anything yet.
- Verification: a site-only change, so I ran `build_site.py` and checked the data embeds. The model pipeline is unchanged.

## 2. "Since launch" record on the site — done
- The brief now has a "Live since launch" panel: model vs Vegas on the same live games, with log-loss, the game count, a small-sample caveat and a link to download `data/live_log.csv`.
- Current: 26 games, model 15–11 (0.715), Vegas 17–9 (0.701).
- Verified the render in the browser pane with no console errors.

## 3. Calibration layer — no gain, not adopted
- Built walk-forward ensemble predictions for 2016–2021 (`outputs/oof_ensemble.csv`, `src/exp_calibration.py`). Each dev season is calibrated only from earlier seasons.
- Dev log-loss: raw 0.6227. Platt with intercept 0.6250 (worse: it carries an outdated home-field bias from 2016–17 into later seasons, when home advantage had shrunk). Slope-only temperature 0.6233 (fitted slopes 1.11–1.23 confirm mild under-confidence, but correcting it doesn't help out of sample).

## 4. Stacking weights — no gain, not adopted
- Non-negative log-odds weights fit walk-forward on earlier seasons. Unshrunk weights favor ridge_margin and the Siamese net but swing by season: dev 0.6256. Shrunk toward equal weights: 0.6237 / 0.6231. The flat average wins at 0.6227.
- Takeaway: member strength isn't stable enough across seasons to exploit, so the flat log-odds average stays.

## 5. Weather — no gain, not adopted in models
- `src/weather.py` pulls kickoff-window weather (4 hours) from Open-Meteo: archive for past games, forecast for the next 15 days, indoor games neutral. Cached in `data/raw/weather_games.csv` (3,430 games). It matches nflverse's recorded temperature at r=0.975 and wind at r=0.66.
- Features: shared wind / cold / precipitation, plus per-team "dome team outdoors in the cold" and wind × team passing EPA. `src/exp_features.py` is a reusable A/B harness.
- Dev ensemble: baseline 0.6233 vs weather 0.6234. No member improved meaningfully. Weather shifts totals, not winners.
- Columns stay in features.csv but out of the model feature lists. The data can feed a site touch later.

## 6. Coaching — no gain, not adopted
- Per-team features from the full 1999+ schedule using only earlier games: new-coach flag (first season with the team), log tenure with the team, and career win rate shrunk toward .500 (16 pseudo-games). The win-rate differential correlates 0.27 with margin on its own.
- Dev ensemble: baseline 0.6233 vs coach 0.6234. Mixed members: logreg 0.6244 and MLP 0.6242 were slightly better, GBM 0.6291 and Siamese 0.6263 worse. Coaching quality is already reflected in team form.
- Candidate features are now fenced out of the production feature sets (`NOT_ADOPTED` in `backtest.py`) until they earn a place, so a new column can't silently enter the live model.

## 7. Kicking quality — no gain, not adopted
- `k_fgoe`: the team's latest kicker's field goals made over expected per attempt. Expected comes from a league distance curve, exponentially weighted, shrunk with 15 pseudo-attempts, using earlier games only. The differential correlates only 0.05 with margin.
- Dev ensemble: 0.6232 vs 0.6233 baseline, below the 0.002 adoption bar. Fenced out.

## 8. Kalman-filter team strength — no gain in the model, kept for the site
- `src/kalman.py`: joint Kalman filters over all 32 teams.
  - Margin model: points, with home-field advantage.
  - EPA model: offense and defense latent states, opponent-adjusted by construction.
  - Weekly drift, offseason regression and widening uncertainty.
  - Snapshots are taken per week before any of that week's results are applied, the same convention as the live picks.
- Signal alone (corr of rating difference with margin, 2016–21): Kalman 0.389, tuned ridge 0.392, Elo 0.379. The tuning surface is flat (0.388–0.389), so defaults are kept.
- Dev ensemble: as an addition 0.6240 (worse; redundant with the ridge ratings). As a replacement for the ridge ratings 0.6236 (a tie, but the Siamese net degrades). Not adopted.
- Kept because each rating comes with an uncertainty (±), which suits the power-rankings page (item 13).
- Fixed the A/B harness so swap candidates exclude features in the candidate run.

## 9. QB model upgrade — near miss, not adopted
- `qbx_rating`: QB EPA per play adjusted for each opponent's pregame pass defense, plus QB age at kickoff (centered at 28, with a squared term). Birth dates come from players.csv.
- Signal alone: qbx_rating 0.343 vs the existing qb_rating 0.343 (the opponent adjustment adds nothing). QB age 0.137.
- Dev ensemble: baseline 0.6233. All qbx features 0.6227 (4 of 5 members improve, but −0.0006 is under the 0.002 bar). Age only 0.6235.
- Not adopted per the ground rules. It's the best candidate so far; revisit if a later change makes it cross the bar.
- Backup-QB prior skipped: QB2 snap shares are tiny and noisy, and the existing draft-based prior already covers unknown starters.

## 10. Playoff simulation — done (site feature; model use skipped)
- `src/playoffs.py` simulates the rest of the season 10,000 times (4 s).
  - Game probabilities: frozen model predictions for this week; Kalman margin ratings with their uncertainty for later weeks.
  - Seeding: 4 division winners plus 3 wild cards per conference. Real NFL tiebreakers aren't modeled; ties are broken at random.
- Writes `docs/data/playoff_odds.json`, regenerated on every site build (wrapped so a failure can't break the site).
- Sanity: playoff, division and #1-seed probabilities sum to exactly 14 / 8 / 2.
- Not tested as a model feature: the crude stakes flags already showed zero value, and a fair backtest would need simulations for every historical week. Odds are rendered on the site in item 15.

## 11. "Why this pick" — done
- `src/explain.py`: the ensemble's logistic member is linear in (team A − team B) feature differences, so its log-odds split exactly into per-feature contributions. ~250 features are summed into 9 readable groups (Quarterback, Roster & injuries, Team strength rating, Offense / Defense / Results this season, Recent form, Special teams, Rest & travel) plus Home field. The top 3 are kept with the team each favors and a strength (slight / moderate / strong by log-odds).
- Shared context (week, division game) is excluded: in the symmetric model it doesn't favor either side.
- `predict.py` stores a `why` JSON column per game. The site shows "Grounds" on each case card. Games frozen before this change show none.
- Example, DAL @ PHI: Roster & injuries → DAL (strong), Offense → DAL (strong), Defense → PHI (moderate).
- Checked that predictions are unchanged and deterministic: DAL @ PHI's committed Sunday prediction matches a fresh run to every digit.

## 12. Model spread line — done
- Each case card shows "Line: Model X −n · Vegas Y −m". The model line converts the ensemble win probability to a home spread with the same 13.45-point normal scale used to turn Vegas spreads into probabilities, rounded to the half point. Exactly 50% shows as "Pick'em".
- I didn't use the ridge-margin member's raw margins: heavy regularization (alpha 10000) shrinks them, which would understate every line.
- Week 7 examples: SF −8 (Vegas −2.5), HOU −7.5 (Vegas −10.5), DAL −3.5 (Vegas −4.5).

## 13 + 15. Client list: power rankings with playoff odds — done
- New page `docs/rankings.html` ("Client list" in the nav).
  - Columns: all 32 teams ranked by Kalman margin rating (points vs an average team) with a ± uncertainty bar, movement since last week, record, offense/defense ranks (Kalman EPA states), starting QB, and simulated odds for playoffs, division and #1 seed (item 10).
  - Item 15's playoff odds table lives here rather than on the home page, to keep the docket uncluttered.
- Shared styling moved to `site/style.css`, which is copied to `docs/style.css`. Both pages link it.
- Added `.claude/launch.json` (gitignored) to serve `docs/` locally on port 8765, because file previews can't load relative stylesheets. Verified both pages there: themed, no overflow, no console errors.
- Current top 3: SEA (+7.8 ±4.1), JAX (+6.3), BUF (+4.7). MIN is 6-0 but 6th by rating (29th offense, 1st defense).

## 14. Weekly recap — done
- "Week N in review" section on the home page, generated only from the live log (frozen pre-kickoff numbers) and results. No hand-written narrative.
- Tiles: model record vs Vegas on the same games, Krish's record that week, biggest upset (lowest Vegas probability for the winner), the model's best call and its worst miss (excluding the upset game so tiles don't repeat).
- Week 6: model 7–6, Vegas 8–5, Krish 5–9. Upset ARI over LA 18-16 (Vegas had ARI at 15%). Best call KC over LAC at 80%.

## 16. Polish — done
- K&A seal favicon (`favicon.svg`), plus Open Graph / Twitter tags and a 1200×630 link-preview card (`og.png`, drawn by `src/make_og.py` from the real held-out numbers) for both pages, so a shared link shows a proper card.
- Accessibility: `<main>` landmark on both pages. Contrast checked: muted text on the panel is about 6.6:1. Existing focus outlines were kept.
- Footer on both pages links to the source code and credits nflverse.
- Checked at 375px on the local server: no page-level horizontal scroll; the rankings table scrolls inside its box.
- Skipped the separate print layout: the frame can't print, and the brief already reads cleanly as a page.
