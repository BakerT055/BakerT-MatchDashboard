# ⚽ BakerT MatchDashboard — Community Edition

The Excel prediction dashboard rebuilt as a Python app: same Poisson logic, no row limits, and everything customisable.

## Run it
**Windows:** double-click `run.bat` (creates a virtual environment, installs requirements, opens the app in your browser).
**Mac/Linux:** `./run.sh`  ·  or manually: `pip install -r requirements.txt` then `streamlit run app.py`.

The app starts with **sample data** (fictional teams). Add your own in the **Data** tab; it is saved to `data/` and reloaded next time.

## Your data
* **Results:** football-data.co.uk files work as-is – CSVs or the `all-euro-data-YYYY-YYYY.xlsx` workbooks (one sheet per league). Upload many seasons at once, or point to a folder. Generic files work too: `Date, League, HomeTeam, AwayTeam, FTHG, FTAG, HTHG, HTAG` (+ optional odds).
* **Fixtures:** `Date, League (Div), HomeTeam, AwayTeam` + optional odds. Football-data's `fixtures.csv` works as-is.
* **Odds columns** recognised: Bet365 (`B365H/D/A`, `B365>2.5`, `B365<2.5`), Pinnacle, Average, Max, and generic `Home odds`, `Draw odds`, `Away odds`, `Over 2.5 odds`, `Under 2.5 odds`, `BTTS Yes odds`, `BTTS No odds`. Choose the source order in the sidebar.
* **Team names** differing between files: add mappings to `data/team_aliases.json`. The Data tab lists fixture teams not found in History, with suggestions.

## Tabs
| Tab | What it does |
|---|---|
| **Dashboard** | Pick a fixture (or any two teams). xG, every market with fair odds and colour-coded signal, top-10 strongest / 5 least likely scenarios, HT/FT, correct scores, scoreline heat-map, **value check vs bookmaker odds**, history check, **recent head-to-head** (all seasons). Odds can be typed in or overridden. |
| **All fixtures** | Predictions for every fixture at once, colour-coded, value flags, strongest scenarios across all fixtures, **CSV download** (timestamped). |
| **Backtest** | Walk-forward test (fit on the past, score the future): log loss, Brier, accuracy vs a reference and calibration tables. |
| **ML Lab** | Train the gradient-boosted ML model and compare it walk-forward with Poisson and bookmaker odds. Then pick the engine (Poisson / ML / Blend) in the sidebar. |
| **Data** | Upload / folder load, summary, team-name checks. No row limit. |

## Model (same as the Excel workbook)
Each team's home/away scoring and conceding rates are compared with the league average (small samples shrunk toward it) to give expected goals; independent Poisson goals give a scoreline grid, from which all markets are derived. Half-time uses each team's league-wide first-half goal share; HT/FT assumes independent halves. Fair odds = 1/probability.
**New, optional (off by default so results equal Excel):** recency half-life weighting and a Dixon-Coles low-score correction. Head-to-head always uses all history.

## Machine-learning engine
Gradient-boosted trees learn from features built **only from information available before each match** (Elo ratings, recent and venue-specific form, rest days, head-to-head, league averages; optionally bookmaker-implied probabilities). Their output is fitted back into the scoreline grid so every market stays consistent. Choose **Poisson**, **ML** or **Blend** in the sidebar.

**Be realistic.** Football is hard to predict and bookmaker odds already pool vast information; a model that merely matches them is a good result. Needs thousands of matches (several seasons, ideally several leagues). The ML Lab tab tells you, with a paired significance test, whether ML or a blend really beats Poisson on *your* data. If it doesn't, keep Poisson. Results on the bundled sample data are not meaningful (the sample is synthetic).

## Customise
* **Sidebar** – shrinkage, history window, half-life, rho, grid size, edge threshold, colour thresholds, odds source. "Save" writes `settings.json`.
* **`matchdash/config.py`** – colour tiers, dashboard panels, column-name aliases, odds sources.
* **`matchdash/markets.py`** – add a market in one line, e.g.
  `add("home_o15", "Home win & Over 1.5", "Combos", lambda c: _p(c, (c.I > c.J) & (c.I + c.J > 1.5)))`
  then list its group in `config.PANELS` to show it.
* **`matchdash/model.py`** – the prediction function; `predict(snapshot, home, away, league)` can be imported by other projects.

## Tests
`python tests/test_core.py` – checks the engine against the Excel workbook's numbers, H2H, options, backtest.
`python tests/test_ml.py` – ML layer: leakage check, grid fitting, engines, walk-forward comparison.
`python tests/test_app_smoke.py` – runs `app.py` against a stand-in for Streamlit to catch code errors.

## Caveats
Plain Poisson: no injuries, line-ups or motivation. A model "edge" versus bookmaker odds is a hypothesis to test (use the Backtest tab), not a guarantee. Edge is computed before removing the bookmaker margin.

## Docs
* [`docs/GIT_GUIDE.md`](docs/GIT_GUIDE.md) – put this project on GitHub, step by step
* [`docs/MONETIZATION.md`](docs/MONETIZATION.md) – what must be in place before charging, and the options
* [`ROADMAP.md`](ROADMAP.md) · [`CHANGELOG.md`](CHANGELOG.md)

## Community and Pro editions

This repository is the **Community Edition**, released under the MIT licence. It includes the statistical engines, gradient-boosted ML predictions, basic walk-forward evaluation and core dashboard.

The Pro edition is intended to be a separately distributed commercial extension. Planned Pro capabilities include feature importance, paired significance testing, parameter sweeps, advanced diagnostics and expanded reporting. Pro code and server-side licence verification are **not included in this repository**.

The Community UI hides Pro-roadmap controls. This is a product/UI distinction, **not a security boundary**: users who run or modify open-source code can inspect and change it. A commercial hosted Pro service must enforce entitlements server-side.

## Licence & disclaimer

Community source code is available under the MIT licence (see [`LICENSE`](LICENSE)). Third-party dependencies retain their own licences. Predictions are statistical estimates for information and entertainment, not betting advice, and carry no guarantee. 18+ only; gamble responsibly.
