# Roadmap
Ideas, roughly in order. Edit freely – this is a working list, not a promise.

## Next (model quality – do before charging anyone)
- [ ] Backtest on 10+ real seasons; record log loss / calibration per league (the product's evidence).
- [ ] Remove the bookmaker margin properly (proportional / Shin) before computing edge.
- [ ] Team strength features beyond goals: Elo, recent form, xG data if available.
- [ ] Compare against closing odds; track a simulated flat-stake ROI with confidence intervals.
- [x] Optional ML layer (calibrated) on top of the Poisson baseline (v1.1.0) – next: prove it on real multi-season data.
- [ ] Tune the ML model (hyper-parameters, recency weights, more features such as xG / shots / lineups) using the ML Lab comparison.

## Product
- [ ] Automatic data download / update (scheduled).
- [ ] Saved predictions history + results tracking (did the pick win?).
- [ ] League tables, form guides, PDF / image export of a match card.
- [ ] Windows installer (PyInstaller) for non-technical users.

## Going commercial (see docs/MONETIZATION.md)
- [ ] Data licensing sorted for the data you ship or use.
- [ ] User accounts, subscription billing, usage limits.
- [ ] Hosted deployment (Docker) with the model kept server-side.
- [ ] Terms of service, privacy policy, responsible-gambling notices.
- [ ] Prediction API (FastAPI) for partners.
