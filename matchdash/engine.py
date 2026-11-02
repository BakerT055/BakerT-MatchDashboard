"""Single entry point used by the app: choose Poisson, ML or a blend, always returning a model.Prediction."""
from __future__ import annotations
from dataclasses import replace
from typing import Optional
from . import ml, model
from .config import Settings


def predict_any(snap: model.Snapshot, bundle: Optional["ml.MLBundle"], s: Settings, home: str, away: str,
                league: Optional[str], date=None, odds: Optional[dict] = None) -> model.Prediction:
    base = model.predict(snap, home, away, league)
    eng = s.engine
    if eng == "Poisson" or bundle is None:
        return base
    out = ml.ml_predict(bundle, home, away, league, date, odds)
    w = 1.0 if eng == "ML" else float(s.ml_blend)
    m = base.markets
    mix = lambda a, b: w * a + (1 - w) * b
    lam_h, lam_a = mix(out["lam_h"], base.lam_h), mix(out["lam_a"], base.lam_a)
    g = model.fit_grid(model.make_grid(lam_h, lam_a, s), mix(out["pH"], m["home"]), mix(out["pD"], m["draw"]), mix(out["pA"], m["away"]),
                       mix(out["pO"], m["o2.5"]), mix(out["pB"], m["btts_y"]))
    sh, sa = base.ht_share
    probs, top = model.markets_from_grid(g, lam_h, lam_a, sh, sa)
    warns = list(base.warnings)
    if bundle.meta.get("use_odds") and not (odds and odds.get("oh")):
        warns.append("The ML model was trained with bookmaker odds as features but no odds were supplied for this match – accuracy is reduced.")
    return replace(base, lam_h=lam_h, lam_a=lam_a, grid=g, markets=probs, top_scores=top, warnings=warns,
                   source="ML" if eng == "ML" else f"Blend ({w:.0%} ML)")
