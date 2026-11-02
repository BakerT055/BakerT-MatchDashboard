"""BakerT MatchDashboard – run with:  streamlit run app.py"""
from __future__ import annotations
from dataclasses import asdict, replace
from datetime import date, datetime

import numpy as np
import pandas as pd
import streamlit as st

from matchdash import __version__, backtest, config, data, engine, entitlements, h2h, ml, model, ui
from matchdash.config import ODDS_LABELS, ODDS_PREF_CHOICES, PANELS, Settings
from matchdash.markets import MARKET_BY_KEY

st.set_page_config(page_title="BakerT MatchDashboard", page_icon="⚽", layout="wide")


# ------------------------------------------------------------------ helpers
def show(obj, **kw):
    """st.dataframe across Streamlit versions (width='stretch' is new, use_container_width is old)."""
    hide = kw.pop("hide_index", True)
    try:
        st.dataframe(obj, width="stretch", hide_index=hide, **kw)
    except TypeError:
        st.dataframe(obj, use_container_width=True, hide_index=hide, **kw)


def init_state():
    ss = st.session_state
    if "history" not in ss:
        h = data.load_saved("history")
        f = data.load_saved("fixtures")
        ss.sample = h is None
        if h is None:
            h = data.load_history([config.DATA_DIR / "sample_history.csv"])
            f = data.load_fixtures([config.DATA_DIR / "sample_fixtures.csv"])
        ss.history, ss.fixtures = h, (f if f is not None else pd.DataFrame())
        ss.data_version = 1
        ss.snaps = {}
        ss.ml = ml.MLBundle.load() if ml.ml_available() else None


def get_snapshot(s: Settings) -> model.Snapshot:
    ss = st.session_state
    key = (ss.data_version, s.model_key())
    if key not in ss.snaps:
        ss.snaps.clear()
        ss.snaps[key] = model.Snapshot(ss.history, s)
    return ss.snaps[key]


def sidebar_settings() -> Settings:
    base = Settings.load()
    sb = st.sidebar
    sb.title("⚽ BakerT MatchDashboard")
    sb.caption(f"{entitlements.EDITION} edition")
    sb.caption("Sample data is loaded until you add your own (Data tab)." if st.session_state.get("sample") else "Using your saved data.")
    sb.subheader("Model")
    shrink = sb.number_input("Shrinkage (pseudo-games)", 0.0, 100.0, float(base.shrinkage), 1.0,
                             help="Pulls small samples toward the league average. Higher = smoother.")
    limit = sb.checkbox("Limit team-form history from a date", value=bool(base.history_from))
    hist_from = ""
    if limit:
        d0 = pd.to_datetime(base.history_from).date() if base.history_from else date(2020, 8, 1)
        hist_from = sb.date_input("History from", d0).isoformat()
    half = sb.number_input("Recency half-life (days, 0 = off)", 0.0, 5000.0, float(base.half_life_days), 30.0,
                           help="A match this many days old counts half as much as one played today.")
    only = sb.checkbox("Only use the fixture's league", value=bool(base.only_fixture_league))
    rho = sb.slider("Dixon-Coles rho (0 = plain Poisson)", -0.30, 0.30, float(base.dc_rho), 0.01)
    maxg = int(sb.number_input("Max goals per team in grid", 6, 15, int(base.max_goals), 1))
    sb.subheader("Flags & colours")
    edge = sb.slider("Min edge for VALUE", 0.0, 0.30, float(base.edge_min), 0.01, format="%.2f")
    minG = int(sb.number_input("Low-sample warning (games)", 0, 100, int(base.min_games_warn), 1))
    with sb.expander("Colour tier thresholds"):
        t1 = st.slider("Very strong ≥", 0.0, 1.0, float(base.tier_very_strong), 0.01)
        t2 = st.slider("Strong ≥", 0.0, 1.0, float(base.tier_strong), 0.01)
        t3 = st.slider("Lean ≥", 0.0, 1.0, float(base.tier_lean), 0.01)
        t4 = st.slider("Weak ≥", 0.0, 1.0, float(base.tier_weak), 0.01)
    with sb.expander("Odds source preference"):
        prefs = st.multiselect("Order of preference", ODDS_PREF_CHOICES, default=[p for p in base.odds_preference if p in ODDS_PREF_CHOICES])
    h2h_rows = int(sb.number_input("Recent H2H rows", 3, 30, int(base.h2h_rows), 1))
    sb.subheader("Prediction engine")
    has_ml = st.session_state.get("ml") is not None
    engine_name = sb.radio("Engine", ["Poisson", "ML", "Blend"], index=["Poisson", "ML", "Blend"].index(base.engine if base.engine in ("Poisson", "ML", "Blend") else "Poisson"),
                           help="ML and Blend need a trained model (ML Lab tab). Only switch if the ML Lab comparison shows a real gain.")
    ml_blend = base.ml_blend
    if engine_name == "Blend":
        ml_blend = sb.slider("ML weight in blend", 0.0, 1.0, float(base.ml_blend), 0.05)
    if engine_name != "Poisson" and not has_ml:
        sb.warning("No trained ML model yet – using Poisson. Train one in the ML Lab tab.")
    s = replace(base, shrinkage=shrink, history_from=hist_from, half_life_days=half, only_fixture_league=only, dc_rho=rho,
                max_goals=maxg, edge_min=edge, min_games_warn=minG, tier_very_strong=t1, tier_strong=t2, tier_lean=t3,
                tier_weak=t4, odds_preference=prefs or ["Generic"], h2h_rows=h2h_rows, engine=engine_name, ml_blend=ml_blend)
    if sb.button("💾 Save these settings as default"):
        s.save(); sb.success("Saved to settings.json")
    return s


def make_predict_fn(snap, s):
    def fn(home, away, league, date=None, odds=None):
        if date is None or date != date:
            date = pd.Timestamp.today().normalize()
        return engine.predict_any(snap, st.session_state.get("ml"), s, home, away, league, date, odds)
    return fn


def fixture_odds(row, prefs) -> dict:
    r = data.resolve_odds(pd.DataFrame([row]), prefs).iloc[0]
    return {k: (float(r[k]) if pd.notna(r[k]) else None) for k in ODDS_LABELS}


# ------------------------------------------------------------------ dashboard
def render_prediction(pred, s, odds, hist, snap):
    st.markdown(f"## {pred.home}  vs  {pred.away}")
    st.caption(f"Engine: {pred.source}")
    if pred.warnings:
        for w in pred.warnings:
            st.warning(w)
    else:
        st.success(f"Model built from {int(pred.n_matches)} matches · {pred.home} {pred.games_home} home games · "
                   f"{pred.away} {pred.games_away} away games")
    c = st.columns(6)
    c[0].metric("Home xG", f"{pred.lam_h:.2f}"); c[1].metric("Away xG", f"{pred.lam_a:.2f}")
    c[2].metric("Total xG", f"{pred.total_xg:.2f}")
    c[3].metric("Most likely score", f"{pred.top_scores[0][0]}-{pred.top_scores[0][1]}")
    c[4].metric(f"{pred.home} home games", pred.games_home); c[5].metric(f"{pred.away} away games", pred.games_away)

    # colour key
    key = " ".join(f"<span style='background:{bg};color:{fg};padding:2px 8px;border-radius:4px;font-size:12px'>{n}</span>"
                   for n, bg, fg in config.TIERS)
    st.markdown(f"Colour key: {key} &nbsp; thresholds {s.tier_very_strong:.0%} / {s.tier_strong:.0%} / {s.tier_lean:.0%} / {s.tier_weak:.0%}",
                unsafe_allow_html=True)

    cols = st.columns(len(PANELS))
    for col, (title, groups) in zip(cols, PANELS):
        with col:
            st.markdown(f"**{title}**")
            for g, keys in ui.keys_for_groups(groups).items():
                if not keys:
                    continue
                st.caption(g)
                df = ui.market_df(pred, keys, s)
                if g == "HT/FT":
                    show(ui.heat_style(df.set_index("Market")[["Prob"]]).format({"Prob": "{:.1%}"}), hide_index=False)
                else:
                    show(ui.style_market(df, s))

    a, b, c3 = st.columns(3)
    with a:
        st.markdown("**Top 10 strongest scenarios**"); show(ui.style_market(ui.ranked_df(pred, s, 10), s))
    with b:
        st.markdown("**5 least likely scenarios**"); show(ui.style_market(ui.ranked_df(pred, s, 5, bottom=True), s))
    with c3:
        st.markdown("**Top 5 correct scores**"); show(ui.top_scores_df(pred).style.format({"Prob": "{:.1%}", "Fair odds": "{:.2f}"}).hide(axis="index"))

    st.markdown("**Scoreline probabilities**")
    show(ui.heat_style(ui.scoreline_df(pred)), hide_index=False)

    st.markdown("**Value check vs bookmaker odds**")
    vdf = ui.value_df(pred, odds, s)
    show(ui.style_value(vdf))
    mg = ui.margin_1x2(odds)
    st.caption(("Bookmaker 1X2 margin: " + f"{mg:.1%}. " if mg is not None else "") +
               "Edge = model probability × book odds − 1 (before margin adjustment). A model edge is a hypothesis, not a guarantee.")

    st.markdown("**History check (actual results)**")
    show(ui.style_check(h2h.history_check(snap, hist, pred.home, pred.away, pred.league), s), hide_index=False)
    st.markdown("**Recent head-to-head meetings (all history, venue as played)**")
    rt = h2h.recent_table(hist, pred.home, pred.away, s.h2h_rows, s.odds_preference)
    if rt.empty:
        st.info("No previous meetings found in History for these two teams.")
    else:
        show(rt.style.format({k: "{:.2f}" for k in ("1", "X", "2", "O2.5", "U2.5")}, na_rep="–"))


def tab_dashboard(s, snap):
    ss = st.session_state
    pf = make_predict_fn(snap, s)
    kick = None
    fx = ss.fixtures
    mode = st.radio("Match source", ["From fixtures", "Custom match"], horizontal=True)
    odds = {k: None for k in ODDS_LABELS}
    if mode == "From fixtures" and fx is not None and len(fx):
        labels = [ui.fixture_label(r) for _, r in fx.iterrows()]
        i = st.selectbox("Select fixture", list(range(len(fx))), format_func=lambda x: labels[x])
        row = fx.iloc[i]
        home, away, league = row["HomeTeam"], row["AwayTeam"], row["League"]
        kick = row["Date"]
        odds = fixture_odds(row, s.odds_preference)
    else:
        if mode == "From fixtures":
            st.info("No fixtures loaded – use a custom match, or add fixtures in the Data tab.")
        leagues = sorted(ss.history["League"].unique())
        league = st.selectbox("League", leagues)
        lg = ss.history[ss.history["League"] == league]
        teams = sorted(set(lg["HomeTeam"]) | set(lg["AwayTeam"]))
        c1, c2 = st.columns(2)
        home = c1.selectbox("Home team", teams)
        away = c2.selectbox("Away team", [t for t in teams if t != home])
    with st.expander("Enter / override bookmaker odds"):
        cols = st.columns(len(ODDS_LABELS))
        for col, (k, lab) in zip(cols, ODDS_LABELS.items()):
            v = col.number_input(lab, 0.0, 1000.0, float(odds[k] or 0.0), 0.01, key=f"odds_{k}_{home}_{away}")
            odds[k] = v if v > 1.0 else None
    pred = pf(home, away, league, kick, odds)
    render_prediction(pred, s, odds, ss.history, snap)


# ------------------------------------------------------------------ all fixtures
def tab_fixtures(s, snap):
    ss = st.session_state
    fx = ss.fixtures
    if fx is None or fx.empty:
        st.info("Load a fixtures file in the Data tab."); return
    leagues = sorted(fx["League"].unique())
    sel = st.multiselect("Leagues", leagues, default=leagues)
    fxs = fx[fx["League"].isin(sel)]
    odds_fn = lambda r: fixture_odds(r, s.odds_preference)
    pf = make_predict_fn(snap, s)
    b = ui.batch_predictions(fxs, pf, s, odds_fn)
    only_value = st.checkbox("Only fixtures with a VALUE flag")
    if only_value:
        b = b[b["VALUE flags"] != ""]
    st.markdown(f"**Predictions for {len(b)} fixtures**")
    shown = b.drop(columns=["idx"]).copy()
    shown["Date"] = shown["Date"].dt.strftime("%d-%b-%Y")
    prob_cols = ["P(1)", "P(X)", "P(2)", "P(O2.5)", "P(U2.5)", "P(BTTS)", "Pick prob"]
    sty = shown.style.format({**{c: "{:.1%}" for c in prob_cols}, "xG H": "{:.2f}", "xG A": "{:.2f}"})
    sty = sty.map(lambda v: ui.tier_css(v, s), subset=prob_cols)
    show(sty)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    export = b.copy()
    export["Date"] = export["Date"].dt.strftime("%Y-%m-%d")
    export = export.drop(columns=["idx"])
    st.download_button("⬇ Download predictions CSV", export.to_csv(index=False).encode("utf-8"),
                       file_name=f"predictions_{stamp}.csv", mime="text/csv")

    st.markdown("**Strongest scenarios across all fixtures**")
    mp = st.slider("Minimum probability", 0.5, 0.95, 0.70, 0.01)
    picks = ui.strongest_picks(fxs, pf, s, mp, odds_fn)
    picks["Market"] = picks["Match"] + " – " + picks["Market"]
    show(ui.style_market(picks[["Market", "Prob", "Fair odds", "Signal"]], s))


# ------------------------------------------------------------------ backtest
def tab_backtest(s):
    ss = st.session_state
    hist = ss.history
    if hist["Date"].notna().sum() < 100:
        st.info("Need a larger, dated History to backtest."); return
    st.caption("Walk-forward test: for each period the model is fitted only on earlier matches, then scored on that period. "
               "Lower log loss / Brier is better; compare with the constant-frequency reference rows.")
    dmin, dmax = hist["Date"].min().date(), hist["Date"].max().date()
    c1, c2, c3 = st.columns(3)
    start = c1.date_input("Test from", max(dmin, (pd.Timestamp(dmax) - pd.Timedelta(days=365)).date()), min_value=dmin, max_value=dmax)
    step = int(c2.number_input("Refit every (days)", 7, 365, 30, 7))
    if entitlements.enabled("parameter_sweeps"):
        param = c3.selectbox("Compare settings", ["(current settings only)", "shrinkage", "half_life_days", "dc_rho"])
        values = None
        if param != "(current settings only)":
            defaults = {"shrinkage": "0, 2, 5, 10, 20", "half_life_days": "0, 180, 365, 730", "dc_rho": "-0.1, -0.05, 0, 0.05"}
            values = [float(x) for x in st.text_input("Values to try (comma-separated)", defaults[param]).split(",") if x.strip()]
    else:
        param, values = "(current settings only)", None
        c3.info("Parameter sweeps are part of the Pro roadmap.")
    if st.button("▶ Run backtest"):
        with st.spinner("Running walk-forward backtest…"):
            bt = backtest.walk_forward(hist, s, pd.Timestamp(start), None, step)
            if bt.empty:
                st.warning("No matches in the chosen period."); return
            st.markdown("**Current settings**")
            m = backtest.metrics(bt)
            show(m.style.format({"Log loss": "{:.4f}", "Brier": "{:.4f}", "Accuracy": "{:.1%}"}), hide_index=False)
            cc = st.columns(3)
            for col, (title, pc, oc) in zip(cc, [("Over 2.5 calibration", "pO25", "over"), ("BTTS calibration", "pBTTS", "btts"), ("Home win calibration", "pH", None)]):
                with col:
                    st.markdown(f"**{title}**")
                    d = bt.assign(home_win=(bt["res"] == 0).astype(int)) if oc is None else bt
                    cal = backtest.calibration(d, pc, oc or "home_win")
                    show(cal.style.format({"Predicted": "{:.1%}", "Actual": "{:.1%}"}), hide_index=False)
            if values and entitlements.enabled("parameter_sweeps"):
                st.markdown(f"**Sweep: {param}**")
                sw = backtest.sweep(hist, s, param, values, pd.Timestamp(start), None, step)
                show(sw.style.format({c: "{:.4f}" for c in sw.columns if "log loss" in c} | {"1X2 accuracy": "{:.1%}"}))
                if len(sw):
                    best = sw.loc[sw["1X2 log loss"].idxmin(), param]
                    st.success(f"Lowest 1X2 log loss at {param} = {best}. Set it in the sidebar and save.")



# ------------------------------------------------------------------ ML lab
def tab_ml(s):
    ss = st.session_state
    if not ml.ml_available():
        st.error("scikit-learn is not installed. Run:  pip install scikit-learn   (then restart the app)"); return
    st.markdown("**Community ML engine.** Gradient-boosted trees learn from leakage-free features (Elo ratings, recent form, "
                "venue-specific goal rates, rest days, head-to-head, league averages). Their predictions are fitted back into the same "
                "scoreline grid, so all markets stay consistent. **Football is very hard to predict – bookmaker odds are the benchmark. "
                "Use the comparison below to decide whether ML (or a blend) actually beats the Poisson model on your data.**")
    b = ss.get("ml")
    c1, c2 = st.columns(2)
    use_odds = c1.checkbox("Use bookmaker odds as features", False,
                           help="Usually the strongest signal, but then the model largely echoes the market and every fixture needs odds.")
    calibrate = c2.checkbox("Calibrate probabilities (slower, usually better)", True)
    n_hist = int(ss.history["Date"].notna().sum())
    if b is not None:
        st.success(f"Trained model: {b.meta['n_rows']:,} matches up to {pd.Timestamp(b.meta['last_date']):%d-%b-%Y} · trained {b.meta['trained_at']} · "
                   f"odds features: {'yes' if b.meta['use_odds'] else 'no'} · calibrated: {'yes' if b.meta['calibrate'] else 'no'}")
        if n_hist > b.meta["n_rows"]:
            st.warning(f"History now has {n_hist:,} dated matches but the model was trained on {b.meta['n_rows']:,} – retrain to include the new results.")
    else:
        st.info("No trained model yet.")
    if n_hist < 2000:
        st.warning(f"History has {n_hist:,} dated matches. Tree models need thousands (several seasons across leagues) to beat simple statistics.")
    if st.button("🤖 Train ML model on all history"):
        with st.spinner("Building features and training (this can take a minute on large histories)…"):
            try:
                nb = ml.train_bundle(ss.history, s.odds_preference, use_odds, calibrate)
                nb.save(); ss.ml = nb
                st.success("Trained and saved. Choose the engine in the sidebar (ML or Blend).")
            except Exception as e:
                st.error(str(e))

    st.divider()
    st.subheader("Does it beat the Poisson model? (walk-forward test)")
    st.caption("For each period the models are trained only on earlier matches and scored on that period – the honest way to test.")
    dmin, dmax = ss.history["Date"].min().date(), ss.history["Date"].max().date()
    a, bcol = st.columns(2)
    start = a.date_input("Test from", max(dmin, (pd.Timestamp(dmax) - pd.Timedelta(days=365)).date()), min_value=dmin, max_value=dmax, key="ml_start")
    step = int(bcol.number_input("Retrain every (days)", 30, 365, 120, 30, key="ml_step"))
    if st.button("▶ Run comparison"):
        with st.spinner("Running walk-forward comparison…"):
            wf = ml.ml_walk_forward(ss.history, pd.Timestamp(start), step, calibrate, use_odds, s.odds_preference)
            if wf.empty:
                st.warning("Not enough earlier data to train for that period. Pick a later start date or add more history.")
            else:
                pw = backtest.walk_forward(ss.history, s, pd.Timestamp(start), None, step)
                all_t, bk_t, _details = ml.compare_metrics(wf, pw)
                ss.ml_cmp = {"all": all_t, "bk": bk_t}
    cmp_ = ss.get("ml_cmp")
    if cmp_:
        fm = {"1X2 log loss": "{:.4f}", "1X2 Brier": "{:.4f}", "1X2 accuracy": "{:.1%}", "O2.5 log loss": "{:.4f}", "BTTS log loss": "{:.4f}"}
        st.markdown("**All test matches** (lower log loss / Brier is better)")
        show(cmp_["all"].style.format(fm, na_rep="–"), hide_index=False)
        if not cmp_["bk"].empty:
            st.markdown("**Matches with bookmaker odds – including the bookmaker (margin removed)**")
            show(cmp_["bk"].style.format(fm, na_rep="–"), hide_index=False)
        st.caption("Use these out-of-sample results to compare ML with Poisson and bookmaker probabilities. A lower log loss / Brier is better; "
                   "results on synthetic sample data are not meaningful.")

    st.divider()
    if entitlements.enabled("feature_importance"):
        if st.button("🔎 Show feature importance"):
            with st.spinner("Computing permutation importance…"):
                ss.ml_imp = ml.feature_importance(ss.history, s.odds_preference, use_odds)
        if ss.get("ml_imp") is not None:
            st.markdown("**What the model relies on** (drop in log loss when a feature is shuffled; higher = more important)")
            show(ss.ml_imp.head(20).style.format({"Importance": "{:.4f}"}))
    else:
        st.info("Pro roadmap: feature importance, paired significance testing, advanced diagnostics and parameter optimisation.")

# ------------------------------------------------------------------ data
def tab_data(s):
    ss = st.session_state
    st.caption("Accepts football-data.co.uk files (CSV or all-euro-data xlsx workbooks, one sheet per league) and generic files with "
               "Date, League, HomeTeam, AwayTeam, FTHG, FTAG, HTHG, HTAG and optional odds columns. No row limit.")
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Results history")
        up = st.file_uploader("Upload one or more results files", type=["csv", "xlsx", "xls", "xlsm"], accept_multiple_files=True, key="up_hist")
        folder = st.text_input("…or load every file in a folder", "")
        if st.button("Load history") and (up or folder):
            with st.spinner("Reading files…"):
                h = data.load_history(list(up)) if up else data.load_folder(folder)
            if h.empty:
                st.error("No usable rows found (need HomeTeam, AwayTeam, FTHG, FTAG).")
            else:
                ss.history = h; ss.data_version += 1; ss.sample = False
                data.save_dataset(h, "history"); st.success(f"Loaded {len(h):,} matches and saved to the data folder.")
    with c2:
        st.subheader("Fixtures")
        upf = st.file_uploader("Upload fixtures file(s)", type=["csv", "xlsx", "xls", "xlsm"], accept_multiple_files=True, key="up_fx")
        if st.button("Load fixtures") and upf:
            f = data.load_fixtures(list(upf))
            f = f[f["HomeTeam"].notna() & f["AwayTeam"].notna()]
            ss.fixtures = f; ss.data_version += 1
            data.save_dataset(f, "fixtures"); st.success(f"Loaded {len(f)} fixtures.")
    hs = data.summarize(ss.history)
    st.markdown("**History summary**")
    if hs["rows"]:
        m = st.columns(6)
        m[0].metric("Matches", f"{hs['rows']:,}"); m[1].metric("Leagues", hs["leagues"]); m[2].metric("Teams", hs["teams"])
        m[3].metric("Seasons", hs["seasons"]); m[4].metric("With HT scores", f"{hs['with_ht']:,}")
        m[5].metric("From → to", f"{hs['from']:%b %Y} → {hs['to']:%b %Y}")
        st.caption("Odds sources detected: " + (", ".join(hs["odds_sources"]) or "none"))
    if ss.fixtures is not None and len(ss.fixtures):
        chk = data.check_team_names(ss.fixtures, ss.history)
        if chk.empty:
            st.success("All fixture teams were found in History.")
        else:
            st.warning("Some fixture teams are not in History (spelling?). Add mappings to data/team_aliases.json and reload.")
            show(chk)
    with st.expander("Preview history"):
        show(ss.history.drop(columns=[c for c in ss.history.columns if "__" in c]).tail(200))


# ------------------------------------------------------------------ main
def main():
    init_state()
    s = sidebar_settings()
    snap = get_snapshot(s)
    t1, t2, t3, t6, t4, t5 = st.tabs(["📊 Dashboard", "📅 All fixtures", "🧪 Backtest", "🤖 ML Lab", "🗂 Data", "ℹ️ Help"])
    with t1: tab_dashboard(s, snap)
    with t2: tab_fixtures(s, snap)
    with t3: tab_backtest(s)
    with t6: tab_ml(s)
    with t4: tab_data(s)
    st.divider()
    st.caption("BakerT MatchDashboard Community v" + __version__ + " · For information and entertainment only. Predictions are statistical estimates, "
               "not betting advice and not guaranteed. Gamble responsibly; 18+ only.")
    with t5:
        st.markdown((config.ROOT / "README.md").read_text(encoding="utf-8") if (config.ROOT / "README.md").exists() else "See README.md")


main()
