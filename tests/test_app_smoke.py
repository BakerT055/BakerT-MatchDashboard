"""Smoke test for app.py WITHOUT Streamlit installed: a stand-in module records widget calls and returns
defaults, so every code path in the script executes and every styled table is rendered to HTML.
It does not test the real browser UI – run `streamlit run app.py` for that."""
import sys, runpy, types
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


class SS(dict):
    __getattr__ = lambda self, k: self[k] if k in self else (_ for _ in ()).throw(AttributeError(k))
    __setattr__ = dict.__setitem__


def make_fake(radio_idx=0, buttons=False, engine_choice="Poisson"):
    log = {"tables": 0, "calls": []}

    class Fake:
        session_state = SS()
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def __getattr__(self, name):
            def generic(*a, **k):
                log["calls"].append(name); return Fake()
            return generic
        def __getitem__(self, i): return Fake()
        # ---- widgets returning values
        def number_input(self, label, mn=None, mx=None, value=None, step=None, **k): return k.get("value", value)
        def slider(self, label, mn=None, mx=None, value=None, step=None, **k): return value
        def checkbox(self, label, value=False, **k): return value
        def date_input(self, label, value=None, **k): return value
        def selectbox(self, label, options, **k): return options[0]
        def radio(self, label, options, **k): return engine_choice if label == "Engine" else options[radio_idx]
        def multiselect(self, label, options, default=None, **k): return default or []
        def text_input(self, label, value="", **k): return value
        def button(self, label, **k): return buttons
        def file_uploader(self, *a, **k): return []
        def columns(self, n): return [Fake() for _ in range(n if isinstance(n, int) else len(n))]
        def tabs(self, labels): return [Fake() for _ in labels]
        def dataframe(self, obj, **k):
            html = obj.to_html() if hasattr(obj, "to_html") else str(obj)
            assert len(html) > 0; log["tables"] += 1
        def download_button(self, *a, **k): log["calls"].append("download")
    fake = Fake()
    fake.session_state = SS()
    fake.sidebar = Fake()
    mod = types.ModuleType("streamlit")
    for n in dir(fake):
        if not n.startswith("__"):
            setattr(mod, n, getattr(fake, n))
    mod.session_state = fake.session_state; mod.sidebar = fake.sidebar
    mod.__getattr__ = lambda name: (lambda *a, **k: Fake())   # module-level generic fallback
    return mod, log


def run(radio_idx=0, buttons=False, engine_choice="Poisson"):
    mod, log = make_fake(radio_idx, buttons, engine_choice)
    sys.modules["streamlit"] = mod
    for m in [m for m in sys.modules if m.startswith("matchdash")]:
        del sys.modules[m]
    runpy.run_path(str(ROOT / "app.py"), run_name="__main__")
    return log


def test_default_and_custom():
    bundle_file = ROOT / "data" / "ml_bundle.pkl"
    bundle_file.unlink(missing_ok=True)
    # 1) no trained ML model: ML/Blend must fall back to Poisson gracefully
    for idx, b, eng in [(0, False, "Poisson"), (1, False, "ML"), (0, True, "Poisson")]:
        log = run(idx, b, eng)
        assert log["tables"] >= 10, log["tables"]
        print(f"  no-ML  radio={idx} buttons={b} engine={eng}: {log['tables']} tables OK")
    # 2) with a trained model saved to disk: every engine, dashboard + fixtures + ML Lab
    for m in [m for m in sys.modules if m.startswith("matchdash")]:
        del sys.modules[m]
    import importlib
    data = importlib.import_module("matchdash.data"); ml = importlib.import_module("matchdash.ml")
    h = data.load_history([ROOT / "data" / "sample_history.csv"])
    ml.train_bundle(h, ["Generic"], use_odds=False, calibrate=False).save(bundle_file)
    try:
        for idx, b, eng in [(0, False, "ML"), (1, False, "Blend"), (0, False, "Blend"), (0, True, "Blend")]:
            log = run(idx, b, eng)
            assert log["tables"] >= 10, log["tables"]
            print(f"  with-ML radio={idx} buttons={b} engine={eng}: {log['tables']} tables OK")
    finally:
        bundle_file.unlink(missing_ok=True)
        (ROOT / "settings.json").unlink(missing_ok=True)


if __name__ == "__main__":
    test_default_and_custom(); print("PASS app smoke")
