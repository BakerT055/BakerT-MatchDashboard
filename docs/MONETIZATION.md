# BakerT MatchDashboard — Open-core plan

This document describes a product direction, not legal, tax, financial or betting advice.
Obtain local legal advice before selling betting-related software or services.

## 1. Editions

### Community (this public repository)
- Poisson and Dixon–Coles predictions
- H2H, team form and market calculations
- CSV/Excel history and fixture ingestion
- Basic odds comparison and exports
- Basic walk-forward backtesting and calibration
- Gradient-boosted ML predictions for 1X2, Over 2.5 and BTTS
- Basic ML-vs-Poisson/bookmaker walk-forward comparison

Community is licensed under MIT. The licence permits use, modification and redistribution,
including commercial use, subject to its terms.

### Pro (separate commercial distribution)
Planned capabilities:
- Feature importance and advanced diagnostics
- Paired statistical significance testing
- Parameter sweeps and optimisation workflows
- Advanced reporting, automation and hosted services
- API access and managed data services

Do not put proprietary Pro source code, signing secrets, API keys or payment credentials
in this public repository. Pro must be a separate package/service with server-verified
entitlements. The Community `entitlements.py` file only controls what the Community UI
displays; it is not a DRM or security mechanism.

## 2. Before charging

- Validate the model with substantial real, out-of-sample data.
- Compare against bookmaker closing odds where legally and contractually permitted.
- Do not advertise guaranteed profits or present predictions as betting advice.
- Check the rights and terms for every results, odds and fixture data source.
- Do not redistribute third-party datasets unless the relevant licence allows it.
- Confirm applicable betting, advertising, consumer-protection, privacy and tax rules
  in Uganda and every market served.
- Prepare Terms of Service, Privacy Policy, refund policy, age controls and responsible
  gambling messaging.
- Confirm ownership of the code and brand, and have the licence reviewed if needed.

## 3. Technical roadmap for Pro

1. Move commercial-only code into a separate private package or hosted service.
2. Keep shared prediction interfaces stable so Community can run without Pro.
3. Verify paid entitlements on the server; never trust a local config flag.
4. Add authentication, tenant isolation, audit logs, rate limits and monitoring.
5. Add payment processing only after confirming provider availability in target countries.
6. Keep customer data and model assets server-side for hosted Pro services.

## 4. Release checklist

- [ ] Confirm copyright holder and approve the MIT release.
- [ ] Review dependency and bundled-data licences.
- [ ] Run the full test suite and inspect the GitHub Actions result.
- [ ] Test Windows installation with `run.bat`.
- [ ] Test a clean install with `pip install -r requirements.txt`.
- [ ] Add a Privacy Policy and Terms before collecting user data or payments.
- [ ] Tag a versioned release after tests pass.
