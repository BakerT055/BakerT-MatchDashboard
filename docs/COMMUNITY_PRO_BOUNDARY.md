# Community / Pro boundary

The public repository is the MIT-licensed Community edition.

Community includes all shared prediction code and the basic ML workflow. The Community
application does not expose feature importance, paired significance testing, parameter
sweeps or advanced reports in its UI.

Those advanced capabilities are planned for a separate commercial Pro distribution.
They are not delivered as a secure hidden package here. Since Community is open source,
users can inspect and modify the code. The `matchdash.entitlements` module is a UI policy,
not a licence enforcement mechanism.

For a real hosted Pro product:
- keep proprietary Pro implementation in a private package/service;
- authenticate users and verify subscriptions server-side;
- enforce permissions at API/service boundaries;
- never store signing secrets or payment credentials in this repository.
