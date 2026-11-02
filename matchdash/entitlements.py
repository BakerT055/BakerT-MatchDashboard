"""Community edition feature policy.

This module controls which features the public Community UI exposes.
It is not a licence verifier or security boundary. Commercial Pro
capabilities must live in a separately distributed/private component
and use server-verified entitlements before being offered to customers.
"""
from __future__ import annotations

EDITION = "Community"

FEATURES = {
    "ml_predictions": True,
    "basic_ml_comparison": True,
    "advanced_ml_comparison": False,
    "feature_importance": False,
    "paired_statistical_test": False,
    "parameter_sweeps": False,
    "advanced_reports": False,
}


def enabled(feature: str) -> bool:
    """Return whether a feature is exposed in this Community build."""
    return bool(FEATURES.get(feature, False))


def require(feature: str) -> None:
    """Raise a clear error when code attempts to use a non-Community feature."""
    if not enabled(feature):
        raise PermissionError(
            f"{feature!r} is not included in the {EDITION} edition."
        )
