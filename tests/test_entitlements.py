import pytest

from matchdash import entitlements


def test_community_includes_core_ml():
    assert entitlements.enabled("ml_predictions")
    assert entitlements.enabled("basic_ml_comparison")


@pytest.mark.parametrize(
    "feature",
    [
        "advanced_ml_comparison",
        "feature_importance",
        "paired_statistical_test",
        "parameter_sweeps",
        "advanced_reports",
    ],
)
def test_pro_features_are_not_exposed_in_community(feature):
    assert not entitlements.enabled(feature)


def test_require_rejects_pro_feature():
    with pytest.raises(PermissionError):
        entitlements.require("feature_importance")
