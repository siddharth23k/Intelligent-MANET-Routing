"""Configuration is the contract both methods share; a typo here is silent."""

from config_loader import get_config


def test_shared_scenario_values_are_present():
    c = get_config()
    assert c.area_center == 500.0
    assert c.communication_radius_default > 0
    assert c.label_horizon >= 1
    assert c.survival_horizon >= 1
    assert c.rssi_sentinel < c.rssi_floor < 0


def test_split_sizes_are_consistent():
    c = get_config()
    assert c.test_run_count >= 1
    assert c.val_run_count >= 0


def test_smoke_settings_are_smaller_than_the_real_ones():
    c = get_config()
    assert c.smoke["rf_estimators"] < c.training["rf_estimators"]
    assert c.smoke["xgb_estimators"] < c.training["xgb_estimators"]
    assert c.smoke["sfrnnr_epochs"] < c.training["sfrnnr_epochs"]
    assert c.smoke["test_run_count"] <= c.test_run_count
