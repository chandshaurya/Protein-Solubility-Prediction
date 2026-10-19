from src.depth_analysis import (
    EXPERIMENTS,
    bootstrap_rmse_interval,
    build_experiments,
)


def test_experiments_cover_requested_feature_groups():
    assert set(EXPERIMENTS) == {
        "E0", "E1", "E2", "E3", "E4", "E5", "E6", "E7",
    }
    assert "mw_kda_reported" in EXPERIMENTS["E1"]
    assert "pi_reported" in EXPERIMENTS["E1"]

    assert "seq_length" in EXPERIMENTS["E2"]
    assert "gravy_score" in EXPERIMENTS["E2"]
    assert "net_charge_density" in EXPERIMENTS["E2"]

    assert any(c.startswith("aac_") for c in EXPERIMENTS["E3"])
    assert len(EXPERIMENTS["E4"]) > len(EXPERIMENTS["E3"])
    assert set(EXPERIMENTS["E1"]).isdisjoint(EXPERIMENTS["E2"])
    assert set(EXPERIMENTS["E2"]).isdisjoint(EXPERIMENTS["E3"])
    assert set(EXPERIMENTS["E4"]) >= set(EXPERIMENTS["E1"])
    assert set(EXPERIMENTS["E4"]) >= set(EXPERIMENTS["E2"])
    assert set(EXPERIMENTS["E4"]) >= set(EXPERIMENTS["E3"])
    all_features = set(EXPERIMENTS["E4"])
    assert set(EXPERIMENTS["E5"]) == all_features - set(EXPERIMENTS["E1"])
    assert set(EXPERIMENTS["E6"]) == all_features - set(EXPERIMENTS["E2"])
    assert set(EXPERIMENTS["E7"]) == all_features - set(EXPERIMENTS["E3"])


def test_experiments_only_include_features_present_in_the_dataset():
    available_features = [
        "mw_kda_reported",
        "seq_length",
        "aac_A",
        "di_AA",
        "gravy_score",
    ]

    experiments = build_experiments(available_features)

    assert experiments["E1"] == ["mw_kda_reported"]
    assert experiments["E2"] == ["seq_length", "gravy_score"]
    assert experiments["E3"] == ["aac_A"]
    assert experiments["E4"] == available_features
    available_set = set(available_features)
    assert set(experiments["E5"]) == available_set - {"mw_kda_reported"}
    assert set(experiments["E6"]) == {
        "mw_kda_reported", "aac_A", "di_AA",
    }
    assert set(experiments["E7"]) == {
        "mw_kda_reported", "seq_length", "di_AA", "gravy_score",
    }


def test_bootstrap_rmse_interval_is_exact_for_constant_error():
    lower, upper = bootstrap_rmse_interval(
        [0.0, 1.0, 2.0],
        [1.0, 2.0, 3.0],
        n_bootstrap=20,
        random_state=7,
    )

    assert lower == 1.0
    assert upper == 1.0
