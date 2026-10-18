from src.depth_analysis import EXPERIMENTS


def test_experiments_cover_requested_feature_groups():
    assert set(EXPERIMENTS) == {"E0", "E1", "E2", "E3", "E4"}
    assert "mw_kda_reported" in EXPERIMENTS["E1"]
    assert "pi_reported" in EXPERIMENTS["E1"]

    assert "length" in EXPERIMENTS["E2"]
    assert "gravy" in EXPERIMENTS["E2"]
    assert "aromaticity" in EXPERIMENTS["E2"]
    assert "instability_index" in EXPERIMENTS["E2"]

    assert any(c.startswith("aac_") for c in EXPERIMENTS["E3"])
    assert len(EXPERIMENTS["E4"]) >= len(EXPERIMENTS["E3"])
