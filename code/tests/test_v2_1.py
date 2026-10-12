from __future__ import annotations

import numpy as np
import pandas as pd

from analysis.statistics import TradePanel, detrend_residuals
from analysis.v2_1 import (
    DATA_DIR,
    TABLE_DIR,
    build_residual_cache,
    residual_block_metric_cube,
)


def _toy_panel() -> dict:
    t = np.arange(12, dtype=float)
    values = np.vstack([
        2 + 0.4 * t + np.sin(t),
        3 - 0.2 * t + 0.7 * np.cos(t * 1.3),
        1 + 0.1 * t + ((t % 4) - 1.5),
    ])
    panel = TradePanel(
        crop="Wheat", direction="Import", country_code=1, country_name="Toy",
        partner_codes=np.array([2, 3, 4]), partner_names=["a", "b", "c"], values=values,
    )
    return {("Wheat", "Import", 1): panel}


def test_residual_block_cube_matches_explicit_reconstruction_without_level_splicing():
    panels = _toy_panel()
    cache = build_residual_cache(panels, np.arange(12))
    cube = residual_block_metric_cube(cache, block_length=4, n_boot=5)
    panel = next(iter(panels.values()))
    residual = detrend_residuals(panel.values, degree=1)
    fitted = panel.values - residual
    idx = cube["indices"][0]
    reconstructed = fitted + residual[:, idx]
    assert np.any(reconstructed < 0)  # negative reconstructed values are retained, not clipped
    refitted_residual = detrend_residuals(reconstructed, degree=1)
    mu = reconstructed.sum(axis=0).mean()
    variance = np.var(refitted_residual.sum(axis=0), ddof=1)
    diagonal = np.var(refitted_residual, axis=1, ddof=1).sum()
    assert np.isclose(cube["mu"][0, 0], mu, rtol=1e-11, atol=1e-11)
    assert np.isclose(cube["variance"][0, 0], variance, rtol=1e-11, atol=1e-11)
    assert np.isclose(cube["D"][0, 0], diagonal, rtol=1e-11, atol=1e-11)


def test_explicit_pairwise_covariance_equals_v_minus_d():
    audit = pd.read_csv(DATA_DIR / "pairwise_covariance_audit.csv")
    assert len(audit) >= 8 * 20
    assert audit["ddof"].eq(1).all()
    assert audit["relative_error"].max() < 1e-8


def test_paired_phi_uses_valid_sign_and_wilcoxon_tests_not_bootstrap_tail_p():
    paired = pd.read_csv(TABLE_DIR / "paired_phi_valid_inference.csv")
    assert {"exact_sign_test_p", "wilcoxon_signed_rank_p"}.issubset(paired.columns)
    assert not any("empirical_p" in c or "bootstrap_tail" in c for c in paired.columns)
    assert paired["exact_sign_test_p"].between(0, 1).all()
    assert paired["wilcoxon_signed_rank_p"].between(0, 1).all()


def test_residual_block_outputs_use_2000_draws_and_prespecified_lengths():
    summary = pd.read_csv(TABLE_DIR / "exact_decomposition_residual_block.csv")
    assert set(summary["block_length"]) == {3, 4, 5}
    assert summary["effective_draws"].eq(2000).all()
    assert set(summary[summary["scheme"] == "residual_two_stage"]["block_length"]) == {4}
    assert {"percentile_ci_low", "percentile_ci_high", "basic_ci_low", "basic_ci_high", "bootstrap_bias", "bootstrap_median"}.issubset(summary.columns)


def test_residual_block_draw_identity_is_exact():
    draws = pd.read_parquet(DATA_DIR / "residual_block_bootstrap_draws.parquet")
    wide = draws.pivot(index=["crop", "direction", "scheme", "block_length", "draw"], columns="component", values="value").dropna()
    error = (wide["alpha_V"] - wide["alpha_D"] - wide["beta_phi"]).abs()
    assert error.max() < 1e-8


def test_paired_alpha_residual_block_uses_common_population_and_valid_intervals():
    paired = pd.read_csv(TABLE_DIR / "paired_alpha_residual_block.csv")
    assert paired["n_common_reporting_economies"].gt(0).all()
    assert paired["effective_draws"].eq(2000).all()
    assert (paired["percentile_ci_low"] <= paired["percentile_ci_high"]).all()
    assert (paired["basic_ci_low"] <= paired["basic_ci_high"]).all()


def test_reporting_aware_and_blank_as_zero_are_side_by_side():
    paired = pd.read_csv(TABLE_DIR / "paired_phi_reporting_aware.csv")
    assert set(paired["specification"]) == {"blank_as_zero", "reporting_aware"}
    assert paired.groupby("specification")["crop"].nunique().eq(4).all()


def test_country_null_adjusted_outputs_are_complete_and_noncausal():
    country = pd.read_csv(TABLE_DIR / "country_level_null_adjusted_phi.csv")
    paired = pd.read_csv(TABLE_DIR / "paired_null_adjusted_direction.csv")
    required = {"phi_observed", "phi_null_mean", "phi_null_median", "phi_null_q025", "phi_null_q975", "delta_sync"}
    assert required.issubset(country.columns)
    assert len(paired) == 4
    assert paired["interpretation"].str.contains("beyond marginal edge time structure").all()


def test_old_bootstrap_p_value_is_absent_from_v21_tables():
    for path in TABLE_DIR.glob("*.csv"):
        frame = pd.read_csv(path)
        assert "empirical_p_median" not in frame.columns
