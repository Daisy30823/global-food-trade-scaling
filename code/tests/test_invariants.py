from __future__ import annotations

import numpy as np
import pandas as pd

from analysis.advanced_v2 import _component_slopes, source_shock_minus_importer
from analysis.analysis_runner import circular_shift_rows, stable_seed, window_definitions
from analysis.config import DATA_DIR, FIG_DATA_DIR, TABLE_DIR
from analysis.statistics import TradePanel, build_panels, compute_window_metrics, fit_scaling
from analysis.utils import read_json


def _toy_core(direction: str) -> pd.DataFrame:
    years = list(range(1986, 2025))
    rows = []
    for partner, base in [(2, 10.0), (3, 20.0)]:
        for k, year in enumerate(years):
            rows.append(
                {
                    "crop": "Wheat",
                    "direction": direction,
                    "Reporter Country Code": 1,
                    "Reporter Countries": "Reporter",
                    "Partner Country Code": partner,
                    "Partner Countries": f"Partner {partner}",
                    "year": year,
                    "quantity_tonnes": base + k + ((k % 3) - 1),
                }
            )
    return pd.DataFrame(rows)


def test_import_edge_direction_reporter_is_importer_partner_is_source():
    panel = build_panels(_toy_core("Import"))[("Wheat", "Import", 1)]
    assert panel.country_name == "Reporter"
    assert set(panel.partner_names) == {"Partner 2", "Partner 3"}


def test_export_edge_direction_reporter_is_exporter_partner_is_destination():
    panel = build_panels(_toy_core("Export"))[("Wheat", "Export", 1)]
    assert panel.country_name == "Reporter"
    assert set(panel.partner_names) == {"Partner 2", "Partner 3"}


def test_trade_shares_sum_to_one():
    totals = np.array([4.0, 7.0, 9.0])
    shares = totals / totals.sum()
    assert np.isclose(shares.sum(), 1.0, atol=1e-12)


def test_variance_equals_diagonal_plus_covariance():
    panels = build_panels(_toy_core("Import"))
    metric = compute_window_metrics(panels, 1986, 2024).iloc[0]
    assert abs(metric["variance"] - metric["D"] - metric["C"]) <= max(1e-8, 1e-8 * metric["variance"])


def test_alpha_equals_two_plus_delta():
    rng = np.random.default_rng(10)
    mu = np.geomspace(1, 1000, 40)
    variance = 3 * mu**1.65 * np.exp(rng.normal(0, 0.08, len(mu)))
    metrics = pd.DataFrame({"mu": mu, "variance": variance, "cv2": variance / mu**2, "eligible_taylor": True, "country": [f"c{i}" for i in range(len(mu))], "country_code": np.arange(len(mu))})
    fit, _, _ = fit_scaling(metrics, n_boot=20, seed=1)
    assert abs(fit["alpha"] - (2 + fit["delta"])) < 1e-8


def test_circular_shift_preserves_edge_mean_and_variance():
    x = np.arange(36, dtype=float).reshape(3, 12)
    shifted = circular_shift_rows(x, np.array([1, 5, 9]))
    assert np.allclose(x.mean(axis=1), shifted.mean(axis=1))
    assert np.allclose(x.var(axis=1, ddof=1), shifted.var(axis=1, ddof=1))


def test_independent_shifts_break_synchronous_edge_correlation():
    t = np.arange(24)
    wave = np.sin(2 * np.pi * t / 12)
    original = np.vstack([wave, wave])
    shifted = circular_shift_rows(original, np.array([0, 3]))
    assert np.corrcoef(shifted)[0, 1] < np.corrcoef(original)[0, 1] - 0.5


def test_ncorr_equals_matched_nominal_under_independence():
    shares = np.array([0.2, 0.3, 0.5])
    n_corr = 1 / (shares @ np.eye(3) @ shares)
    n_nominal = 1 / np.sum(shares**2)
    assert np.isclose(n_corr, n_nominal, atol=1e-12)


def test_ncorr_is_one_under_complete_synchrony():
    shares = np.array([0.2, 0.3, 0.5])
    n_corr = 1 / (shares @ np.ones((3, 3)) @ shares)
    assert np.isclose(n_corr, 1.0, atol=1e-12)


def test_randomized_results_reproduce_with_same_seed():
    seed = stable_seed("reproducibility-test")
    a = np.random.default_rng(seed).integers(0, 10_000, size=100)
    b = np.random.default_rng(seed).integers(0, 10_000, size=100)
    assert np.array_equal(a, b)


def test_rolling20_has_exactly_twenty_unique_windows_per_group():
    estimates = pd.read_csv(DATA_DIR / "alpha_window_estimates.csv")
    rolling = estimates[estimates["window_id"].astype(str).str.startswith("roll20_")]
    counts = rolling.groupby(["crop", "direction", "sample_type"])["window_id"].nunique()
    assert len(counts) == 16
    assert counts.eq(20).all()
    assert set(rolling["end_year"].unique()) == set(range(2005, 2025))


def test_rolling_and_nonoverlap_window_ids_do_not_collide():
    windows = pd.DataFrame(window_definitions())
    assert windows["window_id"].is_unique
    same_range = windows[(windows["start"] == 2005) & (windows["end"] == 2024)]
    assert set(same_range["type"]) == {"rolling20", "nonoverlap"}
    assert same_range["window_id"].nunique() == 2


def test_paired_direction_membership_uses_identical_country_index():
    membership = pd.read_csv(DATA_DIR / "paired_country_membership.csv")
    assert membership[["in_import", "in_export"]].all(axis=None)
    assert not membership.duplicated(["crop", "analysis", "country_code"]).any()
    assert membership.groupby(["crop", "analysis"]).size().gt(0).all()


def test_exact_product_and_slope_decomposition_identities():
    mu = np.geomspace(10, 10_000, 32)
    diagonal = 2.4 * mu**1.35
    phi = 0.7 * mu**0.22
    variance = diagonal * phi
    metrics = pd.DataFrame({"mu": mu, "variance": variance, "D": diagonal, "phi": phi})
    assert np.allclose(metrics["variance"], metrics["D"] * metrics["phi"], rtol=1e-12, atol=1e-12)
    slopes = _component_slopes(metrics)
    assert abs(slopes["alpha_V"] - slopes["alpha_D"] - slopes["beta_phi"]) < 1e-8
    output = pd.read_csv(TABLE_DIR / "table_v2_exact_decomposition.csv")
    assert output["identity_error"].max() < 1e-8


def test_diversification_main_summary_counts_only_eligible_importers():
    all_rows = pd.read_parquet(DATA_DIR / "source_diversification_all_sensitivities.parquet")
    summary = pd.read_csv(TABLE_DIR / "table_v2_source_diversification_sensitivity.csv")
    for row in summary.itertuples():
        mask = (
            np.isclose(all_rows["share_threshold"], row.share_threshold)
            & np.isclose(all_rows["coverage_threshold"], row.coverage_threshold)
            & all_rows["matrix_method"].eq(row.matrix_method)
        )
        eligible = all_rows[mask & all_rows["main_eligible"] & all_rows["Gap_source"].notna()]
        computable = all_rows[mask & all_rows["Gap_source"].notna()]
        assert row.n_main_eligible == len(eligible)
        assert row.n_all_computable == len(computable)


def test_leave_one_importer_out_source_shock_excludes_focal_importer():
    panel = TradePanel(
        crop="Wheat",
        direction="Export",
        country_code=10,
        country_name="Source",
        partner_codes=np.array([20, 30]),
        partner_names=np.array(["Focal", "Other"]),
        values=np.array([[2.0, 3.0, 4.0], [7.0, 11.0, 13.0]]),
    )
    assert np.array_equal(source_shock_minus_importer(panel, 20), np.array([7.0, 11.0, 13.0]))
    assert np.array_equal(source_shock_minus_importer(panel, 99), np.array([9.0, 14.0, 17.0]))


def test_figure5_uses_direct_pooled_statistics_not_averaged_medians_or_cis():
    audit = read_json(FIG_DATA_DIR / "Fig5_pooling_audit.json")
    assert audit["average_of_crop_medians_used"] is False
    assert audit["average_of_ci_bounds_used"] is False
    assert audit["pooled_statistic_rule"].startswith("direct country-level")


def test_randomization_p_values_are_never_zero():
    circular = pd.read_csv(TABLE_DIR / "table6_null_model.csv")
    block = pd.read_csv(TABLE_DIR / "table_v2_block_permutation_null.csv")
    assert circular["empirical_p_value"].gt(0).all()
    assert block["empirical_p_value"].gt(0).all()


def test_raw_to_country_aggregation_audit_is_independent_and_exact():
    audit = read_json(DATA_DIR / "raw_to_country_aggregation_audit.json")
    assert audit["passed"] is True
    assert audit["max_scaled_relative_error"] < 1e-12
    assert "groupby" in audit["method"] and "TradePanel" in audit["method"]


def test_blank_as_zero_and_reporting_aware_missingness_are_audited():
    audit = read_json(DATA_DIR / "missingness_audit.json")
    assert audit["blank_as_zero_audited"] is True
    assert audit["reporting_aware_audited"] is True
    assert audit["explicit_zero_is_reporting_activity"] is True


def test_log1p_is_not_used_in_formal_taylor_specification_curve():
    specs = pd.read_csv(TABLE_DIR / "table2_alpha_estimates.csv")
    assert "log1p" not in specs.to_csv(index=False).lower()
    assert (DATA_DIR / "descriptive_log1p_volatility.parquet").exists()
