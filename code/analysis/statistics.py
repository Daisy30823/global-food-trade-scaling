from __future__ import annotations

from dataclasses import dataclass
from math import ceil

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import optimize, stats
from sklearn.linear_model import TheilSenRegressor

from .config import CROPS, ELEMENTS, SEED, YEARS


@dataclass
class TradePanel:
    crop: str
    direction: str
    country_code: int
    country_name: str
    partner_codes: np.ndarray
    partner_names: list[str]
    values: np.ndarray  # partner x year


def build_panels(core: pd.DataFrame) -> dict[tuple[str, str, int], TradePanel]:
    panels: dict[tuple[str, str, int], TradePanel] = {}
    grouping = ["crop", "direction", "Reporter Country Code", "Reporter Countries"]
    for (crop, direction, code, name), group in core.groupby(grouping, sort=False, observed=True):
        pivot = group.pivot_table(
            index=["Partner Country Code", "Partner Countries"],
            columns="year",
            values="quantity_tonnes",
            aggfunc="sum",
            fill_value=0.0,
        ).reindex(columns=YEARS, fill_value=0.0)
        panels[(str(crop), str(direction), int(code))] = TradePanel(
            crop=str(crop),
            direction=str(direction),
            country_code=int(code),
            country_name=str(name),
            partner_codes=pivot.index.get_level_values(0).to_numpy(dtype=int),
            partner_names=pivot.index.get_level_values(1).astype(str).tolist(),
            values=pivot.to_numpy(dtype=float),
        )
    return panels


def detrend_residuals(values: np.ndarray, degree: int | None = 1) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    one_dim = arr.ndim == 1
    if one_dim:
        arr = arr[None, :]
    if degree is None:
        result = arr.copy()
    else:
        t = np.arange(arr.shape[1], dtype=float)
        design = np.vander(t, N=degree + 1, increasing=True)
        coef = np.linalg.lstsq(design, arr.T, rcond=None)[0]
        result = arr - (design @ coef).T
    return result[0] if one_dim else result


def min_positive_years(n_years: int, full_period: bool = False) -> int:
    if full_period:
        return 20
    if n_years == 20:
        return 15
    return int(ceil(0.75 * n_years))


def compute_window_metrics(
    panels: dict[tuple[str, str, int], TradePanel],
    start: int,
    end: int,
    degree: int | None = 1,
    transform: str = "tonnes",
) -> pd.DataFrame:
    idx = np.array([YEARS.index(y) for y in range(start, end + 1)], dtype=int)
    n_years = len(idx)
    threshold = min_positive_years(n_years, full_period=(start == 1986 and end == 2024))
    rows = []
    for panel in panels.values():
        edge_values = panel.values[:, idx]
        raw_total = edge_values.sum(axis=0)
        if transform == "log1p":
            # The sensitivity analysis transforms the country-level total flow,
            # not every bilateral edge before aggregation.
            total = np.log1p(raw_total)
            total_resid = detrend_residuals(total[None, :], degree=degree)[0]
            edge_resid = None
        else:
            total = raw_total
            edge_resid = detrend_residuals(edge_values, degree=degree)
            total_resid = edge_resid.sum(axis=0)
        mu = float(total.mean())
        variance = float(np.var(total_resid, ddof=1))
        if edge_resid is None:
            diagonal = covariance = phi = cov_share = np.nan
        else:
            edge_variances = np.var(edge_resid, axis=1, ddof=1)
            diagonal = float(edge_variances.sum())
            covariance = float(variance - diagonal)
            phi = variance / diagonal if diagonal > 0 else np.nan
            cov_share = covariance / variance if variance > 0 else np.nan
        partner_active = np.any(edge_values > 0, axis=1)
        partner_count = int(partner_active.sum())
        positive_years = int(np.sum(raw_total > 0))
        eligible = bool(mu > 0 and variance > 0 and positive_years >= threshold)
        identity_error = abs(variance - (diagonal + covariance))
        rows.append(
            {
                "crop": panel.crop,
                "direction": panel.direction,
                "country_code": panel.country_code,
                "country": panel.country_name,
                "start_year": start,
                "end_year": end,
                "n_years": n_years,
                "trend_degree": -1 if degree is None else degree,
                "transform": transform,
                "mu": mu,
                "variance": variance,
                "cv2": variance / (mu * mu) if mu > 0 else np.nan,
                "positive_years": positive_years,
                "partner_count": partner_count,
                "eligible_taylor": eligible,
                "eligible_covariance": bool(transform == "tonnes" and eligible and partner_count >= 2 and diagonal > 0),
                "D": diagonal,
                "C": covariance,
                "phi": phi,
                "C_over_D": covariance / diagonal if np.isfinite(diagonal) and diagonal > 0 else np.nan,
                "CovarianceShare": cov_share,
                "variance_identity_error": identity_error,
            }
        )
    return pd.DataFrame(rows)


def _simple_slope(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    xc = x - x.mean()
    denom = float(np.dot(xc, xc))
    if denom <= 0:
        return np.nan, np.nan
    slope = float(np.dot(xc, y - y.mean()) / denom)
    intercept = float(y.mean() - slope * x.mean())
    return intercept, slope


def bootstrap_slopes(x: np.ndarray, y: np.ndarray, n_boot: int = 1000, seed: int = SEED) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = len(x)
    slopes = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        sample = rng.integers(0, n, size=n)
        slopes[b] = _simple_slope(x[sample], y[sample])[1]
    return slopes[np.isfinite(slopes)]


def loco_slopes(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    if len(x) <= 3:
        return np.array([np.nan])
    result = np.empty(len(x), dtype=float)
    mask = np.ones(len(x), dtype=bool)
    for i in range(len(x)):
        mask[i] = False
        result[i] = _simple_slope(x[mask], y[mask])[1]
        mask[i] = True
    return result


def fit_scaling(metrics: pd.DataFrame, n_boot: int = 1000, seed: int = SEED) -> tuple[dict, pd.DataFrame, np.ndarray]:
    data = metrics.loc[metrics["eligible_taylor"] & metrics["mu"].gt(0) & metrics["variance"].gt(0)].copy()
    if len(data) < 4:
        raise ValueError("Fewer than four eligible countries")
    x = np.log(data["mu"].to_numpy(float))
    y = np.log(data["variance"].to_numpy(float))
    X = sm.add_constant(x)
    base = sm.OLS(y, X).fit()
    robust = base.get_robustcov_results(cov_type="HC3")
    ci = robust.conf_int(alpha=0.05)[1]
    cooks = base.get_influence().cooks_distance[0]
    max_idx = int(np.nanargmax(cooks))
    boot = bootstrap_slopes(x, y, n_boot=n_boot, seed=seed)
    loco = loco_slopes(x, y)

    cv_y = np.log(data["cv2"].to_numpy(float))
    cv_base = sm.OLS(cv_y, X).fit()
    cv_robust = cv_base.get_robustcov_results(cov_type="HC3")
    identity_error = abs(float(base.params[1]) - (2.0 + float(cv_base.params[1])))
    if identity_error >= 1e-8:
        raise AssertionError(f"alpha=2+delta identity failed: {identity_error}")

    huber = sm.RLM(y, X, M=sm.robust.norms.HuberT()).fit()
    theil = TheilSenRegressor(random_state=seed, fit_intercept=True).fit(x.reshape(-1, 1), y)
    t1 = (float(robust.params[1]) - 1.0) / float(robust.bse[1])
    t2 = (float(robust.params[1]) - 2.0) / float(robust.bse[1])
    p1 = 2.0 * stats.t.sf(abs(t1), df=float(robust.df_resid))
    p2 = 2.0 * stats.t.sf(abs(t2), df=float(robust.df_resid))
    result = {
        "alpha": float(robust.params[1]),
        "intercept": float(robust.params[0]),
        "hc3_se": float(robust.bse[1]),
        "ci_low": float(ci[0]),
        "ci_high": float(ci[1]),
        "bootstrap_ci_low": float(np.quantile(boot, 0.025)),
        "bootstrap_ci_high": float(np.quantile(boot, 0.975)),
        "bootstrap_median": float(np.median(boot)),
        "n_countries": int(len(data)),
        "r_squared": float(base.rsquared),
        "max_cooks_distance": float(cooks[max_idx]),
        "max_cooks_country": str(data.iloc[max_idx]["country"]),
        "loco_alpha_min": float(np.nanmin(loco)),
        "loco_alpha_max": float(np.nanmax(loco)),
        "p_alpha_eq_1": float(p1),
        "p_alpha_eq_2": float(p2),
        "delta": float(cv_robust.params[1]),
        "delta_hc3_se": float(cv_robust.bse[1]),
        "delta_ci_low": float(cv_robust.conf_int()[1, 0]),
        "delta_ci_high": float(cv_robust.conf_int()[1, 1]),
        "alpha_identity_error": float(identity_error),
        "huber_alpha": float(huber.params[1]),
        "theil_sen_alpha": float(theil.coef_[0]),
    }
    data["log_mu"] = x
    data["log_variance"] = y
    data["log_cv2"] = cv_y
    data["cooks_distance"] = cooks
    return result, data, boot


def _m2_residual(params: np.ndarray, mu: np.ndarray, log_v: np.ndarray) -> np.ndarray:
    a, b = np.exp(np.clip(params, -50, 50))
    prediction = a * mu + b * mu * mu
    return np.log(np.maximum(prediction, np.finfo(float).tiny)) - log_v


def fit_m2(mu: np.ndarray, variance: np.ndarray) -> dict:
    mu = np.asarray(mu, dtype=float)
    variance = np.asarray(variance, dtype=float)
    log_v = np.log(variance)
    _, alpha = _simple_slope(np.log(mu), log_v)
    scale = float(np.exp(np.mean(log_v - alpha * np.log(mu))))
    median_mu = float(np.median(mu))
    starts = [
        np.log([max(scale, 1e-30), max(scale / max(median_mu, 1e-30), 1e-30)]),
        np.log([max(np.median(variance / mu), 1e-30), 1e-30]),
        np.log([1e-30, max(np.median(variance / (mu * mu)), 1e-30)]),
    ]
    fits = [optimize.least_squares(_m2_residual, x0=s, args=(mu, log_v), max_nfev=3000) for s in starts]
    best = min(fits, key=lambda f: float(np.sum(f.fun * f.fun)))
    a, b = np.exp(np.clip(best.x, -50, 50))
    pred_log = np.log(a * mu + b * mu * mu)
    resid = log_v - pred_log
    return {
        "a": float(a),
        "b": float(b),
        "pred_log": pred_log,
        "rmse": float(np.sqrt(np.mean(resid * resid))),
        "mae": float(np.mean(np.abs(resid))),
        "sse": float(np.sum(resid * resid)),
    }


def fit_model_comparison(metrics: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    data = metrics.loc[metrics["eligible_taylor"] & metrics["mu"].gt(0) & metrics["variance"].gt(0)].copy()
    mu = data["mu"].to_numpy(float)
    variance = data["variance"].to_numpy(float)
    x = np.log(mu)
    y = np.log(variance)
    X = sm.add_constant(x)
    m1 = sm.OLS(y, X).fit()
    m1_pred = m1.predict(X)
    m1_resid = y - m1_pred
    m2 = fit_m2(mu, variance)
    n = len(data)

    cv1 = np.empty(n)
    cv2 = np.empty(n)
    for i in range(n):
        keep = np.ones(n, dtype=bool)
        keep[i] = False
        i1, s1 = _simple_slope(x[keep], y[keep])
        cv1[i] = y[i] - (i1 + s1 * x[i])
        f2 = fit_m2(mu[keep], variance[keep])
        cv2[i] = y[i] - np.log(f2["a"] * mu[i] + f2["b"] * mu[i] * mu[i])

    def info_criteria(sse: float, k: int = 2) -> tuple[float, float]:
        sigma2 = max(sse / n, np.finfo(float).tiny)
        ll = -0.5 * n * (np.log(2 * np.pi * sigma2) + 1)
        return 2 * k - 2 * ll, k * np.log(n) - 2 * ll

    aic1, bic1 = info_criteria(float(np.sum(m1_resid * m1_resid)))
    aic2, bic2 = info_criteria(m2["sse"])
    result = {
        "M1_alpha": float(m1.params[1]),
        "M1_rmse": float(np.sqrt(np.mean(m1_resid * m1_resid))),
        "M1_mae": float(np.mean(np.abs(m1_resid))),
        "M1_cv_rmse": float(np.sqrt(np.mean(cv1 * cv1))),
        "M1_cv_mae": float(np.mean(np.abs(cv1))),
        "M1_AIC": float(aic1),
        "M1_BIC": float(bic1),
        "M2_a": m2["a"],
        "M2_b": m2["b"],
        "M2_rmse": m2["rmse"],
        "M2_mae": m2["mae"],
        "M2_cv_rmse": float(np.sqrt(np.mean(cv2 * cv2))),
        "M2_cv_mae": float(np.mean(np.abs(cv2))),
        "M2_AIC": float(aic2),
        "M2_BIC": float(bic2),
        "better_model_cv_rmse": "M1" if np.mean(cv1 * cv1) < np.mean(cv2 * cv2) else "M2",
        "n_countries": int(n),
    }
    data["M1_fitted_log_variance"] = m1_pred
    data["M1_residual"] = m1_resid
    data["M2_fitted_log_variance"] = m2["pred_log"]
    data["M2_residual"] = y - m2["pred_log"]
    return result, data


def m2_alpha_eff(mu: np.ndarray, a: float, b: float) -> np.ndarray:
    mu = np.asarray(mu, dtype=float)
    return (a + 2 * b * mu) / (a + b * mu)
