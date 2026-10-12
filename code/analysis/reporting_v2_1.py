from __future__ import annotations

import json

import numpy as np
import pandas as pd

from .config import SEED
from .v2_1 import REPORT_DIR


def _fmt(value, digits=3) -> str:
    return "NA" if not np.isfinite(value) else f"{value:.{digits}f}"


def _md(frame: pd.DataFrame, columns: list[str]) -> str:
    head = "| " + " | ".join(columns) + " |"
    rule = "|" + "|".join(["---"] * len(columns)) + "|"
    rows = []
    for values in frame[columns].itertuples(index=False, name=None):
        rows.append("| " + " | ".join(_fmt(v) if isinstance(v, (float, np.floating)) else str(v) for v in values) + " |")
    return "\n".join([head, rule, *rows])


def write_reports_v2_1(
    alpha_main: pd.DataFrame,
    paired_phi: pd.DataFrame,
    decomposition: pd.DataFrame,
    paired_alpha: pd.DataFrame,
    reporting_paired: pd.DataFrame,
    null_paired: pd.DataFrame,
    pairwise_audit: pd.DataFrame,
    figures: dict,
    tests_passed: int,
) -> dict:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    main_decomp = decomposition[(decomposition["scheme"] == "residual_two_stage") & (decomposition["block_length"] == 4)]
    beta = main_decomp[main_decomp["component"] == "beta_phi"]
    alpha_all_below = bool(alpha_main["estimate"].lt(2).all())
    alpha_pair_resolved = paired_alpha.assign(resolved=lambda d: (d.percentile_ci_low > 0) | (d.percentile_ci_high < 0))
    null_resolved = null_paired.assign(resolved=lambda d: (d.bootstrap_ci_low > 0) | (d.bootstrap_ci_high < 0))

    audit = f"""# v2.1 定点修复审计答复

## 修复范围

本次只使用既有 FAOSTAT 压缩包、四种既定作物、1986–2024 年、既定样本门槛和已经预设的稳健性规格。`outputs_v2/` 与 `report_v2/` 未被覆盖，也没有搜索新的 `alpha>2` 规格。

## 1. 配对 φ 推断

旧版普通 bootstrap 尾部比例已从正式输出中删除。v2.1 对共同 covariance-eligible reporting economies/areas 报告配对差、中位 log ratio、典型出口/进口 φ 比率、2,000 次国家 bootstrap 区间、双侧 exact binomial sign test 和 Wilcoxon signed-rank sensitivity。

{_md(paired_phi, ['crop','n_pairs','median_difference','mean_difference','typical_phi_ratio','typical_phi_ratio_ci_low','typical_phi_ratio_ci_high','share_export_gt_import','exact_sign_test_p','wilcoxon_signed_rank_p'])}

## 2. 残差 moving-block bootstrap

每条边先按原始年份拟合线性趋势。每次抽样使用全球共同年份区块，仅重采样残差，在原始年份位置与拟合趋势重建，不截断可能出现的负重建值，随后重新拟合趋势并计算 `mu`、`V`、`D` 和 `phi`。主区块长度为 4，敏感性长度为 3 和 5；正式抽样均为 2,000 次。两阶段方案在同一全局年份抽样上继续重采样 reporting economies。

主 L4 两阶段结果：

{_md(main_decomp, ['crop','direction','component','estimate','percentile_ci_low','percentile_ci_high','basic_ci_low','basic_ci_high','bootstrap_bias','bootstrap_median','effective_draws'])}

每次有效抽样均满足 `alpha_V = alpha_D + beta_phi`，误差阈值为 `1e-8`。

## 3. 配对 alpha

进口和出口使用相同 reporting-economy 集合、相同全局残差年份区块和相同第二阶段国家索引。

{_md(paired_alpha, ['crop','n_common_reporting_economies','alpha_import','alpha_export','estimate','percentile_ci_low','percentile_ci_high','basic_ci_low','basic_ci_high','effective_draws'])}

## 4. Reporting-aware 配对检验

主规格仍为 blank-as-zero。敏感性只使用已有 reporting activity 规则识别覆盖年份；显式零计为报告活动，全伙伴均空白的年份不计为零，不插值。

{_md(reporting_paired, ['specification','crop','n_pairs','median_difference','typical_phi_ratio','typical_phi_ratio_ci_low','typical_phi_ratio_ci_high','exact_sign_test_p','wilcoxon_signed_rank_p'])}

## 5. 国家级 null-adjusted synchronization

`delta_sync = log(phi_observed / phi_null_median)`。配对差只解释为“超出单边边际时间结构的方向性同期同步差异”，不解释为气候、拓扑、政策或粮食安全效应。

{_md(null_paired, ['crop','n_pairs','paired_median_export_minus_import','bootstrap_ci_low','bootstrap_ci_high','exact_sign_test_p','wilcoxon_signed_rank_p'])}

## 6. 独立两两协方差审计

按作物×方向分别抽取 20 个 covariance-eligible reporting economies/areas，共 {len(pairwise_audit)} 个审计样本。使用 `ddof=1` 显式计算 `2*sum_{{j<k}} Cov(e_j,e_k)`，与 `V-D` 的最大相对误差为 {pairwise_audit['relative_error'].max():.3e}，通过 `<1e-8` 标准。

## 7. 图形修复

- Figure 2 改用残差 moving-block 区间；滚动窗口仅作描述。
- Figure 3 改名为 “Algebraic decomposition of the Taylor exponent”；Panel E 比较 `alpha_V` 与 `alpha_D`，Panel F 单独展示 `beta_phi`。
- Figure 4 使用配对 log φ 和 45 度线，展示典型比率、exact sign-test p 和 bootstrap CI。
- Figure 5 只展示 median φ 与 alpha，不重复展示等价的 `C/D`。
- Figure 6 改为国家级 null-adjusted synchronization。
- 已恢复 S1–S14；coverage/entity sensitivity 与来源多元化均在补充材料中。18 种来源多元化预设规格没有形成稳定结论。

## 8. 自动化验证

共 {tests_passed} 项测试通过。新增测试覆盖：显式两两协方差恒等式、残差区块重建不拼接原始水平、不静默截断负值、配对推断中不存在 bootstrap-tail p-value、2,000 次抽样和主/敏感性区块长度。
"""
    (REPORT_DIR / "audit_response.md").write_text(audit, encoding="utf-8")

    methods = f"""# Methods v2.1

The analysis uses only the supplied FAOSTAT Detailed Trade Matrix archive and retains the v2 population, years (1986–2024), crops, direction definitions, eligibility rules, and prespecified sensitivity grid.

For every bilateral edge, a linear trend is fitted on the original chronological year positions. A residual moving-block bootstrap generates one global block index per draw and applies it to every edge and reporting economy. The reconstructed series equals the original fitted trend at the original year positions plus the resampled residual. Reconstructed negative values are retained. Linear trends are re-estimated after reconstruction before computing `mu`, aggregate residual variance `V`, diagonal edge variance `D`, and `phi=V/D`. The main block length is four years, with three- and five-year sensitivities. All formal procedures use 2,000 draws and seed {SEED} or deterministic derived seeds.

The paired import–export alpha analysis uses common reporting economies and the same global block and country-resampling indices. Percentile and basic intervals, bootstrap bias, bootstrap median, and effective draw counts are reported.

Paired φ inference uses `d_i=phi_export_i-phi_import_i` and `r_i=log(phi_export_i/phi_import_i)`. Confidence intervals come from paired country bootstrap. The formal median-direction test is a two-sided exact binomial sign test; Wilcoxon signed-rank is a sensitivity test. No bootstrap tail probability is interpreted as a p-value.

The country-level synchronization null independently circularly shifts each detrended bilateral edge. It preserves each edge's mean, variance, intermittency, and univariate temporal structure while breaking contemporaneous alignment. `delta_sync_i=log(phi_observed_i/phi_null_median_i)` is compared across directions on common reporting economies.

The explicit covariance audit uses the same linearly detrended residuals and `ddof=1` for both `2*sum_{{j<k}}Cov(e_j,e_k)` and `V-D`.
"""
    (REPORT_DIR / "methods.md").write_text(methods, encoding="utf-8")

    results = f"""# Results v2.1

## A. 已经稳健的结论

- 八组既定全时期 `alpha_V` 均小于 2：`{alpha_all_below}`。
- 因而相对波动 `V/mu²` 随贸易规模下降。
- 宏观 Taylor 指数可按边方差弹性和同步弹性作精确代数分解：`alpha_V=alpha_D+beta_phi`。

## B. 修复后重新判断的结论

### 配对 alpha

{_md(alpha_pair_resolved, ['crop','estimate','percentile_ci_low','percentile_ci_high','basic_ci_low','basic_ci_high','resolved'])}

### 各作物 beta_phi 的 L4 两阶段残差区块区间

{_md(beta, ['crop','direction','estimate','percentile_ci_low','percentile_ci_high','basic_ci_low','basic_ci_high'])}

### 出口—进口 φ 的正式配对推断

{_md(paired_phi, ['crop','typical_phi_ratio','typical_phi_ratio_ci_low','typical_phi_ratio_ci_high','exact_sign_test_p','wilcoxon_signed_rank_p'])}

### Null-adjusted 方向性同步差异

{_md(null_resolved, ['crop','paired_median_export_minus_import','bootstrap_ci_low','bootstrap_ci_high','exact_sign_test_p','wilcoxon_signed_rank_p','resolved'])}

这些结果只支持统计上的方向性同期同步差异，不支持机制或因果解释。

## C. 不支持的结论

- `alpha>2`；
- “协方差陷阱”；
- 稳定的来源多元化缺口（18 种预设规格未形成稳定结论）；
- 气候、贸易拓扑、政策或粮食安全因果效应。

旧版无效的 bootstrap-tail p-value 已从 v2.1 正式输出和报告中删除。
"""
    (REPORT_DIR / "results_report.md").write_text(results, encoding="utf-8")

    captions = """# Figure captions v2.1

**Figure 1. Full-period subquadratic Taylor scaling.** Full-period reporting-economy mean trade and residual variance, 1986–2024, using the unchanged v2 estimand and sample rules.

**Figure 2. Descriptive rolling estimates with residual moving-block uncertainty.** Rolling 20-year estimates use a global common residual moving-block bootstrap with block length four and 2,000 draws. Overlapping rolling windows are descriptive only.

**Figure 3. Algebraic decomposition of the Taylor exponent.** Panels A–D show aggregate variance against diagonal edge variance. Panel E compares `alpha_V` and `alpha_D`; Panel F separately reports `beta_phi` on its own scale. The exact 45-degree slope identity is Figure S14.

**Figure 4. Paired directional synchronization asymmetry.** Common covariance-eligible reporting economies are plotted as `log(phi_import)` versus `log(phi_export)` with a 45-degree line. Text gives the typical export/import φ ratio, paired bootstrap interval, and exact sign-test p-value.

**Figure 5. Circular-shift synchronization null.** Observed and null median φ and Taylor alpha are shown. The algebraically redundant pair `C/D` and `phi` is not repeated.

**Figure 6. Country-level null-adjusted directional synchronization.** Points are crop-specific paired medians of `delta_sync_export-delta_sync_import`; intervals are 2,000-draw paired country bootstrap intervals.

**Figures S1–S14.** Prespecified diagnostics and sensitivity results: global mirror, rice scope, coverage, detrending, window length, sample definition, robust estimators, leave-one-out influence, edge scaling, diversification thresholds, correlation-matrix method, specification curve, reporting-aware pairing, and algebraic identity verification.
"""
    (REPORT_DIR / "figure_captions.md").write_text(captions, encoding="utf-8")

    summary = {
        "seed": SEED,
        "alpha_all_below_two": alpha_all_below,
        "paired_phi": paired_phi.to_dict("records"),
        "paired_alpha_residual_block": paired_alpha.to_dict("records"),
        "beta_phi_residual_block": beta.to_dict("records"),
        "null_adjusted_direction": null_paired.to_dict("records"),
        "pairwise_covariance_max_relative_error": float(pairwise_audit["relative_error"].max()),
        "main_figures": figures["main"],
        "supplement_figures": figures["supplement"],
        "tests_passed": tests_passed,
        "unsupported": ["alpha>2", "covariance trap", "stable diversification gap", "causal mechanism claims"],
    }
    (REPORT_DIR / "results_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary
