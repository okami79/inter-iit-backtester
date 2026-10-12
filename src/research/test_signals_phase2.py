"""
test_signals_phase2.py - Rigorous Cross-Sectional Ranking Signal Evaluation
Inter IIT Tech Meet 15.0 Quant Evaluation - Track A: Phase 2

Deliverables Covered:
1. Candidate Signals Formulation (Reversal, Skip-Momentum, Vol-Adjusted Reversal, Composite)
2. Spearman Rank Information Coefficient (Mean IC, IC Volatility, IC IR, Cumulative IC)
3. Fama-MacBeth (1973) Cross-Sectional Regressions with Newey-West HAC correction
4. Quintile Portfolio Monotonicity & Long-Short Spreads (with HAC t-stats and CIs)
5. Top-Ranked Asset Concentration & Correlation Analysis (Deliverable 2 requirement)
6. Signal Turnover, Rank Autocorrelation & Net Spread after Transaction Costs
"""

from __future__ import annotations
import os
import math
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.environ["MPLCONFIGDIR"] = "/tmp/mpl_cache"
np.random.seed(42)

BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results"
PLOTS_DIR = RESULTS_DIR / "plots"
DATA_PATH = Path("data/Dataset_PS-A.csv")

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


# ==============================================================================
# 1. FEATURE & SIGNAL ENGINEERING
# ==============================================================================
def prepare_signals_and_targets(df: pd.DataFrame) -> pd.DataFrame:
    """
    Constructs clean, causal signals and forward return targets for all 24 assets.
    Strictly avoids lookahead bias (features use data <= t, targets use t+1 to t+H).
    """
    df = df.sort_values(["ticker", "date"]).copy()
    
    # Intraday Garman-Klass variance for volatility adjustment
    log_hl = np.log(df["high"] / df["low"])
    log_co = np.log(df["close"] / df["open"])
    gk_var = 0.5 * (log_hl ** 2) - (2.0 * np.log(2.0) - 1.0) * (log_co ** 2)
    df["gk_var"] = np.maximum(gk_var, 0.0)

    # Calculate by ticker group
    g = df.groupby("ticker")

    # 1. Volatility estimator: 20-day rolling annualized Garman-Klass
    df["vol_gk_20d"] = g["gk_var"].transform(lambda s: np.sqrt(s.rolling(20).mean()) * np.sqrt(252))

    # 2. Cumulative returns over various horizons
    df["ret_5d"] = g["close"].pct_change(5)
    df["ret_20d"] = g["close"].pct_change(20)
    df["ret_63d"] = g["close"].pct_change(63)
    df["ret_126d"] = g["close"].pct_change(126)

    # 3. Candidate Signals:
    # (A) Short-Term Reversal (Negative of recent return)
    # Rationale: Liquidity provision / overreaction bounce
    df["sig_rev_5d"] = -df["ret_5d"]
    df["sig_rev_20d"] = -df["ret_20d"]

    # (B) Classic Skip-Momentum (126d - 21d and 63d - 21d)
    # Rationale: Asness et al. (2013) & Jegadeesh & Titman (1993)
    # Strips out recent 21-day reversal to capture pure intermediate momentum
    close_lag21 = g["close"].shift(21)
    close_lag63 = g["close"].shift(63)
    close_lag126 = g["close"].shift(126)
    df["sig_skip_mom_63_21"] = (close_lag21 / close_lag63) - 1.0
    df["sig_skip_mom_126_21"] = (close_lag21 / close_lag126) - 1.0

    # (C) Volatility-Adjusted Reversal
    # Rationale: Normalizes reversal by range volatility to avoid betting on noise
    df["sig_vol_adj_rev_20d"] = df["sig_rev_20d"] / np.maximum(df["vol_gk_20d"], 0.05)

    # (D) Composite Trend + Pullback Signal
    # Standardize both signals cross-sectionally per day then sum:
    # Goes long stocks that have strong medium-term trends (126-21 mom) but are experiencing a short-term pullback (20d rev)
    # We will standardize this after cross-sectional grouping.

    # 4. Forward return targets (H = 5 and H = 20 trading days)
    df["fwd_ret_5d"] = g["close"].shift(-5) / df["close"] - 1.0
    df["fwd_ret_20d"] = g["close"].shift(-20) / df["close"] - 1.0

    return df


def add_composite_signal(df: pd.DataFrame) -> pd.DataFrame:
    """Cross-sectionally standardizes Skip-Mom and Vol-Adj Reversal to form a Composite Signal."""
    def zscore(s):
        std = s.std()
        return (s - s.mean()) / std if std > 1e-8 else s - s.mean()

    df = df.copy()
    z_mom = df.groupby("date")["sig_skip_mom_126_21"].transform(zscore)
    z_rev = df.groupby("date")["sig_vol_adj_rev_20d"].transform(zscore)
    df["sig_composite"] = 0.5 * z_mom + 0.5 * z_rev
    return df


# ==============================================================================
# 2. STATISTICAL TESTS: RANK IC, FAMA-MACBETH, QUINTILE SPREADS
# ==============================================================================
def compute_rank_ic(df: pd.DataFrame, signal_col: str, target_col: str) -> pd.DataFrame:
    """Computes daily Spearman Rank Correlation between signal and forward return."""
    clean = df[["date", "ticker", signal_col, target_col]].dropna()
    records = []
    for date, sub in clean.groupby("date"):
        if len(sub) >= 10:
            rho, pval = stats.spearmanr(sub[signal_col], sub[target_col])
            records.append({"date": date, "rank_ic": rho, "p_value": pval})
    return pd.DataFrame(records).sort_values("date").reset_index(drop=True)


def run_fama_macbeth(df: pd.DataFrame, signal_col: str, target_col: str, horizon: int) -> Dict[str, float]:
    """
    Executes Fama & MacBeth (1973) two-pass cross-sectional regression:
    R_{i, t+H} = alpha_t + lambda_t * z(Signal_{i, t}) + eps_{i, t}
    Evaluates time series average lambda with Newey-West HAC standard error.
    """
    clean = df[["date", "ticker", signal_col, target_col]].dropna()
    lambdas = []
    dates = []

    for date, sub in clean.groupby("date"):
        if len(sub) >= 10:
            x = sub[signal_col].values
            std_x = np.std(x)
            if std_x > 1e-8:
                x_norm = (x - np.mean(x)) / std_x
            else:
                continue
            y = sub[target_col].values
            x_const = sm.add_constant(x_norm)
            model = sm.OLS(y, x_const).fit()
            lambdas.append(model.params[1])
            dates.append(date)

    lambdas = np.array(lambdas)
    n = len(lambdas)
    if n < 10:
        return {"mean_lambda_%": np.nan, "t_stat": np.nan, "p_value": np.nan}

    # Time series OLS on constant with Newey-West lags = horizon - 1
    hac_lags = max(1, horizon - 1)
    mod = sm.OLS(lambdas, np.ones((n, 1))).fit(cov_type="HAC", cov_kwds={"maxlags": hac_lags, "use_correction": True})
    ci_low, ci_high = mod.conf_int()[0]

    return {
        "mean_lambda_%": mod.params[0] * 100,
        "se_pct": mod.bse[0] * 100,
        "t_stat": mod.tvalues[0],
        "p_value": mod.pvalues[0],
        "ci_low_%": ci_low * 100,
        "ci_high_%": ci_high * 100,
        "n_periods": n,
        "raw_lambdas": pd.Series(lambdas, index=dates)
    }


def evaluate_quintile_portfolios(
    df: pd.DataFrame, signal_col: str, target_col: str, horizon: int
) -> Tuple[Dict[str, float], pd.DataFrame]:
    """
    Forms 5 equal quintile portfolios on each date, computes forward returns,
    evaluates monotonicity, Long-Short spread, and HAC significance.
    """
    clean = df[["date", "ticker", signal_col, target_col]].dropna().copy()
    
    # Assign quintile 1 to 5 per date
    clean["quintile"] = clean.groupby("date")[signal_col].transform(
        lambda s: pd.qcut(s.rank(method="first"), 5, labels=False) + 1
    ).astype(int)

    # Daily quintile means
    q_means_daily = clean.pivot_table(index="date", columns="quintile", values=target_col, aggfunc="mean")
    q_means_overall = q_means_daily.mean() * 100

    # Long-Short spread (Q5 - Q1)
    spread = (q_means_daily[5] - q_means_daily[1]).dropna()
    y = spread.values
    n = len(y)

    hac_lags = max(1, horizon - 1)
    mod = sm.OLS(y, np.ones((n, 1))).fit(cov_type="HAC", cov_kwds={"maxlags": hac_lags, "use_correction": True})
    ci_low, ci_high = mod.conf_int()[0]

    # Monotonicity check: Spearman correlation between quintile index [1,2,3,4,5] and mean returns
    mono_corr, mono_p = stats.spearmanr(np.arange(1, 6), q_means_overall.values)

    results = {
        "Q1_mean_%": q_means_overall.get(1, np.nan),
        "Q2_mean_%": q_means_overall.get(2, np.nan),
        "Q3_mean_%": q_means_overall.get(3, np.nan),
        "Q4_mean_%": q_means_overall.get(4, np.nan),
        "Q5_mean_%": q_means_overall.get(5, np.nan),
        "Q5_minus_Q1_mean_%": mod.params[0] * 100,
        "spread_se_%": mod.bse[0] * 100,
        "spread_t_stat": mod.tvalues[0],
        "spread_p_value": mod.pvalues[0],
        "spread_ci_low_%": ci_low * 100,
        "spread_ci_high_%": ci_high * 100,
        "monotonicity_spearman": mono_corr,
        "n_rebalance_dates": n
    }

    return results, clean


# ==============================================================================
# 3. CONCENTRATION & CORRELATION ANALYSIS (DELIVERABLE 2 REQUIREMENT)
# ==============================================================================
def analyze_top_assets(clean_df: pd.DataFrame, return_matrix: pd.DataFrame) -> Dict[str, Any]:
    """
    Analyzes the correlation and concentration of top-ranked (Q5) and bottom-ranked (Q1) assets.
    Computes pairwise correlation among selected assets and selection concentration.
    """
    # 1. Average pairwise correlation within Q5 on each date
    q5_corrs = []
    q1_corrs = []

    for date, sub in clean_df.groupby("date"):
        q5_tickers = sub[sub["quintile"] == 5]["ticker"].tolist()
        q1_tickers = sub[sub["quintile"] == 1]["ticker"].tolist()

        if len(q5_tickers) >= 2 and date in return_matrix.index:
            # Look at past 60-day correlation among these tickers up to this date
            idx_loc = return_matrix.index.get_loc(date)
            if idx_loc >= 60:
                past_rets = return_matrix.iloc[idx_loc - 60 : idx_loc]
                q5_sub_corr = past_rets[q5_tickers].corr()
                mask_q5 = np.triu(np.ones_like(q5_sub_corr, dtype=bool), k=1)
                vals_q5 = q5_sub_corr.values[mask_q5]
                if len(vals_q5) > 0 and not np.isnan(vals_q5).all():
                    q5_corrs.append(np.nanmean(vals_q5))

                q1_sub_corr = past_rets[q1_tickers].corr()
                mask_q1 = np.triu(np.ones_like(q1_sub_corr, dtype=bool), k=1)
                vals_q1 = q1_sub_corr.values[mask_q1]
                if len(vals_q1) > 0 and not np.isnan(vals_q1).all():
                    q1_corrs.append(np.nanmean(vals_q1))

    # 2. Asset selection frequency (concentration in specific assets)
    q5_counts = clean_df[clean_df["quintile"] == 5]["ticker"].value_counts(normalize=True)
    # Herfindahl-Hirschman Index (HHI) of selection frequency
    hhi = np.sum(q5_counts.values ** 2)
    effective_n = 1.0 / hhi

    return {
        "mean_q5_pairwise_corr": float(np.mean(q5_corrs)) if q5_corrs else np.nan,
        "mean_q1_pairwise_corr": float(np.mean(q1_corrs)) if q1_corrs else np.nan,
        "selection_hhi": float(hhi),
        "effective_n_assets": float(effective_n),
        "top_frequent_assets": q5_counts.head(5).to_dict()
    }


# ==============================================================================
# 4. SIGNAL TURNOVER & TRANSACTION COST IMPACT
# ==============================================================================
def evaluate_turnover_and_costs(
    clean_df: pd.DataFrame, gross_spread_pct: float, horizon_days: int
) -> Dict[str, float]:
    """
    Computes Rank Autocorrelation, one-way portfolio turnover per rebalance,
    and net return spread after subtracting slippage (5 bps) + commission (1 bp).
    """
    pivot_ranks = clean_df.pivot(index="date", columns="ticker", values="quintile")
    
    # Rebalance every 'horizon_days' sessions
    rebalance_dates = pivot_ranks.index[::horizon_days]
    rebal_ranks = pivot_ranks.loc[rebalance_dates]

    # Q5 portfolio membership turnover
    # On each rebalance, what fraction of stocks in Q5 are replaced?
    turnovers = []
    for i in range(1, len(rebal_ranks)):
        prev_q5 = set(rebal_ranks.iloc[i - 1][rebal_ranks.iloc[i - 1] == 5].index)
        curr_q5 = set(rebal_ranks.iloc[i][rebal_ranks.iloc[i] == 5].index)
        if len(prev_q5) > 0 and len(curr_q5) > 0:
            # Fraction of names that changed
            turnover = len(curr_q5 - prev_q5) / len(curr_q5)
            turnovers.append(turnover)

    avg_turnover = float(np.mean(turnovers)) if turnovers else 0.50

    # Backtester friction: 5 bps slippage + 1 bp commission = 6 bps (0.06%) one-way per trade
    # For a Long-Short spread, turnover occurs in both Q5 and Q1 legs:
    # Round-trip cost per replaced position = 2 * 0.06% = 0.12%
    cost_per_rebalance_pct = 2.0 * avg_turnover * 0.12  # in percent
    net_spread_pct = gross_spread_pct - cost_per_rebalance_pct

    return {
        "rebalance_cadence_days": horizon_days,
        "mean_basket_turnover_%": avg_turnover * 100,
        "transaction_friction_per_rebal_%": cost_per_rebalance_pct,
        "gross_spread_%": gross_spread_pct,
        "net_spread_%": net_spread_pct,
        "cost_drag_ratio_%": (cost_per_rebalance_pct / abs(gross_spread_pct) * 100) if gross_spread_pct != 0 else np.nan
    }


# ==============================================================================
# 5. MAIN PHASE 2 EXECUTION
# ==============================================================================
def main():
    print("=" * 75)
    print("RUNNING PHASE 2: CROSS-SECTIONAL SIGNAL FORMULATION & STATISTICAL PROOF")
    print("=" * 75)

    df_raw = pd.read_csv(DATA_PATH)
    df_raw["date"] = pd.to_datetime(df_raw["date"])
    
    # Return matrix for correlation checks
    pivot_rets = df_raw.pivot(index="date", columns="ticker", values="close").pct_change().dropna()

    print("\n[Step 1] Constructing Candidate Causal Signals...")
    df = prepare_signals_and_targets(df_raw)
    df = add_composite_signal(df)

    candidate_signals = [
        ("sig_rev_5d", "5d Short-Term Reversal", 5),
        ("sig_rev_20d", "20d Short-Term Reversal", 20),
        ("sig_vol_adj_rev_20d", "20d Vol-Adjusted Reversal", 20),
        ("sig_skip_mom_63_21", "63-21d Skip-Momentum", 20),
        ("sig_skip_mom_126_21", "126-21d Skip-Momentum", 20),
        ("sig_composite", "Composite (126-21 Mom + 20d Vol-Adj Rev)", 20),
    ]

    # --------------------------------------------------------------------------
    # Step 2: Full Statistical Evaluation
    # --------------------------------------------------------------------------
    ic_summary_list = []
    fm_summary_list = []
    quintile_summary_list = []
    concentration_summary_list = []
    turnover_summary_list = []

    ic_series_dict = {}
    fm_series_dict = {}
    quintile_data_dict = {}

    print("\n[Step 2] Executing Rigorous Econometric Battery across Candidate Signals...")

    for sig_col, sig_name, horizon in candidate_signals:
        target_col = f"fwd_ret_{horizon}d"
        print(f"  -> Testing {sig_name} (Horizon = {horizon}d)...")

        # 2A. Rank Information Coefficient
        df_ic = compute_rank_ic(df, sig_col, target_col)
        mean_ic = df_ic["rank_ic"].mean()
        std_ic = df_ic["rank_ic"].std()
        ic_ir = mean_ic / std_ic if std_ic > 0 else np.nan
        t_stat_ic = ic_ir * np.sqrt(len(df_ic)) if not np.isnan(ic_ir) else np.nan
        pct_positive = (df_ic["rank_ic"] > 0).mean() * 100

        ic_summary_list.append({
            "Signal Name": sig_name,
            "Horizon (Days)": horizon,
            "Mean Rank IC": mean_ic,
            "IC Standard Deviation": std_ic,
            "IC Information Ratio (IR)": ic_ir,
            "IC t-statistic": t_stat_ic,
            "IC p-value": 2.0 * (1.0 - stats.norm.cdf(abs(t_stat_ic))),
            "% Periods Positive IC": pct_positive,
            "Total Dates": len(df_ic)
        })
        ic_series_dict[sig_name] = df_ic

        # 2B. Fama-MacBeth Regressions
        fm_res = run_fama_macbeth(df, sig_col, target_col, horizon)
        fm_summary_list.append({
            "Signal Name": sig_name,
            "Horizon (Days)": horizon,
            "Mean Lambda (Slope %)": fm_res["mean_lambda_%"],
            "HAC Standard Error %": fm_res["se_pct"],
            "t-statistic": fm_res["t_stat"],
            "p-value": fm_res["p_value"],
            "95% CI Low %": fm_res["ci_low_%"],
            "95% CI High %": fm_res["ci_high_%"],
            "N Periods": fm_res["n_periods"]
        })
        fm_series_dict[sig_name] = fm_res["raw_lambdas"]

        # 2C. Quintile Monotonicity & Long-Short Spread
        q_res, clean_quintiles = evaluate_quintile_portfolios(df, sig_col, target_col, horizon)
        quintile_summary_list.append({
            "Signal Name": sig_name,
            "Horizon (Days)": horizon,
            "Q1 Mean %": q_res["Q1_mean_%"],
            "Q2 Mean %": q_res["Q2_mean_%"],
            "Q3 Mean %": q_res["Q3_mean_%"],
            "Q4 Mean %": q_res["Q4_mean_%"],
            "Q5 Mean %": q_res["Q5_mean_%"],
            "Long-Short (Q5 - Q1) %": q_res["Q5_minus_Q1_mean_%"],
            "Spread HAC t-stat": q_res["spread_t_stat"],
            "Spread p-value": q_res["spread_p_value"],
            "95% CI Low %": q_res["spread_ci_low_%"],
            "95% CI High %": q_res["spread_ci_high_%"],
            "Monotonicity Spearman": q_res["monotonicity_spearman"]
        })
        quintile_data_dict[sig_name] = q_res

        # 2D. Concentration & Pairwise Correlation of Selected Assets
        conc_res = analyze_top_assets(clean_quintiles, pivot_rets)
        concentration_summary_list.append({
            "Signal Name": sig_name,
            "Mean Pairwise Corr in Long (Q5)": conc_res["mean_q5_pairwise_corr"],
            "Mean Pairwise Corr in Short (Q1)": conc_res["mean_q1_pairwise_corr"],
            "Universe Baseline Corr": 0.306,
            "Selection HHI": conc_res["selection_hhi"],
            "Effective Number of Assets (N_eff)": conc_res["effective_n_assets"],
        })

        # 2E. Turnover & Net Spread
        cost_res = evaluate_turnover_and_costs(clean_quintiles, q_res["Q5_minus_Q1_mean_%"], horizon)
        turnover_summary_list.append({
            "Signal Name": sig_name,
            "Rebalance Cadence (Days)": horizon,
            "Gross Spread %": cost_res["gross_spread_%"],
            "One-Way Turnover %": cost_res["mean_basket_turnover_%"],
            "Friction Cost per Rebal %": cost_res["transaction_friction_per_rebal_%"],
            "Net Spread %": cost_res["net_spread_%"],
            "Cost Drag %": cost_res["cost_drag_ratio_%"]
        })

    df_ic_summary = pd.DataFrame(ic_summary_list)
    df_fm_summary = pd.DataFrame(fm_summary_list)
    df_quintile_summary = pd.DataFrame(quintile_summary_list)
    df_conc_summary = pd.DataFrame(concentration_summary_list)
    df_turnover_summary = pd.DataFrame(turnover_summary_list)

    print("\n" + "=" * 75)
    print("SUMMARY 1: RANK INFORMATION COEFFICIENT (IC & IC IR)")
    print("=" * 75)
    print(df_ic_summary[["Signal Name", "Mean Rank IC", "IC Information Ratio (IR)", "IC t-statistic", "IC p-value", "% Periods Positive IC"]].round(4).to_string(index=False))

    print("\n" + "=" * 75)
    print("SUMMARY 2: FAMA-MACBETH (1973) CROSS-SECTIONAL REGRESSIONS")
    print("=" * 75)
    print(df_fm_summary[["Signal Name", "Mean Lambda (Slope %)", "HAC Standard Error %", "t-statistic", "p-value"]].round(4).to_string(index=False))

    print("\n" + "=" * 75)
    print("SUMMARY 3: QUINTILE MONOTONICITY & LONG-SHORT SPREADS")
    print("=" * 75)
    print(df_quintile_summary[["Signal Name", "Q1 Mean %", "Q3 Mean %", "Q5 Mean %", "Long-Short (Q5 - Q1) %", "Spread HAC t-stat", "Spread p-value", "Monotonicity Spearman"]].round(4).to_string(index=False))

    print("\n" + "=" * 75)
    print("SUMMARY 4: NET SPREAD AFTER TRANSACTION COSTS & TURNOVER")
    print("=" * 75)
    print(df_turnover_summary.round(4).to_string(index=False))

    print("\n" + "=" * 75)
    print("SUMMARY 5: CONCENTRATION & CORRELATION OF TOP-RANKED ASSETS")
    print("=" * 75)
    print(df_conc_summary.round(4).to_string(index=False))

    # --------------------------------------------------------------------------
    # Step 3: Save Workbook
    # --------------------------------------------------------------------------
    excel_path = RESULTS_DIR / "phase2_signal_analysis.xlsx"
    print(f"\n[Step 3] Saving Comprehensive Phase 2 Workbook to {excel_path}...")
    with pd.ExcelWriter(excel_path) as writer:
        df_ic_summary.to_excel(writer, sheet_name="rank_ic_summary", index=False)
        df_fm_summary.to_excel(writer, sheet_name="fama_macbeth_summary", index=False)
        df_quintile_summary.to_excel(writer, sheet_name="quintile_portfolios", index=False)
        df_turnover_summary.to_excel(writer, sheet_name="turnover_and_costs", index=False)
        df_conc_summary.to_excel(writer, sheet_name="top_asset_concentration", index=False)

    # --------------------------------------------------------------------------
    # Step 4: Generate Publication Figures
    # --------------------------------------------------------------------------
    print("\n[Step 4] Generating Publication-Grade Figures in plots/...")

    # Figure 5: Cumulative Rank IC Curves
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = ["#1f77b4", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#ff7f0e"]
    for idx, (sig_col, sig_name, _) in enumerate(candidate_signals):
        df_ic = ic_series_dict[sig_name]
        cum_ic = df_ic["rank_ic"].cumsum()
        ax.plot(df_ic["date"], cum_ic, label=sig_name, color=colors[idx % len(colors)], lw=2.0)
    ax.set_title("Cumulative Rank Information Coefficient (2016 - 2022)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Date", fontsize=11)
    ax.set_ylabel("Cumulative Spearman Rank IC", fontsize=11)
    ax.legend(loc="upper left", frameon=True, fontsize=9)
    plt.tight_layout()
    fig.savefig(PLOTS_DIR / "fig5_cumulative_rank_ic.png", dpi=300)
    plt.close(fig)

    # Figure 6: Quintile Monotonicity Profiles
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharey=True)
    axes = axes.flatten()
    for idx, (sig_col, sig_name, _) in enumerate(candidate_signals):
        q_data = quintile_data_dict[sig_name]
        q_vals = [q_data[f"Q{q}_mean_%"] for q in range(1, 6)]
        bar_colors = ["#d62728", "#ff9896", "#c7c7c7", "#98df8a", "#2ca02c"]
        axes[idx].bar(range(1, 6), q_vals, color=bar_colors, edgecolor="black", alpha=0.85)
        axes[idx].set_title(f"{sig_name}\nMono: {q_data['monotonicity_spearman']:.2f} | L-S: {q_data['Q5_minus_Q1_mean_%']:.2f}% (t={q_data['spread_t_stat']:.2f})", fontsize=10, fontweight="bold")
        axes[idx].set_xlabel("Quintile", fontsize=9)
        axes[idx].set_ylabel("Mean Forward Return (%)", fontsize=9)
        axes[idx].set_xticks(range(1, 6))
        axes[idx].grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    fig.savefig(PLOTS_DIR / "fig6_quintile_monotonicity.png", dpi=300)
    plt.close(fig)

    # Figure 7: Cumulative Fama-MacBeth Premium
    fig, ax = plt.subplots(figsize=(12, 6))
    for idx, (sig_col, sig_name, _) in enumerate(candidate_signals):
        raw_l = fm_series_dict[sig_name]
        cum_l = raw_l.cumsum() * 100
        ax.plot(cum_l.index, cum_l.values, label=sig_name, color=colors[idx % len(colors)], lw=2.0)
    ax.set_title("Cumulative Fama-MacBeth Cross-Sectional Factor Premium (\\sum \\lambda_t)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Date", fontsize=11)
    ax.set_ylabel("Cumulative Factor Slope (%)", fontsize=11)
    ax.legend(loc="upper left", frameon=True, fontsize=9)
    plt.tight_layout()
    fig.savefig(PLOTS_DIR / "fig7_fama_macbeth_slopes.png", dpi=300)
    plt.close(fig)

    print("\nPhase 2 Signal Research Pipeline Finished Successfully!")
    print(f"Results saved to: {excel_path}")
    print(f"Plots saved to: {PLOTS_DIR}")


if __name__ == "__main__":
    main()

