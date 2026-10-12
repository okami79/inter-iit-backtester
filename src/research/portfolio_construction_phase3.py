"""
portfolio_construction_phase3.py - Advanced Portfolio Optimization & Volatility Sizing
Inter IIT Tech Meet 15.0 Quant Evaluation - Track A: Phase 3

Deliverables Covered:
1. Volatility-Adjusted Position Sizing Justification (Deliverable 1)
2. Portfolio Construction Models Comparison:
   - Model 1: Equal-Weight Long-Short (1/N Naive Baseline)
   - Model 2: Inverse-Volatility Weighted Long-Short (Garman-Klass Scaling)
   - Model 3: Equal Risk Contribution (Risk Parity) with Ledoit-Wolf Covariance
   - Model 4: Mean-Variance Optimization (MVO) with Ledoit-Wolf Shrinkage & L1 Constraint
   - Model 5: Volatility-Targeted Dynamic Sizing (Target Vol = 15%, De-leveraging)
3. Rebalancing Cadence Optimization (5d, 10d, 20d, 40d) & Turnover Cost Frontier
4. Rigorous Risk Metrics: Sharpe, Sortino, Calmar, Max Drawdown, Net CAGR, Leverage Checks
"""

from __future__ import annotations
import os
import math
from pathlib import Path
from typing import Dict, List, Tuple, Any
import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize
from sklearn.covariance import LedoitWolf
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

# Transaction costs matching backtester.py
SLIPPAGE_RATE = 0.0005     # 5 bps
COMMISSION_RATE = 0.0001   # 1 bp
ONE_WAY_COST = SLIPPAGE_RATE + COMMISSION_RATE  # 6 bps total per side


# ==============================================================================
# 1. DATA PREPARATION & SIGNAL GENERATION
# ==============================================================================
def prepare_backtest_data(df_raw: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Constructs prices, returns, Garman-Klass volatility, and the winning Phase 2 signal:
    20-day Volatility-Adjusted Reversal = -(Close_t / Close_{t-20} - 1) / Vol_GK_20d
    Strictly causal (no lookahead).
    """
    df = df_raw.sort_values(["ticker", "date"]).copy()
    
    # Intraday Garman-Klass variance
    log_hl = np.log(df["high"] / df["low"])
    log_co = np.log(df["close"] / df["open"])
    gk_var = 0.5 * (log_hl ** 2) - (2.0 * np.log(2.0) - 1.0) * (log_co ** 2)
    df["gk_var"] = np.maximum(gk_var, 0.0)

    g = df.groupby("ticker")
    df["vol_gk_20d"] = g["gk_var"].transform(lambda s: np.sqrt(s.rolling(20).mean()) * np.sqrt(252))
    df["ret_20d"] = g["close"].pct_change(20)
    
    # Winning signal from Phase 2
    df["signal"] = -df["ret_20d"] / np.maximum(df["vol_gk_20d"], 0.05)

    # Pivot tables for rapid backtesting
    close_pivot = df.pivot(index="date", columns="ticker", values="close")
    open_pivot = df.pivot(index="date", columns="ticker", values="open")
    signal_pivot = df.pivot(index="date", columns="ticker", values="signal")
    vol_pivot = df.pivot(index="date", columns="ticker", values="vol_gk_20d")

    return close_pivot, open_pivot, signal_pivot, vol_pivot


# ==============================================================================
# 2. PORTFOLIO OPTIMIZATION ALGORITHMS
# ==============================================================================
def get_equal_weight_long_short(
    signals_t: pd.Series, top_k: int = 5
) -> Dict[str, float]:
    """
    Model 1: Naive 1/N Equal-Weight Long-Short.
    Long top_k assets (+0.5 total weight), Short bottom_k assets (-0.5 total weight).
    Gross leverage = 1.0. Dollar-neutral.
    """
    valid = signals_t.dropna().sort_values(ascending=False)
    if len(valid) < 2 * top_k:
        return {}
    
    long_tickers = valid.index[:top_k]
    short_tickers = valid.index[-top_k:]

    weights = {}
    for t in long_tickers:
        weights[t] = 0.5 / top_k
    for t in short_tickers:
        weights[t] = -0.5 / top_k
        
    return weights


def get_inverse_vol_long_short(
    signals_t: pd.Series, vols_t: pd.Series, top_k: int = 5
) -> Dict[str, float]:
    """
    Model 2: Inverse-Volatility Weighted Long-Short.
    Weights within the long/short legs are proportional to 1 / vol.
    Low-volatility assets receive higher weight; high-volatility names are scaled down.
    Gross leverage = 1.0. Dollar-neutral.
    """
    valid = signals_t.dropna().sort_values(ascending=False)
    if len(valid) < 2 * top_k:
        return {}

    long_tickers = valid.index[:top_k]
    short_tickers = valid.index[-top_k:]

    # Long leg inverse-vol
    inv_vol_long = {t: 1.0 / max(vols_t.get(t, 0.25), 0.05) for t in long_tickers}
    sum_long = sum(inv_vol_long.values())
    w_long = {t: 0.5 * (v / sum_long) for t, v in inv_vol_long.items()}

    # Short leg inverse-vol
    inv_vol_short = {t: 1.0 / max(vols_t.get(t, 0.25), 0.05) for t in short_tickers}
    sum_short = sum(inv_vol_short.values())
    w_short = {t: -0.5 * (v / sum_short) for t, v in inv_vol_short.items()}

    weights = {**w_long, **w_short}
    return weights


def compute_ledoit_wolf_cov(return_window: pd.DataFrame) -> np.ndarray:
    """Computes well-conditioned covariance matrix via Ledoit & Wolf (2004) shrinkage."""
    clean_rets = return_window.dropna(axis=1)
    if clean_rets.shape[0] < 10 or clean_rets.shape[1] < 2:
        return np.eye(return_window.shape[1]) * 0.04
    lw = LedoitWolf(assume_centered=False)
    cov_lw = lw.fit(clean_rets.values).covariance_ * 252  # Annualized
    return cov_lw


def get_risk_parity_weights(
    cov_matrix: np.ndarray, tickers: List[str], target_budget: float = 0.5
) -> Dict[str, float]:
    """
    Equal Risk Contribution (ERC / Risk Parity) solver (Maillard et al. 2010):
    Finds weights such that w_i * (Sigma * w)_i = 1/N * (w^T * Sigma * w).
    """
    n = len(tickers)
    if n == 0:
        return {}
    if n == 1:
        return {tickers[0]: target_budget}

    init_w = np.full(n, 1.0 / n)
    bounds = [(1e-4, 1.0) for _ in range(n)]

    def erc_objective(w):
        port_var = w.T @ cov_matrix @ w
        if port_var <= 1e-8:
            return 1e5
        marginal_risk = cov_matrix @ w
        risk_contributions = w * marginal_risk
        # Target equal risk contribution = port_var / n
        target_rc = port_var / n
        return np.sum((risk_contributions - target_rc) ** 2)

    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    res = minimize(erc_objective, init_w, method="SLSQP", bounds=bounds, constraints=constraints)
    opt_w = res.x if res.success else init_w
    opt_w = opt_w / np.sum(opt_w) * target_budget

    return {tickers[i]: float(opt_w[i]) for i in range(n)}


def get_risk_parity_long_short(
    signals_t: pd.Series, return_history: pd.DataFrame, top_k: int = 5
) -> Dict[str, float]:
    """
    Model 3: Risk Parity Long-Short with Ledoit-Wolf Covariance.
    Solves Equal Risk Contribution separately for Long and Short legs.
    """
    valid = signals_t.dropna().sort_values(ascending=False)
    if len(valid) < 2 * top_k:
        return {}

    long_tickers = list(valid.index[:top_k])
    short_tickers = list(valid.index[-top_k:])

    # Covariance for long leg
    cov_long = compute_ledoit_wolf_cov(return_history[long_tickers])
    w_long = get_risk_parity_weights(cov_long, long_tickers, target_budget=0.5)

    # Covariance for short leg
    cov_short = compute_ledoit_wolf_cov(return_history[short_tickers])
    w_short_raw = get_risk_parity_weights(cov_short, short_tickers, target_budget=0.5)
    w_short = {t: -w_short_raw[t] for t in short_tickers}

    return {**w_long, **w_short}


def get_mvo_long_short(
    signals_t: pd.Series, return_history: pd.DataFrame, top_k: int = 5, gamma: float = 2.0
) -> Dict[str, float]:
    """
    Model 4: Mean-Variance Optimization (Markowitz 1952) with Ledoit-Wolf Shrinkage Covariance.
    Maximizes expected return minus risk penalty: w^T mu - (gamma/2) w^T Sigma w
    Subject to sum(|w|) <= 1.0 and sum(w) = 0 (dollar neutrality).
    """
    valid = signals_t.dropna().sort_values(ascending=False)
    if len(valid) < 2 * top_k:
        return {}

    selected_tickers = list(valid.index[:top_k]) + list(valid.index[-top_k:])
    n = len(selected_tickers)
    
    # Expected returns proportional to normalized signals
    sub_sigs = signals_t[selected_tickers].values
    mu = (sub_sigs - np.mean(sub_sigs)) / (np.std(sub_sigs) + 1e-8) * 0.10  # Annualized alpha scale

    cov = compute_ledoit_wolf_cov(return_history[selected_tickers])

    # Initial guess: inverse vol
    vols = np.sqrt(np.diag(cov))
    init_w = np.zeros(n)
    init_w[:top_k] = 0.5 / top_k
    init_w[top_k:] = -0.5 / top_k

    def neg_utility(w):
        ret = w @ mu
        var = w.T @ cov @ w
        return -(ret - 0.5 * gamma * var)

    bounds = [(-0.25, 0.25) for _ in range(n)]
    constraints = [
        {"type": "eq", "fun": lambda w: np.sum(w)},            # Dollar neutral
        {"type": "ineq", "fun": lambda w: 1.0 - np.sum(np.abs(w))} # Gross leverage <= 1.0
    ]

    res = minimize(neg_utility, init_w, method="SLSQP", bounds=bounds, constraints=constraints)
    opt_w = res.x if res.success else init_w
    # Normalize leverage to 1.0 if not zero
    gross = np.sum(np.abs(opt_w))
    if gross > 1e-4:
        opt_w = opt_w / gross

    return {selected_tickers[i]: float(opt_w[i]) for i in range(n)}


# ==============================================================================
# 3. HIGH-FIDELITY EVENT-DRIVEN SIMULATION ENGINE (MATCHES BACKTESTER.PY)
# ==============================================================================
def run_simulation(
    close_pivot: pd.DataFrame,
    open_pivot: pd.DataFrame,
    signal_pivot: pd.DataFrame,
    vol_pivot: pd.DataFrame,
    model_func: Any,
    rebalance_cadence: int = 20,
    vol_targeting: bool = False,
    target_ann_vol: float = 0.15,
    initial_cash: float = 100_000.0,
    top_k: int = 5
) -> Dict[str, Any]:
    """
    Simulates portfolio execution matching the exact specifications of backtester.py:
    - Signals calculated at step t using data <= t
    - Target weights executed at Open_{t+1} with 5 bps slippage + 1 bp commission
    - Daily Mark-to-Market at Close_{t+1}
    - Gross leverage <= 1.0 strictly enforced
    - Strict Non-Negative Capital / Bankruptcy check
    """
    dates = close_pivot.index
    n_days = len(dates)
    tickers = close_pivot.columns.tolist()

    cash = initial_cash
    equity = initial_cash
    positions: Dict[str, float] = {t: 0.0 for t in tickers}  # number of shares
    
    equity_curve = [initial_cash]
    equity_dates = [dates[0]]
    daily_returns = []
    daily_turnovers = []
    leverage_records = []
    
    total_commission = 0.0
    total_slippage = 0.0
    total_turnover_value = 0.0
    trades_count = 0

    daily_pct_rets = close_pivot.pct_change()

    current_weights: Dict[str, float] = {}
    pending_target_weights: Dict[str, float] = {}

    for t in range(n_days - 1):
        curr_date = dates[t]
        next_date = dates[t + 1]

        # ----------------------------------------------------------------------
        # 1. Strategy Decision Step at Date t Close
        # ----------------------------------------------------------------------
        if t >= 60 and (t % rebalance_cadence == 0):
            sigs_t = signal_pivot.loc[curr_date]
            vols_t = vol_pivot.loc[curr_date]
            ret_hist = daily_pct_rets.iloc[max(0, t - 60) : t]

            # Generate target weights
            if model_func == "equal_weight":
                raw_weights = get_equal_weight_long_short(sigs_t, top_k=top_k)
            elif model_func == "inverse_vol":
                raw_weights = get_inverse_vol_long_short(sigs_t, vols_t, top_k=top_k)
            elif model_func == "risk_parity":
                raw_weights = get_risk_parity_long_short(sigs_t, ret_hist, top_k=top_k)
            elif model_func == "mvo":
                raw_weights = get_mvo_long_short(sigs_t, ret_hist, top_k=top_k)
            else:
                raw_weights = {}

            # Volatility Targeting (Model 5 overlay):
            # Scales down gross leverage when universe/portfolio volatility exceeds target
            if vol_targeting and raw_weights:
                cov_t = compute_ledoit_wolf_cov(ret_hist[list(raw_weights.keys())])
                w_vec = np.array([raw_weights[k] for k in raw_weights.keys()])
                exp_ann_vol = math.sqrt(max(w_vec.T @ cov_t @ w_vec, 1e-6))
                if exp_ann_vol > target_ann_vol:
                    vol_scale = target_ann_vol / exp_ann_vol
                    raw_weights = {k: v * vol_scale for k, v in raw_weights.items()}

            # Enforce gross leverage <= 1.0 (RuleGuard check)
            gross_lev = sum(abs(w) for w in raw_weights.values())
            if gross_lev > 1.000001:
                scale = 1.0 / gross_lev
                raw_weights = {k: v * scale for k, v in raw_weights.items()}

            pending_target_weights = raw_weights

        # ----------------------------------------------------------------------
        # 2. Execution Step at Date t+1 Open (AccountingLedger check)
        # ----------------------------------------------------------------------
        if pending_target_weights:
            next_opens = open_pivot.loc[next_date]
            base_equity = equity

            # Bankruptcy check
            if base_equity <= 0.0:
                positions = {tk: 0.0 for tk in tickers}
                equity = 0.0
                break

            # Calculate desired shares
            rebal_turnover_dollars = 0.0
            
            for ticker in tickers:
                target_w = pending_target_weights.get(ticker, 0.0)
                open_p = next_opens.get(ticker, 0.0)
                if open_p <= 0.0:
                    continue

                desired_shares = (base_equity * target_w) / open_p
                current_shares = positions.get(ticker, 0.0)
                delta_shares = desired_shares - current_shares

                if abs(delta_shares) > 1e-4:
                    trade_val = abs(delta_shares) * open_p
                    rebal_turnover_dollars += trade_val

                    # Fill price with slippage
                    fill_price = open_p * (1.0 + SLIPPAGE_RATE) if delta_shares > 0 else open_p * (1.0 - SLIPPAGE_RATE)
                    comm = trade_val * COMMISSION_RATE
                    slip = trade_val * SLIPPAGE_RATE

                    cash -= (delta_shares * fill_price + comm)
                    positions[ticker] = desired_shares
                    total_commission += comm
                    total_slippage += slip
                    trades_count += 1

            total_turnover_value += rebal_turnover_dollars
            daily_turnovers.append(rebal_turnover_dollars / base_equity if base_equity > 0 else 0.0)
            pending_target_weights = {}
        else:
            daily_turnovers.append(0.0)

        # ----------------------------------------------------------------------
        # 3. Mark-to-Market at Date t+1 Close
        # ----------------------------------------------------------------------
        next_closes = close_pivot.loc[next_date]
        holdings_value = sum(positions[tk] * next_closes[tk] for tk in tickers)
        new_equity = cash + holdings_value

        if new_equity <= 0.0:
            new_equity = 0.0
            equity = 0.0
            positions = {tk: 0.0 for tk in tickers}
            equity_curve.append(0.0)
            equity_dates.append(next_date)
            break

        day_ret = (new_equity - equity) / equity
        daily_returns.append(day_ret)

        # Record gross leverage: sum(|shares * price|) / equity
        cur_gross_lev = sum(abs(positions[tk] * next_closes[tk]) for tk in tickers) / new_equity
        leverage_records.append(cur_gross_lev)

        equity = new_equity
        equity_curve.append(equity)
        equity_dates.append(next_date)

    # --------------------------------------------------------------------------
    # 4. Compute Performance & Risk Metrics
    # --------------------------------------------------------------------------
    returns_arr = np.array(daily_returns)
    equity_arr = np.array(equity_curve)

    total_return_pct = (equity / initial_cash - 1.0) * 100
    n_years = len(daily_returns) / 252.0
    cagr_pct = ((equity / initial_cash) ** (1.0 / max(n_years, 0.1)) - 1.0) * 100

    mean_ret = np.mean(returns_arr) if len(returns_arr) > 0 else 0.0
    std_ret = np.std(returns_arr, ddof=1) if len(returns_arr) > 1 else 1e-6
    ann_vol_pct = std_ret * np.sqrt(252) * 100
    sharpe = (mean_ret / std_ret) * np.sqrt(252) if std_ret > 0 else 0.0

    # Sortino (downside deviation)
    downside_rets = returns_arr[returns_arr < 0]
    downside_std = np.std(downside_rets, ddof=1) if len(downside_rets) > 1 else 1e-6
    sortino = (mean_ret / downside_std) * np.sqrt(252) if downside_std > 0 else 0.0

    # Drawdown
    peaks = np.maximum.accumulate(equity_arr)
    drawdowns = (equity_arr - peaks) / peaks
    max_dd_pct = abs(np.min(drawdowns)) * 100
    calmar = cagr_pct / max_dd_pct if max_dd_pct > 0 else 0.0

    # Win rate
    win_rate = (np.sum(returns_arr > 0) / len(returns_arr)) * 100 if len(returns_arr) > 0 else 0.0

    return {
        "final_equity": equity,
        "total_return_pct": total_return_pct,
        "cagr_pct": cagr_pct,
        "ann_vol_pct": ann_vol_pct,
        "sharpe_ratio": sharpe,
        "sortino_ratio": sortino,
        "max_drawdown_pct": max_dd_pct,
        "calmar_ratio": calmar,
        "win_rate_pct": win_rate,
        "mean_gross_leverage": float(np.mean(leverage_records)) if leverage_records else 0.0,
        "max_gross_leverage": float(np.max(leverage_records)) if leverage_records else 0.0,
        "total_trades": trades_count,
        "total_commission_paid": total_commission,
        "total_slippage_paid": total_slippage,
        "total_friction_cost": total_commission + total_slippage,
        "total_turnover_pct": float(np.sum(daily_turnovers)) * 100,
        "equity_curve": pd.Series(equity_arr, index=equity_dates),
        "drawdown_curve": pd.Series(drawdowns, index=equity_dates),
        "daily_returns": pd.Series(returns_arr, index=dates[1:len(returns_arr)+1])
    }


# ==============================================================================
# 4. MAIN PHASE 3 EXECUTION PIPELINE
# ==============================================================================
def main():
    print("=" * 75)
    print("RUNNING PHASE 3: PORTFOLIO OPTIMIZATION & VOLATILITY-ADJUSTED SIZING")
    print("=" * 75)

    df_raw = pd.read_csv(DATA_PATH)
    df_raw["date"] = pd.to_datetime(df_raw["date"])

    print("\n[Step 1] Preparing Causal Market & Volatility State...")
    close_p, open_p, sig_p, vol_p = prepare_backtest_data(df_raw)

    # --------------------------------------------------------------------------
    # Step 2: Compare 5 Portfolio Construction Models (Cadence = 20 Days)
    # --------------------------------------------------------------------------
    print("\n[Step 2] Benchmarking 5 Distinct Portfolio Construction Models (20-Day Cadence)...")

    models_to_test = [
        ("Equal-Weight Long-Short (1/N Baseline)", "equal_weight", False, 0.15),
        ("Inverse-Volatility Weighting (Garman-Klass)", "inverse_vol", False, 0.15),
        ("Risk Parity / ERC (Ledoit-Wolf Covariance)", "risk_parity", False, 0.15),
        ("Mean-Variance Optimization (Ledoit-Wolf)", "mvo", False, 0.15),
        ("Vol-Targeted Inverse-Vol (Target Vol = 15%)", "inverse_vol", True, 0.15),
    ]

    model_results = []
    equity_curves_dict = {}
    drawdown_curves_dict = {}

    for name, m_func, vol_target, t_vol in models_to_test:
        print(f"  -> Simulating {name}...")
        res = run_simulation(
            close_p, open_p, sig_p, vol_p,
            model_func=m_func,
            rebalance_cadence=20,
            vol_targeting=vol_target,
            target_ann_vol=t_vol,
            top_k=5
        )
        model_results.append({
            "Portfolio Model": name,
            "CAGR (%)": res["cagr_pct"],
            "Annualized Vol (%)": res["ann_vol_pct"],
            "Sharpe Ratio": res["sharpe_ratio"],
            "Sortino Ratio": res["sortino_ratio"],
            "Max Drawdown (%)": res["max_drawdown_pct"],
            "Calmar Ratio": res["calmar_ratio"],
            "Daily Win Rate (%)": res["win_rate_pct"],
            "Mean Leverage": res["mean_gross_leverage"],
            "Max Leverage": res["max_gross_leverage"],
            "Total Turnover (%)": res["total_turnover_pct"],
            "Total Friction ($)": res["total_friction_cost"],
            "Final Equity ($)": res["final_equity"]
        })
        equity_curves_dict[name] = res["equity_curve"]
        drawdown_curves_dict[name] = res["drawdown_curve"]

    df_models = pd.DataFrame(model_results)

    # --------------------------------------------------------------------------
    # Step 3: Rebalance Cadence Sensitivity (Trade-off Frontier)
    # --------------------------------------------------------------------------
    print("\n[Step 3] Optimizing Rebalancing Cadence (5d, 10d, 20d, 40d)...")
    cadences = [5, 10, 20, 40]
    cadence_results = []

    for cad in cadences:
        print(f"  -> Simulating Cadence = {cad} days (Inverse-Vol Sizing)...")
        res_cad = run_simulation(
            close_p, open_p, sig_p, vol_p,
            model_func="inverse_vol",
            rebalance_cadence=cad,
            vol_targeting=False,
            top_k=5
        )
        cadence_results.append({
            "Rebalance Cadence (Days)": cad,
            "CAGR (%)": res_cad["cagr_pct"],
            "Annualized Vol (%)": res_cad["ann_vol_pct"],
            "Sharpe Ratio": res_cad["sharpe_ratio"],
            "Max Drawdown (%)": res_cad["max_drawdown_pct"],
            "Calmar Ratio": res_cad["calmar_ratio"],
            "Annual Turnover (%)": res_cad["total_turnover_pct"] / (len(close_p) / 252),
            "Total Friction Paid ($)": res_cad["total_friction_cost"],
            "Final Equity ($)": res_cad["final_equity"]
        })

    df_cadences = pd.DataFrame(cadence_results)

    # --------------------------------------------------------------------------
    # Step 4: Display Summary Tables
    # --------------------------------------------------------------------------
    print("\n" + "=" * 75)
    print("PHASE 3 RESULTS: PORTFOLIO CONSTRUCTION MODELS COMPARISON (CADENCE = 20d)")
    print("=" * 75)
    print(df_models[["Portfolio Model", "CAGR (%)", "Sharpe Ratio", "Sortino Ratio", "Max Drawdown (%)", "Calmar Ratio", "Total Turnover (%)"]].round(3).to_string(index=False))

    print("\n" + "=" * 75)
    print("PHASE 3 RESULTS: REBALANCING CADENCE TRADE-OFF FRONTIER")
    print("=" * 75)
    print(df_cadences.round(3).to_string(index=False))

    # --------------------------------------------------------------------------
    # Step 5: Save Excel Workbook
    # --------------------------------------------------------------------------
    excel_path = RESULTS_DIR / "phase3_portfolio_construction.xlsx"
    print(f"\n[Step 5] Saving Structured Results to {excel_path}...")
    with pd.ExcelWriter(excel_path) as writer:
        df_models.to_excel(writer, sheet_name="model_comparisons", index=False)
        df_cadences.to_excel(writer, sheet_name="cadence_sensitivity", index=False)

    # --------------------------------------------------------------------------
    # Step 6: Generate Publication Figures
    # --------------------------------------------------------------------------
    print("\n[Step 6] Generating Publication-Grade Figures in plots/...")

    # Figure 8: Cumulative Equity Curves Comparison
    fig, ax = plt.subplots(figsize=(13, 6))
    colors = ["#7f7f7f", "#2ca02c", "#1f77b4", "#9467bd", "#d62728"]
    for idx, (name, _, _, _) in enumerate(models_to_test):
        eq = equity_curves_dict[name]
        ax.plot(eq.index, eq.values, label=f"{name}", color=colors[idx], lw=2.2 if "Vol" in name else 1.6)
    ax.axhline(100_000, color="black", linestyle="--", alpha=0.5, label="Initial Capital ($100k)")
    ax.set_title("Portfolio Equity Growth: Comparison of 5 Sizing & Optimization Models (2016 - 2022)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Date", fontsize=11)
    ax.set_ylabel("Portfolio Equity ($)", fontsize=11)
    ax.legend(loc="upper left", frameon=True, fontsize=9)
    plt.tight_layout()
    fig.savefig(PLOTS_DIR / "fig8_equity_curves_comparison.png", dpi=300)
    plt.close(fig)

    # Figure 9: Underwater Drawdown Curves
    fig, ax = plt.subplots(figsize=(13, 5))
    for idx, (name, _, _, _) in enumerate(models_to_test):
        dd = drawdown_curves_dict[name] * 100
        ax.plot(dd.index, dd.values, label=name, color=colors[idx], lw=1.8)
    ax.axhline(0, color="black", linestyle="-", alpha=0.7)
    ax.set_title("Underwater Drawdown Profile: Volatility Sizing vs Equal Weighting", fontsize=13, fontweight="bold")
    ax.set_xlabel("Date", fontsize=11)
    ax.set_ylabel("Drawdown (%)", fontsize=11)
    ax.legend(loc="lower left", frameon=True, fontsize=9)
    plt.tight_layout()
    fig.savefig(PLOTS_DIR / "fig9_drawdown_underwater_curves.png", dpi=300)
    plt.close(fig)

    # Figure 10: Rebalancing Cadence Frontier
    fig, ax1 = plt.subplots(figsize=(10, 5))
    color_sharpe = "#1f77b4"
    color_turnover = "#d62728"
    
    ax1.set_xlabel("Rebalance Cadence (Trading Days)", fontsize=11)
    ax1.set_ylabel("Net Sharpe Ratio", color=color_sharpe, fontsize=11)
    line1 = ax1.plot(df_cadences["Rebalance Cadence (Days)"], df_cadences["Sharpe Ratio"], "o-", color=color_sharpe, lw=2.2, label="Net Sharpe Ratio")
    ax1.tick_params(axis="y", labelcolor=color_sharpe)

    ax2 = ax1.twinx()
    ax2.set_ylabel("Total Trading Friction ($)", color=color_turnover, fontsize=11)
    line2 = ax2.plot(df_cadences["Rebalance Cadence (Days)"], df_cadences["Total Friction Paid ($)"], "s--", color=color_turnover, lw=2.0, label="Friction Cost ($)")
    ax2.tick_params(axis="y", labelcolor=color_turnover)

    plt.title("Rebalance Cadence Frontier: Alpha Decay vs Transaction Cost Erosion", fontsize=12, fontweight="bold")
    plt.tight_layout()
    fig.savefig(PLOTS_DIR / "fig10_rebalance_cadence_turnover_frontier.png", dpi=300)
    plt.close(fig)

    print("\nPhase 3 Portfolio Optimization Pipeline Completed Successfully!")
    print(f"Results saved to: {excel_path}")
    print(f"Plots saved to: {PLOTS_DIR}")


if __name__ == "__main__":
    main()

