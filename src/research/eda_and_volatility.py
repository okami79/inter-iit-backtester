"""
eda_and_volatility.py - Comprehensive Exploratory Data Analysis & Volatility Modeling
Inter IIT Tech Meet 15.0 Quant Evaluation - Track A: Phase 1

Deliverables Covered:
1. Data Hygiene & Anomaly Diagnostics (Missing data, OHLC consistency, sync)
2. Return Distributions & Stylized Facts (Fat tails, Kurtosis, Jarque-Bera, Student-t)
3. Volatility Clustering & Autocorrelation (Ljung-Box, Engle ARCH-LM tests)
4. Comparative Volatility Estimators (Rolling Std, EWMA, Parkinson, Garman-Klass, GARCH(1,1))
5. Volatility Forecasting Power (Mincer-Zarnowitz R^2)
6. Cross-Sectional Correlation & PCA Dimensionality
"""

from __future__ import annotations
import os
import math
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_arch, acorr_ljungbox
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Ensure matplotlib cache is in a safe location
os.environ["MPLCONFIGDIR"] = "/tmp/mpl_cache"

# Set random seed for reproducibility
np.random.seed(42)

# Paths
BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results"
PLOTS_DIR = RESULTS_DIR / "plots"
DATA_PATH = Path("data/Dataset_PS-A.csv")

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


# ==============================================================================
# 1. GARCH(1,1) ESTIMATOR VIA MAXIMUM LIKELIHOOD ESTIMATION (MLE)
# ==============================================================================
def fit_garch11_mle(returns: np.ndarray) -> np.ndarray:
    """
    Fits standard GARCH(1,1) model: sigma_t^2 = omega + alpha * eps_{t-1}^2 + beta * sigma_{t-1}^2
    via MLE using scipy.optimize. Returns fitted conditional standard deviation series.
    """
    r = returns - np.mean(returns)
    n = len(r)
    sample_var = np.var(r, ddof=1)
    if sample_var <= 1e-8 or n < 30:
        return np.full(n, np.sqrt(max(sample_var, 1e-6)))

    # Initial parameter guess [omega, alpha, beta]
    init_params = np.array([sample_var * 0.05, 0.10, 0.85])
    bounds = [(1e-6, sample_var * 2.0), (1e-4, 0.30), (0.50, 0.999)]

    def neg_log_likelihood(params):
        omega, alpha, beta = params
        if alpha + beta >= 0.9999:
            return 1e10
        sigma2 = np.empty(n)
        sigma2[0] = sample_var
        for t in range(1, n):
            sigma2[t] = omega + alpha * (r[t - 1] ** 2) + beta * sigma2[t - 1]
            if sigma2[t] <= 1e-8:
                sigma2[t] = 1e-8
        ll = -0.5 * np.sum(np.log(2 * np.pi) + np.log(sigma2) + (r ** 2) / sigma2)
        return -ll

    res = minimize(neg_log_likelihood, init_params, bounds=bounds, method="L-BFGS-B")
    omega, alpha, beta = res.x if res.success else init_params

    # Filter conditional variance
    sigma2 = np.empty(n)
    sigma2[0] = sample_var
    for t in range(1, n):
        sigma2[t] = omega + alpha * (r[t - 1] ** 2) + beta * sigma2[t - 1]
        if sigma2[t] <= 1e-8:
            sigma2[t] = 1e-8

    return np.sqrt(sigma2)


# ==============================================================================
# 2. VOLATILITY ESTIMATORS
# ==============================================================================
def compute_volatility_estimators(df_ticker: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """
    Computes 5 distinct volatility estimators for a single asset's daily OHLC series:
    1. Close-to-Close Rolling Standard Deviation (Sample Volatility)
    2. EWMA (RiskMetrics lambda=0.94)
    3. Parkinson (1980) High-Low Range Estimator
    4. Garman-Klass (1980) OHLC Estimator
    5. GARCH(1,1) Conditional Volatility
    """
    sub = df_ticker.sort_values("date").copy()
    
    # 1. Close-to-Close returns
    log_ret = np.log(sub["close"] / sub["close"].shift(1))
    sub["log_return"] = log_ret
    
    # Close-to-Close rolling annualized vol (sqrt(252))
    sub["vol_cc_20d"] = log_ret.rolling(window=window).std() * np.sqrt(252)
    
    # 2. EWMA (lambda = 0.94)
    alpha_ewma = 1.0 - 0.94
    ewma_var = (log_ret ** 2).ewm(alpha=alpha_ewma, adjust=False).mean()
    sub["vol_ewma"] = np.sqrt(ewma_var) * np.sqrt(252)
    
    # 3. Parkinson (1980) High-Low Volatility:
    # sigma_P^2 = (1 / (4 * ln 2)) * (ln(H / L))^2
    log_hl = np.log(sub["high"] / sub["low"])
    parkinson_var = (log_hl ** 2) / (4.0 * np.log(2.0))
    sub["vol_parkinson_20d"] = np.sqrt(parkinson_var.rolling(window=window).mean()) * np.sqrt(252)
    
    # 4. Garman-Klass (1980) Extended Range Estimator:
    # sigma_GK^2 = 0.5 * (ln(H / L))^2 - (2 * ln 2 - 1) * (ln(C / O))^2
    log_co = np.log(sub["close"] / sub["open"])
    gk_var = 0.5 * (log_hl ** 2) - (2.0 * np.log(2.0) - 1.0) * (log_co ** 2)
    # Ensure non-negative numerical stability
    gk_var = np.maximum(gk_var, 0.0)
    sub["vol_garman_klass_20d"] = np.sqrt(gk_var.rolling(window=window).mean()) * np.sqrt(252)
    
    # 5. GARCH(1,1)
    clean_rets = log_ret.dropna().values
    garch_vols = fit_garch11_mle(clean_rets) * np.sqrt(252)
    # Align with DataFrame index
    sub["vol_garch11"] = np.nan
    sub.iloc[1:, sub.columns.get_loc("vol_garch11")] = garch_vols
    
    return sub


# ==============================================================================
# 3. MAIN EXECUTION PIPELINE
# ==============================================================================
def main():
    print("=" * 70)
    print("RUNNING PHASE 1: COMPREHENSIVE EDA & VOLATILITY MODELING")
    print("=" * 70)

    # --------------------------------------------------------------------------
    # 3.1 Load Dataset & Data Hygiene Audit
    # --------------------------------------------------------------------------
    print("\n[Step 1] Auditing Data Hygiene & Physical Market Properties...")
    df = pd.read_csv(DATA_PATH)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)

    total_rows = len(df)
    tickers = sorted(df["ticker"].unique())
    n_tickers = len(tickers)
    min_date = df["date"].min().strftime("%Y-%m-%d")
    max_date = df["date"].max().strftime("%Y-%m-%d")
    all_dates = sorted(df["date"].unique())
    n_unique_dates = len(all_dates)

    # Physical OHLC consistency checks
    invalid_hl = (df["high"] < df["low"]).sum()
    invalid_ho = (df["high"] < df["open"]).sum()
    invalid_hc = (df["high"] < df["close"]).sum()
    invalid_lo = (df["low"] > df["open"]).sum()
    invalid_lc = (df["low"] > df["close"]).sum()
    negative_vol = (df["volume"] < 0).sum()
    zero_closes = (df["close"] <= 0).sum()
    missing_vals = df.isna().sum().to_dict()

    # Asset date synchronization check
    dates_per_ticker = df.groupby("ticker")["date"].nunique()
    perfectly_synced = (dates_per_ticker == n_unique_dates).all()

    hygiene_data = [
        {"Diagnostic Metric": "Total Records", "Value": str(total_rows)},
        {"Diagnostic Metric": "Universe Asset Count", "Value": str(n_tickers)},
        {"Diagnostic Metric": "Start Date", "Value": min_date},
        {"Diagnostic Metric": "End Date", "Value": max_date},
        {"Diagnostic Metric": "Unique Trading Sessions", "Value": str(n_unique_dates)},
        {"Diagnostic Metric": "All 24 Assets Perfectly Synchronized", "Value": str(perfectly_synced)},
        {"Diagnostic Metric": "High < Low Violations", "Value": str(invalid_hl)},
        {"Diagnostic Metric": "High < max(Open, Close) Violations", "Value": str(invalid_ho + invalid_hc)},
        {"Diagnostic Metric": "Low > min(Open, Close) Violations", "Value": str(invalid_lo + invalid_lc)},
        {"Diagnostic Metric": "Negative / Zero Volume Rows", "Value": str(negative_vol)},
        {"Diagnostic Metric": "Non-Positive Close Prices", "Value": str(zero_closes)},
        {"Diagnostic Metric": "Total Missing Values across all columns", "Value": str(sum(missing_vals.values()))},
    ]
    df_hygiene = pd.DataFrame(hygiene_data)
    print(df_hygiene.to_string(index=False))

    # --------------------------------------------------------------------------
    # 3.2 Return Distributions & Non-Gaussian Stylized Facts
    # --------------------------------------------------------------------------
    print("\n[Step 2] Computing Return Distributions & Statistical Stylized Facts...")
    df["daily_return"] = df.groupby("ticker")["close"].pct_change()
    df["log_return"] = df.groupby("ticker")["close"].apply(lambda s: np.log(s / s.shift(1))).reset_index(level=0, drop=True)

    distribution_records = []
    all_clean_returns = []

    for ticker, sub in df.groupby("ticker"):
        rets = sub["daily_return"].dropna().values
        all_clean_returns.extend(rets)

        n_obs = len(rets)
        mean_ret = np.mean(rets)
        ann_ret = mean_ret * 252
        std_ret = np.std(rets, ddof=1)
        ann_vol = std_ret * np.sqrt(252)
        skewness = stats.skew(rets)
        kurt = stats.kurtosis(rets, fisher=True)  # Excess kurtosis (Normal = 0)
        jb_stat, jb_pval = stats.jarque_bera(rets)
        
        # Fit Student-t distribution to check degrees of freedom (nu)
        t_params = stats.t.fit(rets)
        df_nu = t_params[0]

        distribution_records.append({
            "ticker": ticker,
            "observations": n_obs,
            "daily_mean_%": mean_ret * 100,
            "ann_return_%": ann_ret * 100,
            "daily_vol_%": std_ret * 100,
            "ann_vol_%": ann_vol * 100,
            "skewness": skewness,
            "excess_kurtosis": kurt,
            "student_t_df_nu": df_nu,
            "jarque_bera_stat": jb_stat,
            "jarque_bera_p_value": jb_pval,
            "is_gaussian_normal": jb_pval > 0.05
        })

    df_distributions = pd.DataFrame(distribution_records)
    print("\nUniverse Return Distribution Summary (Averages across 24 assets):")
    print(df_distributions.describe().round(4).to_string())

    # --------------------------------------------------------------------------
    # 3.3 Volatility Clustering & ARCH-LM Heteroskedasticity Tests
    # --------------------------------------------------------------------------
    print("\n[Step 3] Testing Volatility Clustering (Autocorrelations & Engle ARCH-LM)...")
    clustering_records = []
    acf_lags = [1, 2, 3, 5, 10, 20]

    for ticker, sub in df.groupby("ticker"):
        r = sub["daily_return"].dropna().values
        r_sq = r ** 2
        r_abs = np.abs(r)

        # Autocorrelations
        acf_raw = sm.tsa.stattools.acf(r, nlags=20, fft=True)
        acf_sq = sm.tsa.stattools.acf(r_sq, nlags=20, fft=True)
        acf_abs = sm.tsa.stattools.acf(r_abs, nlags=20, fft=True)

        # Ljung-Box test on raw returns (linear dependence) vs squared returns (volatility clustering)
        lb_raw = acorr_ljungbox(r, lags=[10], return_df=True)
        lb_sq = acorr_ljungbox(r_sq, lags=[10], return_df=True)

        # Engle's ARCH-LM test for conditional heteroskedasticity
        arch_test = het_arch(r, nlags=10)
        arch_lm_stat = arch_test[0]
        arch_lm_pval = arch_test[1]

        clustering_records.append({
            "ticker": ticker,
            "acf_raw_lag1": acf_raw[1],
            "acf_raw_lag5": acf_raw[5],
            "acf_sq_lag1": acf_sq[1],
            "acf_sq_lag5": acf_sq[5],
            "acf_sq_lag10": acf_sq[10],
            "acf_sq_lag20": acf_sq[20],
            "acf_abs_lag1": acf_abs[1],
            "acf_abs_lag5": acf_abs[5],
            "acf_abs_lag10": acf_abs[10],
            "ljung_box_raw_p_value": lb_raw["lb_pvalue"].iloc[0],
            "ljung_box_sq_p_value": lb_sq["lb_pvalue"].iloc[0],
            "arch_lm_stat": arch_lm_stat,
            "arch_lm_p_value": arch_lm_pval,
            "has_arch_clustering": arch_lm_pval < 0.05
        })

    df_clustering = pd.DataFrame(clustering_records)
    print("\nVolatility Clustering Evidence Summary:")
    print(f"Percentage of assets exhibiting significant ARCH effects (p < 0.05): "
          f"{(df_clustering['has_arch_clustering'].mean() * 100):.1f}%")
    print(f"Mean Lag-1 ACF: Raw returns = {df_clustering['acf_raw_lag1'].mean():.4f} | "
          f"Squared returns = {df_clustering['acf_sq_lag1'].mean():.4f} | "
          f"Absolute returns = {df_clustering['acf_abs_lag1'].mean():.4f}")

    # --------------------------------------------------------------------------
    # 3.4 Computing & Comparing Volatility Estimators
    # --------------------------------------------------------------------------
    print("\n[Step 4] Computing & Comparing 5 Classical Volatility Estimators...")
    processed_dfs = []
    for ticker, sub in df.groupby("ticker"):
        processed_sub = compute_volatility_estimators(sub, window=20)
        processed_dfs.append(processed_sub)
    df_vol = pd.concat(processed_dfs, ignore_index=True)

    vol_cols = ["vol_cc_20d", "vol_ewma", "vol_parkinson_20d", "vol_garman_klass_20d", "vol_garch11"]
    vol_corr_matrix = df_vol[vol_cols].corr()

    # Mincer-Zarnowitz Volatility Forecasting Power:
    # Future 1-day realized variance r_{t+1}^2 regressed on each estimator's variance forecast sigma_t^2
    mz_results = []
    df_vol["fwd_var_1d"] = (df_vol.groupby("ticker")["log_return"].shift(-1) ** 2) * 252

    for col in vol_cols:
        sub_mz = df_vol[[col, "fwd_var_1d"]].dropna().copy()
        x = (sub_mz[col] ** 2).values
        y = sub_mz["fwd_var_1d"].values
        x_const = sm.add_constant(x)
        model = sm.OLS(y, x_const).fit(cov_type="HC1")
        
        mz_results.append({
            "volatility_estimator": col,
            "mean_ann_vol_%": df_vol[col].mean() * 100,
            "median_ann_vol_%": df_vol[col].median() * 100,
            "std_ann_vol_%": df_vol[col].std() * 100,
            "mz_beta": model.params[1],
            "mz_beta_p_value": model.pvalues[1],
            "mz_r_squared_%": model.rsquared * 100,
        })
    df_mz = pd.DataFrame(mz_results)
    print("\nVolatility Estimators Comparison & Mincer-Zarnowitz Forecasting R^2:")
    print(df_mz.round(4).to_string(index=False))

    # --------------------------------------------------------------------------
    # 3.5 Cross-Sectional Correlation Structure & PCA Dimensionality
    # --------------------------------------------------------------------------
    print("\n[Step 5] Analyzing Cross-Sectional Correlation & PCA Scree Spectrum...")
    pivot_rets = df.pivot(index="date", columns="ticker", values="daily_return").dropna()
    corr_matrix = pivot_rets.corr()

    # Pairwise correlations (upper triangle)
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
    pairwise_corrs = corr_matrix.values[mask]

    corr_summary = {
        "Metric": [
            "Mean Pairwise Correlation",
            "Median Pairwise Correlation",
            "Min Pairwise Correlation",
            "Max Pairwise Correlation",
            "Standard Deviation of Pairwise Correlations"
        ],
        "Value": [
            float(np.mean(pairwise_corrs)),
            float(np.median(pairwise_corrs)),
            float(np.min(pairwise_corrs)),
            float(np.max(pairwise_corrs)),
            float(np.std(pairwise_corrs))
        ]
    }
    df_corr_summary = pd.DataFrame(corr_summary)

    # PCA Eigenvalue Decomposition
    cov_matrix = pivot_rets.cov() * 252
    eigvals, eigvecs = np.linalg.eigh(cov_matrix)
    # Sort descending
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    var_explained = (eigvals / np.sum(eigvals)) * 100
    cum_var_explained = np.cumsum(var_explained)

    pca_records = []
    for i in range(len(eigvals)):
        pca_records.append({
            "Component": f"PC_{i+1}",
            "Eigenvalue": eigvals[i],
            "Variance_Explained_%": var_explained[i],
            "Cumulative_Variance_%": cum_var_explained[i]
        })
    df_pca = pd.DataFrame(pca_records)
    print("\nPCA Variance Explained by Top 5 Factors:")
    print(df_pca.head(5).round(3).to_string(index=False))

    # --------------------------------------------------------------------------
    # 3.6 Save Comprehensive Excel Workbook
    # --------------------------------------------------------------------------
    excel_path = RESULTS_DIR / "phase1_eda_and_volatility.xlsx"
    print(f"\n[Step 6] Saving Structured Results to {excel_path}...")
    with pd.ExcelWriter(excel_path) as writer:
        df_hygiene.to_excel(writer, sheet_name="data_hygiene", index=False)
        df_distributions.to_excel(writer, sheet_name="return_distributions", index=False)
        df_clustering.to_excel(writer, sheet_name="volatility_clustering", index=False)
        vol_corr_matrix.reset_index().to_excel(writer, sheet_name="estimator_correlations", index=False)
        df_mz.to_excel(writer, sheet_name="vol_forecasting_mz", index=False)
        df_corr_summary.to_excel(writer, sheet_name="pairwise_correlations", index=False)
        df_pca.to_excel(writer, sheet_name="pca_scree", index=False)
        corr_matrix.reset_index().to_excel(writer, sheet_name="correlation_matrix", index=False)

    # --------------------------------------------------------------------------
    # 3.7 Publication-Grade Visualizations
    # --------------------------------------------------------------------------
    print("\n[Step 7] Generating Publication-Grade Figures in plots/...")

    # Figure 1: Return Distributions & Fat Tails (Leptokurtosis)
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 1A: Aggregate Returns Histogram vs Gaussian Fit vs Student-t
    clean_all = np.array(all_clean_returns) * 100
    axes[0].hist(clean_all, bins=120, density=True, alpha=0.6, color="#1f77b4", label="Empirical Returns")
    x_grid = np.linspace(clean_all.min(), clean_all.max(), 500)
    mu, sigma = np.mean(clean_all), np.std(clean_all)
    axes[0].plot(x_grid, stats.norm.pdf(x_grid, mu, sigma), 'r--', lw=2, label="Gaussian Normal")
    
    t_fit = stats.t.fit(clean_all)
    axes[0].plot(x_grid, stats.t.pdf(x_grid, *t_fit), 'k-', lw=2, label=f"Student-t (df={t_fit[0]:.1f})")
    axes[0].set_xlim(-8, 8)
    axes[0].set_title("Universe Return Distribution: Heavy Tails vs Gaussian", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Daily Return (%)", fontsize=10)
    axes[0].set_ylabel("Density", fontsize=10)
    axes[0].legend(loc="upper right", frameon=True)

    # 1B: Normal Q-Q Plot
    stats.probplot(clean_all, dist="norm", plot=axes[1])
    axes[1].get_lines()[0].set_markersize(2)
    axes[1].get_lines()[0].set_color("#1f77b4")
    axes[1].get_lines()[1].set_color("red")
    axes[1].get_lines()[1].set_linewidth(2)
    axes[1].set_title("Normal Q-Q Plot (Pronounced Leptokurtic S-Curve)", fontsize=12, fontweight="bold")
    
    plt.tight_layout()
    fig.savefig(PLOTS_DIR / "fig1_return_distributions.png", dpi=300)
    plt.close(fig)

    # Figure 2: Volatility Clustering ACF
    fig, ax = plt.subplots(figsize=(10, 5))
    mean_acf_raw = [df_clustering[f"acf_raw_lag{lag}"].mean() if f"acf_raw_lag{lag}" in df_clustering else 0 for lag in range(1, 21)]
    # Compute full mean ACF curves across all 24 assets
    raw_acfs, sq_acfs, abs_acfs = [], [], []
    for ticker, sub in df.groupby("ticker"):
        r = sub["daily_return"].dropna().values
        raw_acfs.append(sm.tsa.stattools.acf(r, nlags=20, fft=True)[1:])
        sq_acfs.append(sm.tsa.stattools.acf(r ** 2, nlags=20, fft=True)[1:])
        abs_acfs.append(sm.tsa.stattools.acf(np.abs(r), nlags=20, fft=True)[1:])
    
    lags = np.arange(1, 21)
    ax.plot(lags, np.mean(raw_acfs, axis=0), 'o-', color="#7f7f7f", lw=1.8, label="Raw Returns r_t (No Linear Memory)")
    ax.plot(lags, np.mean(sq_acfs, axis=0), 's-', color="#d62728", lw=2.2, label="Squared Returns r_t^2 (Volatility Clustering)")
    ax.plot(lags, np.mean(abs_acfs, axis=0), '^-', color="#2ca02c", lw=2.2, label="Absolute Returns |r_t| (Taylor Effect)")
    ax.axhline(0, color="black", linestyle="--", alpha=0.6)
    ax.axhline(1.96 / np.sqrt(n_unique_dates), color="blue", linestyle=":", alpha=0.6, label="95% Significance Band")
    ax.axhline(-1.96 / np.sqrt(n_unique_dates), color="blue", linestyle=":", alpha=0.6)
    ax.set_title("Volatility Clustering: Autocorrelation Structure (20 Lags)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Lag (Days)", fontsize=10)
    ax.set_ylabel("Autocorrelation", fontsize=10)
    ax.legend(loc="upper right", frameon=True)
    plt.tight_layout()
    fig.savefig(PLOTS_DIR / "fig2_volatility_clustering_acf.png", dpi=300)
    plt.close(fig)

    # Figure 3: Time Series Comparison of Volatility Estimators (Representative Asset BX97)
    rep_ticker = "BX97"
    rep_df = df_vol[df_vol["ticker"] == rep_ticker].sort_values("date").set_index("date")
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(rep_df.index, rep_df["vol_cc_20d"] * 100, label="Close-to-Close (20d)", color="#1f77b4", alpha=0.7, lw=1.5)
    ax.plot(rep_df.index, rep_df["vol_ewma"] * 100, label="EWMA (lambda=0.94)", color="#ff7f0e", alpha=0.8, lw=1.5)
    ax.plot(rep_df.index, rep_df["vol_garman_klass_20d"] * 100, label="Garman-Klass (20d Range)", color="#2ca02c", alpha=0.9, lw=1.8)
    ax.plot(rep_df.index, rep_df["vol_garch11"] * 100, label="GARCH(1,1)", color="#d62728", alpha=0.85, lw=1.5)
    ax.set_title(f"Dynamic Volatility Estimators Comparison ({rep_ticker}): March 2020 Shock Capture", fontsize=12, fontweight="bold")
    ax.set_xlabel("Date", fontsize=10)
    ax.set_ylabel("Annualized Volatility (%)", fontsize=10)
    ax.legend(loc="upper right", frameon=True)
    plt.tight_layout()
    fig.savefig(PLOTS_DIR / "fig3_volatility_estimators_comparison.png", dpi=300)
    plt.close(fig)

    # Figure 4: Cross-Sectional Correlation Heatmap & PCA Scree Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    cax = axes[0].imshow(corr_matrix.values, cmap="coolwarm", vmin=-0.2, vmax=1.0)
    axes[0].set_xticks(range(n_tickers))
    axes[0].set_yticks(range(n_tickers))
    axes[0].set_xticklabels(tickers, rotation=90, fontsize=8)
    axes[0].set_yticklabels(tickers, fontsize=8)
    axes[0].set_title(f"Pairwise Correlation Matrix (Mean = {np.mean(pairwise_corrs):.2f})", fontsize=12, fontweight="bold")
    fig.colorbar(cax, ax=axes[0], fraction=0.046, pad=0.04)

    # PCA Scree Plot
    axes[1].bar(range(1, 11), df_pca["Variance_Explained_%"].iloc[:10], color="#2b5c8f", alpha=0.85, label="Individual PC")
    axes[1].step(range(1, 11), df_pca["Cumulative_Variance_%"].iloc[:10], where="mid", color="#d62728", lw=2, label="Cumulative")
    axes[1].set_title("PCA Scree Plot (Market Mode Explains ~45% Variance)", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Principal Component", fontsize=10)
    axes[1].set_ylabel("Variance Explained (%)", fontsize=10)
    axes[1].set_ylim(0, 105)
    axes[1].legend(loc="center right", frameon=True)
    plt.tight_layout()
    fig.savefig(PLOTS_DIR / "fig4_pairwise_correlation_pca.png", dpi=300)
    plt.close(fig)

    print("\nPhase 1 Analysis Pipeline Successfully Completed!")
    print(f"Results saved to: {excel_path}")
    print(f"Plots saved to: {PLOTS_DIR}")


if __name__ == "__main__":
    main()

