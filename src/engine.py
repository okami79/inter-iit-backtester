"""
src/engine.py - Mandatory Participant Strategy Entry Point
Inter IIT Tech Meet 15.0 Quant Evaluation

Track A: Volatility-Adjusted Cross-Sectional Strategy
- Proven Ranking Signal: 20-Day Volatility-Adjusted Short-Term Reversal
- Proven Position Sizing: Range-Based Inverse Volatility (Garman-Klass OHLC)
- Rebalancing Cadence: 20 Trading Days (Monthly) with zero daily turnover churn
- Strictly enforces Gross Leverage <= 1.0, Dollar Neutrality, and Non-Negative Capital
"""

from __future__ import annotations
import math
import random
from typing import Dict, List, Optional, Tuple

from strategy_base import BaseStrategy, Context, Bar


class ParticipantStrategy(BaseStrategy):
    """
    Participant Strategy implementing BaseStrategy.
    Implements the statistically justified Track A strategy:
    20-day Garman-Klass Volatility-Adjusted Cross-Sectional Reversal with Inverse-Vol Sizing.
    """

    def initialize(self, context: Context) -> None:
        """
        Executed once before backtest simulation starts.
        Initialize indicators, lookback buffers, and configuration parameters.
        """
        random.seed(42)

        # ----------------------------------------------------------------------
        # Track A Configuration
        # ----------------------------------------------------------------------
        self.rebalance_cadence: int = 20       # Rebalance every 20 trading sessions
        self.lookback_window: int = 25         # Sufficient buffer for 20-day returns & vol
        self.top_k: int = 5                    # 5 long assets, 5 short assets (out of 24)

        # Sliding window histories for Track A assets
        # Stores list of tuples: (open, high, low, close)
        self.ohlc_history: Dict[str, List[Tuple[float, float, float, float]]] = {}

        # ----------------------------------------------------------------------
        # Track B Configuration (Preserved for compatibility)
        # ----------------------------------------------------------------------
        self.track_b_decision_interval: int = 10

        print(f"[ParticipantStrategy] Initialized with universe: {len(context.universe)} assets.")
        if context.symbol:
            print(f"[ParticipantStrategy] Single-asset mode active for ticker: {context.symbol}")

    def on_bar(self, context: Context, bars: Dict[str, Bar]) -> None:
        """
        Executed at every simulation step t.
        Dispatches to Track A or Track B depending on universe size.
        """
        num_assets = len(bars)
        if num_assets > 1:
            self._handle_track_a(context, bars)
        else:
            self._handle_track_b(context, bars)

    def _handle_track_a(self, context: Context, bars: Dict[str, Bar]) -> None:
        """
        Track A Cross-Sectional Strategy:
        1. Ingests daily OHLC bars to compute 20-day returns and Garman-Klass volatility.
        2. Computes the 20-day Volatility-Adjusted Reversal signal.
        3. On rebalance dates (every 20 days), selects top 5 long and bottom 5 short assets,
           sizes them via Inverse-Volatility weighting (sum(|w_i|) = 1.0, sum(w_i) = 0.0),
           and queues target weights for execution at t+1 Open.
        4. On non-rebalance dates, sets target weights to {} to prevent daily drift rebalancing churn.
        """
        # 1. Update historical OHLC sliding windows
        for ticker, bar in bars.items():
            if ticker not in self.ohlc_history:
                self.ohlc_history[ticker] = []
            self.ohlc_history[ticker].append((bar.open, bar.high, bar.low, bar.close))
            if len(self.ohlc_history[ticker]) > self.lookback_window:
                self.ohlc_history[ticker].pop(0)

        # 2. Check rebalancing cadence
        # If not a rebalance day, queue empty target weights so backtester holds positions
        # without paying daily turnover friction costs.
        if context.step < self.rebalance_cadence or (context.step % self.rebalance_cadence != 0):
            context.set_target_weights({})
            return

        # 3. Compute Signals and Volatilities across all assets
        signals: Dict[str, float] = {}
        vols: Dict[str, float] = {}

        for ticker, ohlc_list in self.ohlc_history.items():
            if len(ohlc_list) < 21:
                continue

            # 20-day return: Close_t / Close_{t-20} - 1.0
            close_t = ohlc_list[-1][3]
            close_t_minus_20 = ohlc_list[-21][3]
            if close_t_minus_20 <= 0:
                continue
            ret_20d = (close_t / close_t_minus_20) - 1.0

            # 20-day Garman-Klass Volatility Estimator
            # sigma_GK^2 = 0.5 * (ln(H/L))^2 - (2*ln(2) - 1) * (ln(C/O))^2
            gk_vars = []
            for op, hi, lo, cl in ohlc_list[-20:]:
                if hi > 0 and lo > 0 and op > 0 and cl > 0 and hi >= lo:
                    log_hl = math.log(hi / lo)
                    log_co = math.log(cl / op)
                    var_i = 0.5 * (log_hl ** 2) - (2.0 * math.log(2.0) - 1.0) * (log_co ** 2)
                    gk_vars.append(max(0.0, var_i))
                else:
                    gk_vars.append(0.0)

            mean_gk_var = sum(gk_vars) / len(gk_vars) if gk_vars else 0.0004
            ann_vol = math.sqrt(mean_gk_var) * math.sqrt(252)
            ann_vol = max(ann_vol, 0.05)  # Floor at 5% to avoid divide-by-zero

            # 20-day Volatility-Adjusted Short-Term Reversal Signal
            # Higher score = more oversold relative to volatility -> higher long rank
            sig_val = -ret_20d / ann_vol

            signals[ticker] = sig_val
            vols[ticker] = ann_vol

        # Need at least 2 * top_k valid assets
        if len(signals) < 2 * self.top_k:
            context.set_target_weights({})
            return

        # 4. Rank assets cross-sectionally
        sorted_tickers = sorted(signals.keys(), key=lambda t: signals[t], reverse=True)
        long_basket = sorted_tickers[:self.top_k]
        short_basket = sorted_tickers[-self.top_k:]

        # 5. Position Sizing: Inverse-Volatility Weighting
        # Long leg: total weight = +0.50
        inv_vols_long = {t: 1.0 / vols[t] for t in long_basket}
        sum_inv_long = sum(inv_vols_long.values())
        w_long = {t: 0.50 * (inv_vols_long[t] / sum_inv_long) for t in long_basket}

        # Short leg: total weight = -0.50
        inv_vols_short = {t: 1.0 / vols[t] for t in short_basket}
        sum_inv_short = sum(inv_vols_short.values())
        w_short = {t: -0.50 * (inv_vols_short[t] / sum_inv_short) for t in short_basket}

        # Merge target weights
        target_weights = {**w_long, **w_short}

        # Strictly enforce zero leverage violations: sum(|w_i|) <= 1.0
        gross_lev = sum(abs(w) for w in target_weights.values())
        if gross_lev > 1.0:
            scale = 1.0 / gross_lev
            target_weights = {t: w * scale for t, w in target_weights.items()}

        # Submit target allocations to context
        context.set_target_weights(target_weights)

    def _handle_track_b(self, context: Context, bars: Dict[str, Bar]) -> None:
        """
        Track B Logic (Preserved for compatibility and sanity checks):
        Continuous high-frequency microstructure stream.
        """
        ticker = context.symbol or next(iter(bars.keys()))
        bar = bars[ticker]
        t = context.step

        if context.portfolio_value <= 0.0:
            context.set_target_weights({ticker: 0.0})
            return

        if t % self.track_b_decision_interval != 0:
            return

        ofi = bar.ofi
        if ofi > 0.2:
            target_weight = 0.5
        elif ofi < -0.2:
            target_weight = -0.5
        else:
            target_weight = 0.0

        context.set_target_weights({ticker: target_weight})
