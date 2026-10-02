"""Minute-level execution modelling for hourly-bar signals.

Builds an "execution close panel" aligned to the signal index: for the signal
bar labelled ``T`` (close known at wall-clock ``T+1h``), the panel holds the
last real 1-minute close at-or-before ``T+1h+delay_minutes``. Forward returns
computed from this panel embed the execution delay, so the existing eval
functions (`run_light_backtest`, `compute_ic_series`, ...) are used unchanged
with ``execution_delay_bars=0``.
"""

import pandas as pd

from src.data.panel_io import fetch_minute_exec_closes
from src.research.momentum_eval import run_light_backtest
from src.signals.cs_momentum import apply_traded_mask, compute_forward_returns


def build_execution_close_panel(exec_closes_long, signal_index, signal_close_wide=None):
    """Align bucketed 1m execution closes to a signal index.

    Parameters
    ----------
    exec_closes_long : DataFrame with columns (bucket_ts, symbol, exec_close),
        as returned by ``fetch_minute_exec_closes``. Buckets with no trades
        are absent.
    signal_index : DatetimeIndex of the signal panel (hourly bar-open labels).
    signal_close_wide : optional wide close panel used as a final fallback for
        cells with no 1m history at all (e.g. 1m file coverage starting later
        than hourly). Falling back to the signal close assumes a zero-delay
        fill there, so the count is reported in the diagnostics.

    Returns
    -------
    (exec_close_wide, diagnostics) where exec_close_wide is indexed exactly by
    ``signal_index`` and diagnostics counts how each cell was filled:
    - n_direct: bucket had a trade in the delay window's trailing hour
    - n_locf_fallback: filled from an earlier bucket's close (still <= exec
      time, no lookahead)
    - n_signal_close_fallback: filled from ``signal_close_wide``
    - n_unpriceable: still NaN after all fills (excluding rows before a
      symbol's first observation, which are expected to be NaN)
    - direct_fill_mask: boolean panel, True only where the fill is a real
      print from the cell's own delay-window bucket. LOCF / signal-close
      fallbacks are stale prints — lookahead-safe but not executable prices —
      so returns priced off them should be excluded (pass this mask to
      ``compute_execution_forward_returns``), exactly as ``trades == 0``
      hourly bars are excluded from close-to-close analysis.
    """
    signal_index = pd.DatetimeIndex(signal_index)

    if exec_closes_long.empty:
        raise ValueError("exec_closes_long is empty — no 1m data in the window")

    wide = exec_closes_long.pivot(
        index="bucket_ts", columns="symbol", values="exec_close"
    ).sort_index()

    # Dense hourly grid spanning both the buckets and the signal index, so
    # LOCF can carry values across empty buckets and onto signal labels.
    grid_start = min(wide.index.min(), signal_index.min())
    grid_end = max(wide.index.max(), signal_index.max())
    grid = pd.date_range(grid_start, grid_end, freq="1h", tz="UTC")

    on_grid = wide.reindex(grid)
    direct = on_grid.reindex(signal_index)
    filled = on_grid.ffill().reindex(signal_index)

    if signal_close_wide is not None:
        aligned_fallback = signal_close_wide.reindex(
            index=signal_index, columns=filled.columns
        )
        final = filled.fillna(aligned_fallback)
    else:
        final = filled

    # Leading NaNs before a symbol's first observation anywhere are expected,
    # not unpriceable cells.
    ever_observed = final.ffill().notna()
    n_direct = int(direct.notna().sum().sum())
    n_locf = int((filled.notna() & direct.isna()).sum().sum())
    n_signal_fallback = int((final.notna() & filled.isna()).sum().sum())
    unpriceable_mask = final.isna() & ever_observed
    diagnostics = {
        "n_cells": int(final.shape[0] * final.shape[1]),
        "n_direct": n_direct,
        "n_locf_fallback": n_locf,
        "n_signal_close_fallback": n_signal_fallback,
        "n_unpriceable": int(unpriceable_mask.sum().sum()),
        "fallback_by_symbol": (filled.notna() & direct.isna()).sum(),
        "direct_fill_mask": direct.notna(),
    }
    return final, diagnostics


def compute_execution_forward_returns(
    exec_close_wide,
    holding_period_bars,
    log_returns=True,
    direct_fill_mask=None,
):
    """Forward returns priced at execution closes: entry ``t+delay``, exit
    ``t+H+delay`` in wall-clock close terms. Same math as
    ``compute_forward_returns`` — the delay is embedded in the panel, so use
    ``execution_delay_bars=0`` downstream.

    ``direct_fill_mask`` (the panel from ``build_execution_close_panel``
    diagnostics) restricts returns to real-print fills: a cell survives only
    if **both** the entry bar ``t`` and the exit bar ``t+H`` were priced from
    their own bucket's trades rather than LOCF / signal-close fallbacks. The
    backtest then drops any period holding an unpriceable name instead of
    pricing a fill that could not have happened.
    """
    fwd = compute_forward_returns(
        exec_close_wide,
        holding_period_bars=holding_period_bars,
        log_returns=log_returns,
    )
    if direct_fill_mask is not None:
        fwd = apply_traded_mask(
            fwd, direct_fill_mask, forward_bars=int(holding_period_bars)
        )
    return fwd


def get_execution_forward_returns(
    engine,
    signal_index,
    delay_minutes,
    holding_period_bars,
    symbols=None,
    signal_close_wide=None,
    log_returns=True,
    require_direct_fills=True,
):
    """Convenience wrapper: fetch → build panel → forward returns.

    ``require_direct_fills=True`` (default) excludes returns whose entry or
    exit fill came from an LOCF / signal-close fallback rather than a real
    print — see ``compute_execution_forward_returns``. Set False to price
    every cell (the pre-exclusion behaviour) for diagnostics.

    Returns (fwd_ret_exec, exec_close_wide, diagnostics).
    """
    signal_index = pd.DatetimeIndex(signal_index)
    exec_closes = fetch_minute_exec_closes(
        engine,
        start_ts=signal_index.min(),
        end_ts=signal_index.max(),
        delay_minutes=delay_minutes,
        symbols=symbols,
    )
    exec_close_wide, diagnostics = build_execution_close_panel(
        exec_closes, signal_index, signal_close_wide=signal_close_wide
    )
    fwd_ret_exec = compute_execution_forward_returns(
        exec_close_wide,
        holding_period_bars=holding_period_bars,
        log_returns=log_returns,
        direct_fill_mask=(
            diagnostics["direct_fill_mask"] if require_direct_fills else None
        ),
    )
    return fwd_ret_exec, exec_close_wide, diagnostics


class ExecutionPriceCache:
    """Execution close panels per delay, fetched once and reused.

    Every IC and backtest in a study should price off the same execution
    model; caching by delay guarantees that and avoids re-querying the 1m
    table. Panels are aligned to ``close_wide.index`` with the signal close as
    the final fallback (see ``build_execution_close_panel``).
    """

    def __init__(self, engine, close_wide):
        self.engine = engine
        self.close_wide = close_wide
        self._panels = {}
        self._diagnostics = {}

    def _load(self, delay_minutes):
        delay = int(delay_minutes)
        if delay not in self._panels:
            closes_long = fetch_minute_exec_closes(
                self.engine,
                start_ts=self.close_wide.index.min(),
                end_ts=self.close_wide.index.max(),
                delay_minutes=delay,
                symbols=list(self.close_wide.columns),
            )
            panel, diagnostics = build_execution_close_panel(
                closes_long, self.close_wide.index, signal_close_wide=self.close_wide
            )
            self._panels[delay] = panel
            self._diagnostics[delay] = diagnostics
        return delay

    def close_panel(self, delay_minutes):
        """Execution close panel at ``delay_minutes`` after the signal close."""
        return self._panels[self._load(delay_minutes)]

    def diagnostics(self, delay_minutes):
        """Fill diagnostics from ``build_execution_close_panel`` for the delay."""
        return self._diagnostics[self._load(delay_minutes)]

    def forward_returns(self, delay_minutes, holding_period_bars, log_returns=True):
        """H-bar execution-priced forward returns, restricted to real-print fills."""
        delay = self._load(delay_minutes)
        return compute_execution_forward_returns(
            self._panels[delay],
            holding_period_bars=holding_period_bars,
            log_returns=log_returns,
            direct_fill_mask=self._diagnostics[delay]["direct_fill_mask"],
        )


def run_delay_sweep(
    signal_wide,
    exec_prices,
    delays_minutes,
    holding_period_bars,
    log_returns=True,
    **backtest_kwargs,
):
    """Backtest one signal at each execution delay (minutes).

    The weight schedule is identical across delays — only the fill prices
    move — so turnover and cost are ~constant and gross returns isolate alpha
    decay with latency. Returns ``{delay: run_light_backtest result}``.
    """
    results = {}
    for delay in delays_minutes:
        fwd = exec_prices.forward_returns(
            delay, holding_period_bars=holding_period_bars, log_returns=log_returns
        )
        results[int(delay)] = run_light_backtest(
            signal_wide,
            fwd,
            execution_delay_bars=0,
            returns_are_log=log_returns,
            holding_period_bars=holding_period_bars,
            **backtest_kwargs,
        )
    return results


def _zero_delay_mismatch(direct0_wide, close_wide, volume_wide):
    """Compare a δ=0 execution-close panel against the stored hourly closes.

    Both tables label bars at bar open, so with zero delay the last 1m close in
    ``[T, T+1h)`` must equal the hourly close at ``T`` on every bar the symbol
    actually traded. Comparison is restricted to hourly bars with
    ``volume > 0`` (a forward-filled hourly bar — a gap in the hourly export
    the 1m data happens to cover — is not a real print and is reported
    separately, not flagged as drift). Pure/DB-free; returns a summary dict.
    """
    hourly_real = (
        volume_wide.reindex(index=close_wide.index, columns=close_wide.columns) > 0
    )
    direct0 = direct0_wide.reindex(index=close_wide.index, columns=close_wide.columns)
    cmp_mask = direct0.notna() & hourly_real
    rel = ((direct0 - close_wide).abs() / close_wide.abs()).where(cmp_mask)
    n_cells = int(cmp_mask.sum().sum())
    return {
        "n_cells": n_cells,
        "n_synth_with_1m": int((direct0.notna() & ~hourly_real).sum().sum()),
        "max_rel_diff": float(rel.max().max()) if n_cells else 0.0,
    }


def assert_zero_delay_consistency(engine, close_wide, volume_wide, symbols=None, rtol=1e-9):
    """Data-integrity gate: assert the δ=0 execution close reproduces the
    stored hourly close on every real-trade bar.

    This is the timestamp-convention check — if bar labels drifted (an
    off-by-one-hour or off-by-one-minute bug in the bucketing), δ=0 execution
    would no longer equal the bar's own close and this would fire. Returns the
    summary dict from ``_zero_delay_mismatch``; raises ``AssertionError`` on
    drift.
    """
    direct0 = (
        fetch_minute_exec_closes(
            engine,
            start_ts=close_wide.index.min(),
            end_ts=close_wide.index.max(),
            delay_minutes=0,
            symbols=symbols if symbols is not None else list(close_wide.columns),
        )
        .pivot(index="bucket_ts", columns="symbol", values="exec_close")
    )
    summary = _zero_delay_mismatch(direct0, close_wide, volume_wide)
    assert summary["max_rel_diff"] < rtol, (
        f"δ=0 exec close != hourly close (max rel diff {summary['max_rel_diff']:.3e})"
    )
    return summary
