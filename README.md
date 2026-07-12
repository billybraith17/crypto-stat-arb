# Cross-Sectional Statistical Arbitrage — Crypto

A research codebase for developing and validating **cross-sectional alpha signals** in
cryptocurrency markets, using hourly and 1-minute OHLCV data for liquid Kraken USD pairs.

The objective is **not** a single profitable strategy. It is to replicate the research process
used by systematic trading firms: form a hypothesis, engineer features, turn them into
cross-sectional signals, and validate them with robust statistics *before* worrying about
optimisation — favouring interpretable, robust signals over impressive in-sample backtests.

The two research notebooks in [`research/`](research/) are the main deliverable. They are
committed **with their rendered outputs**, so they can be read end-to-end on GitHub without a
database:

- [`research/cs_momentum_research.ipynb`](research/cs_momentum_research.ipynb) — cross-sectional momentum
- [`research/cs_mean_reversion_research.ipynb`](research/cs_mean_reversion_research.ipynb) — cross-sectional mean reversion

Each runs top-to-bottom: setup → configuration → data → signal pipeline → IC analysis →
diagnostics → backtest → execution/events → robustness grid → interpretation.

---

## Motivation

Most public quant-trading projects predict the direction of a single asset or optimise a single
backtest. That differs substantially from how systematic firms actually research. This project was
designed to mirror that workflow more closely by focusing on:

- Cross-sectional rather than directional forecasting
- Many weak alpha signals rather than one predictive model
- Statistical validation *before* optimisation
- Realistic execution assumptions and transaction costs
- Modular, reproducible research infrastructure

Crypto was chosen because it offers continuous trading, a diverse cross-section of liquid
assets, and freely accessible historical data — making it feasible for an independent research
project.

---

## Research methodology

Each signal follows the same structured process:

1. Formulate an economic or behavioural hypothesis.
2. Engineer candidate features.
3. Convert features into cross-sectional alpha signals (rank / z-score normalised).
4. Evaluate predictive power via Information Coefficient (IC) analysis.
5. Assess significance with **Newey-West** standard errors that account for overlapping
   forward-return and signal-lookback windows.
6. Analyse quantile monotonicity and long/short portfolio spreads.
7. Test robustness across lookback, holding period, transform, and universe size.
8. Stress realistic transaction costs — including per-asset effective spreads estimated from
   OHLC ranges — and execution delays.
9. Judge best-of-grid results as **max-statistics**: max-over-trials p-values and deflated
   Sharpe ratios, so a winner picked from N trials is never quoted at single-test significance.
10. Check signal stability with **walk-forward evaluation** over embargoed folds, with a later
    block of data kept fully held out for a future out-of-sample test.

The emphasis throughout is on avoiding overfitting and preferring robust, interpretable signals.

---

## What's implemented

### Data pipeline

Raw Kraken CSVs → **PostgreSQL** (`ohlcv`, `ohlcv_1m`, `monthly_universe`), with hard
data-quality checks, soft warning diagnostics, and per-run telemetry. A monthly tradable
universe is rebuilt from trailing liquidity, with minimum listing-age and liquidity filters.
Alongside the dense hourly table, a **sparse 1-minute table** (real trade bars only, universe
symbols only) supports minute-level execution modelling, with a cross-timeframe consistency
check tying the two tables together.

### Momentum sleeve (`src/signals/cs_momentum.py`)

Whether relative strength across the universe predicts cross-sectional returns. Feature families:
multiple lookback horizons, peer-relative momentum, volatility-adjusted momentum, **residual
momentum** (momentum on beta-residualized returns, Blitz–Huij–Martens style — market neutrality
built into the alpha itself), and cross-sectional ranking. Per-asset rolling betas are estimated
against an equal-weight universe index on traded-masked returns, with shrinkage toward 1.

### Mean-reversion sleeve (`src/signals/cs_mean_reversion.py`)

Whether short-term price dislocations reverse. Features include short-term return reversal, price
z-score / Bollinger band-touch, volatility-adjusted moves, extreme-move normalisation, volume
exhaustion, VWAP/MA distance, an RSI proxy, and range position — all sign-normalised so that
higher always means "buy".

### Evaluation framework (`src/research/momentum_eval.py`, `execution.py`, `spreads.py`)

Spearman/rank IC, Newey-West adjusted t-statistics, IC-decay heatmaps, quantile portfolio
analysis, a cost-aware long/short backtest (flat or per-asset spread costs), execution-delay
sensitivity, walk-forward IC evaluation with embargoed folds, and selection-bias corrections
(max-over-trials p-values, deflated Sharpe). The same helpers are reused across both sleeves.

Market neutrality is measured and enforced, not assumed: ex-ante portfolio beta (Σ wᵢβᵢ) and
its coverage are tracked bar by bar, realized betas are split by up-/down-market regime (a
dollar-neutral book can hide large offsetting regime betas behind a full-sample correlation of
~0), and an optional **beta-hedge overlay** adds a benchmark position sized to cancel the
ex-ante beta at each rebalance — netted against any existing benchmark position, with hedge
turnover fully costed in the backtest.

Backtests fill at **real 1-minute closes** a configurable number of minutes after the signal
bar's close (`src/research/execution.py`), rather than assuming execution at the very close the
signal was computed on — the delay-sensitivity curve (0–60 minutes) quantifies how much edge
survives realistic latency, and how much of the naive result was look-ahead.

A set of measurement-integrity guards keeps headline numbers honest: execution fills that are
not real prints (LOCF-carried) are excluded from backtest returns, forward-filled no-trade bars
are masked out of the IC analysis to bound the stale-print artifact, and per-asset effective
spreads estimated from OHLC ranges (Corwin–Schultz, `src/research/spreads.py`) bound how much of
a short-horizon reversal edge is just bid-ask bounce a taker could never capture.

---

## Future work

These are research directions under consideration, not yet built:

- Refining the existing momentum and mean-reversion features (stronger construction, additional
  hypotheses — residual momentum is now built; composite/low-turnover variants remain).
- Moving from spot to perpetual futures to exploit lower fees and tighter spreads — directly
  relevant to the mean-reversion result, whose edge is currently consumed by spot transaction
  costs — with funding rates as an additional signal and cost dimension.
- Composite alpha construction and multi-signal combination.
- Portfolio construction with turnover-aware optimisation and attribution.
- An order-flow / market-microstructure sleeve (trade and order-book imbalance, liquidity pressure).

---

## Running it yourself

```bash
pip install -r requirements.txt      # uses .venv

# PostgreSQL credentials are read from .env (see "Configuration" below)
make build-db        # build the DB from raw CSVs
make pipeline        # full pipeline: build → quality checks → warnings → run logging
make quality         # quality checks against an existing DB
pytest               # run the test suite
```

A database is only needed to *re-run* the notebooks; the committed outputs can be read as-is.

### Configuration

Postgres credentials are read from a `.env` file (not committed): `PG_HOST`, `PG_PORT`, `PG_DB`,
`PG_USER`, `PG_PASSWORD` (defaults to `localhost:5432/crypto`). See `.env.example`. Research and
production knobs live in [`configs/`](configs/) (`base.yaml`,
`research/momentum_signal.yaml`, `research/mean_reversion_signal.yaml`). Raw hourly USD CSVs are
expected under `src/data/raw/` (configurable via `raw_dir`).

---

## Repository structure

```text
configs/        # YAML config: universe, liquidity, signal/research knobs
research/       # Jupyter research notebooks (the main showcase)
src/
├── common/     # config loading, DB engine, pipeline runner, run logging
├── data/       # CSV → PostgreSQL loader, quality checks, DB-to-research accessors
├── signals/    # cross-sectional momentum & mean-reversion signal construction
└── research/   # evaluation helpers (IC, quantiles, backtest, execution fills, spreads)
tests/          # unit tests + data-quality integration checks
run_pipeline.py # pipeline entry point
```

---

## Technologies

Python · pandas · NumPy · SciPy · PostgreSQL (SQLAlchemy / psycopg2) · Matplotlib · Jupyter · pytest

---

## Disclaimer

This is an educational research project intended to demonstrate quantitative research
methodology. It is not investment advice, and no claim is made that any signal presented here is
profitable or suitable for live trading.
