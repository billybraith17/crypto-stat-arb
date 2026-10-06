# Cross-Sectional Statistical Arbitrage — Crypto

[![CI](https://github.com/billybraith17/crypto-stat-arb/actions/workflows/ci.yml/badge.svg)](https://github.com/billybraith17/crypto-stat-arb/actions/workflows/ci.yml)

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
diagnostics → backtest → execution/events → robustness and walk-forward → findings.

---

## Key findings

Spot Kraken USD pairs, 2020-10 to 2024-06, top-20 monthly liquidity universe. Costs are a 10 bps
one-way taker fee (an institutional volume tier: the cost base of a systematic fund) plus
half-spread on every fill, with 5–40 bps fee tiers as a sensitivity.

**Momentum (24h bars): no robust signal.**
- The best-looking configuration is an artefact of searching ~275 variants. Corrected for that
  search it is no better than the best of as many random signals (deflated Sharpe ratio 0.28;
  max-over-trials p = 0.09), and re-selecting it using only past data earns nothing out of sample
  (walk-forward IC 0.003, t = 0.12).
- The dollar-neutral book carried hidden market exposure: its beta flipped sign with the
  market's direction (β ≈ −0.38 in rising markets, +0.11 in falling ones), which a full-sample
  correlation of 0.05 averages away. Momentum computed on market-neutralised returns has zero
  predictive power.

**Mean reversion (1h bars): real, but not tradable on spot even at institutional costs.**
- Coins that move furthest in an hour revert relative to their peers. The effect survives
  realistic fill timing, multiple-testing correction and out-of-sample re-selection
  (out-of-sample IC 0.071, t = 21), is market-neutral, and is not a stale-price artifact.
- The gross profit per holding period (≈ 7 bps) is smaller than the bid-ask bounce a taker pays to
  capture it: break-even requires the total cost per fill (fee plus half-spread) to stay below ≈ 2.4 bps,
  less than the tightest estimated half-spread on its own.
- Trading more slowly does not help. A pre-specified low-turnover version cuts turnover ~9× but
  loses money before costs: the mispricings resolve before a cheap-to-trade book can hold them.

**Next step.** Perpetual futures offer tighter spreads, cheap two-sided shorting and ~10× the
breadth, which address the binding constraints here: spread cost and too few names.

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
7. Test robustness across lookback, holding period, and universe size.
8. Stress realistic transaction costs — including per-asset effective spreads estimated from
   OHLC ranges — and execution delays.
9. Judge best-of-grid results as **max-statistics**: max-over-trials p-values and deflated
   Sharpe ratios, so a winner picked from N trials is never quoted at single-test significance.
10. Check signal stability with **walk-forward evaluation** over embargoed folds, with a later
    block of data kept fully held out for a future out-of-sample test.

### Design choices

| Choice | Rationale |
|---|---|
| Top-20 universe by trailing 30-day dollar volume, rebuilt monthly; 90-day minimum listing age | Point-in-time and tradable: names enter and leave as they would have in real time, including later-delisted ones (LUNA, UST) |
| 1h bars for reversal, 24h bars for momentum | Each signal is measured at the horizon its hypothesis lives at: intraday overreaction vs multi-week trend |
| Cross-sectional rank transform; equal-weight top/bottom 20% book | Ranks are robust to crypto's fat tails; a dollar-neutral quantile book isolates relative performance from the market's direction |
| Spearman IC with Newey-West lag `max(H, lookback) − 1` | Overlapping holding and lookback windows make consecutive ICs correlated; the lag covers the longer overlap |
| Fills at real 1-minute prints one minute after the signal close; carried-forward prices excluded | Removes the look-ahead of trading at the price that generated the signal, and never prices a fill at a print that did not happen |
| One-way fee plus half-spread on every fill; per-asset spreads from hourly high-low ranges | Close prices proxy the mid; daily ranges are dominated by volatility in a 24/7 market, hourly ranges are not |
| Max-over-trials p-values, deflated Sharpe, embargoed walk-forward re-selection | A best-of-grid result is a maximum over many trials and must be judged as one |
| Pre-specified designs (residual momentum, low-turnover construction) | Fixing the design before seeing results makes each one a single trial, not a search |
| Data after 2024-06 untouched | Reserved for a single out-of-sample test of whatever survives |

### Limitations

- Light backtest: no market impact, and no weight drift within a holding period.
- Shorts carry no borrow cost; spot shorting requires margin. Negligible at the reversal sleeve's
  2-hour hold, material at momentum's 14-day hold.
- Corwin–Schultz spreads are estimates; they overstate costs on the most liquid pairs.
- One venue and ~14 names on average limit statistical power, particularly for momentum.

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
on traded-masked returns with shrinkage toward 1: against an equal-weight universe index for the
residual-momentum feature, and against XBT/USD (the tradable hedge instrument) for the hedge
overlay and exposure diagnostics.

### Mean-reversion sleeve (`src/signals/cs_mean_reversion.py`)

Whether short-term price dislocations reverse. Features include short-term return reversal, price
z-score / Bollinger band-touch, volatility-adjusted moves, extreme-move normalisation, volume
exhaustion, VWAP/MA distance, an RSI proxy, and range position — all sign-normalised so that
higher always means "buy". A pre-specified **equal-weight composite** of z-scored reversal
features (one trial, not an argmax over the family) feeds a **low-turnover construction**:
hysteresis no-trade bands hold quantile membership until a name exits a buffer zone, cutting
churn where the fast book's edge was consumed by costs.

### Evaluation framework (`src/research/momentum_eval.py`, `execution.py`, `spreads.py`)

Spearman/rank IC, Newey-West adjusted t-statistics, IC-decay heatmaps, quantile portfolio
analysis, a cost-aware long/short backtest (one-way taker fees plus flat or per-asset
half-spreads, charged on every fill), execution-delay
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

A set of measurement-integrity guards keeps headline numbers reliable: execution fills that are
not real prints (LOCF-carried) are excluded from backtest returns, forward-filled no-trade bars
are masked out of the IC analysis to bound the stale-print artifact, and per-asset effective
spreads estimated from OHLC ranges (Corwin–Schultz, `src/research/spreads.py`) bound how much of
a short-horizon reversal edge is just bid-ask bounce a taker could never capture. A zero-delay
consistency gate checks that minute-level fills reproduce the stored hourly close on every
real-trade bar, so a bar-labelling error cannot silently shift execution timing.

---

## Future work

These are research directions under consideration, not yet built:

- Moving from spot to perpetual futures to exploit lower fees and tighter spreads — directly
  relevant to the mean-reversion result, whose edge is currently consumed by spot transaction
  costs — with funding rates as an additional signal and cost dimension.
- Signal combination across the momentum and mean-reversion sleeves.
- Portfolio construction with turnover-aware optimisation and attribution.
- An order-flow / market-microstructure sleeve (trade and order-book imbalance, liquidity pressure).

---

## Running it yourself

```bash
uv sync              # create .venv from uv.lock (requires uv)

# PostgreSQL credentials are read from .env (see "Configuration" below)
make build-db        # build the DB from raw CSVs
make pipeline        # full pipeline: build → quality checks → warnings → run logging
make quality         # quality checks against an existing DB
uv run pytest        # run the test suite
uv run ruff check .  # lint
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
└── research/   # evaluation helpers (IC, quantiles, backtest, execution fills, spreads, robustness grids)
tests/          # unit tests + data-quality integration checks
run_pipeline.py # pipeline entry point
```

---

## Technologies

Python · pandas · NumPy · SciPy · PostgreSQL (SQLAlchemy / psycopg2) · Matplotlib · Jupyter · pytest · uv · ruff · GitHub Actions

---

## Disclaimer

This is an educational research project intended to demonstrate quantitative research
methodology. It is not investment advice, and no claim is made that any signal presented here is
profitable or suitable for live trading.
