## Sparse Autoencoders for Interpretable Latent Factor Models in Asset Pricing

Data pipeline scaffold for sparse-autoencoder experiments on interpretable
asset-pricing factor returns.

### Data source

We are using two data sources.

Open Source Asset Pricing / Open Asset Pricing:

- Website: https://www.openassetpricing.com/data/
- Release used by the downloader: October 2025, version 2.0.0
- Source file: `PredictorPortsFull.csv`
- Coverage in the pulled file: January 1926 through December 2024

The project script downloads the original-paper portfolio return file from
Open Asset Pricing, saves a raw copy, and derives two model-ready panels:

- `data/factors.csv`: long-short predictor portfolio returns, one column per
  signal.
- `data/openassetpricing_sorted_portfolio_returns.csv`: non-long-short sorted
  portfolio returns, named as `<signal>_p<portfolio>`.
- `data/openassetpricing_signal_metadata.csv`: signal definitions, categories,
  signs, original paper metadata, and replication summary fields from
  `SignalDoc.csv`.

Open Asset Pricing includes missing values for some signals and portfolio
sorts. This is expected because not every predictor exists for the full sample
or has well-defined sorted portfolios in every month.

Kenneth French's Dartmouth Data Library:

- Website: https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html
- Source files:
  - `F-F_Research_Data_Factors_CSV.zip`
  - `F-F_Research_Data_5_Factors_2x3_CSV.zip`
- `data/fama_french_3_factors.csv`: canonical monthly FF3 factors:
  `Mkt-RF`, `SMB`, `HML`, `RF`.
- `data/fama_french_5_factors.csv`: canonical monthly FF5 factors:
  `Mkt-RF`, `SMB`, `HML`, `RMW`, `CMA`, `RF`.

The Ken French returns are kept in percent units, matching the Open Asset
Pricing files.

### Expected layout

After running the downloader, the data directory should look like this:

```text
data/
  factors.csv
  openassetpricing_sorted_portfolio_returns.csv
  openassetpricing_signal_metadata.csv
  fama_french_3_factors.csv
  fama_french_5_factors.csv
  raw/
    openassetpricing_predictor_ports_full.csv
    ken_french_ff3_csv.zip
    ken_french_ff5_csv.zip
```

### Expected columns

`factors.csv`
- `date`
- one column per long-short factor return

`openassetpricing_sorted_portfolio_returns.csv`
- `date`
- one column per sorted benchmark portfolio return

`openassetpricing_signal_metadata.csv`
- `Acronym`
- descriptive/category fields such as `LongDescription`, `Cat.Signal`,
  `Cat.Data`, `Cat.Economic`, `Detailed Definition`
- original-paper fields such as `Authors`, `Year`, `Journal`
- construction and evidence fields such as `Sign`, `Return`, `T-Stat`,
  `LS Quantile`, `Portfolio Period`

Dates can be monthly strings or timestamps. The loader will try to parse them.

### Refresh Open Asset Pricing data

Install the downloader dependency if needed:

```bash
python3 -m pip install openassetpricing
```

Then run:

```bash
python3 scripts/pull_openassetpricing.py
```

If `data/raw/openassetpricing_predictor_ports_full.csv` already exists, rebuild
the model-ready panels without another download:

```bash
python3 scripts/pull_openassetpricing.py --from-raw
```

The latest pull produced:

- raw portfolio data shape: `(1226794, 7)`
- factor panel shape: `(1188, 213)`
- benchmark panel shape: `(1188, 1278)`
- signal metadata shape: `(331, 29)`
- date range: `1926-01-30` to `2024-12-31`

### Refresh Fama-French data

Run:

```bash
python3 scripts/pull_fama_french.py
```

The latest pull produced:

- FF3 shape: `(1197, 5)`
- FF3 date range: `1926-07-31` to `2026-03-31`
- FF5 shape: `(753, 7)`
- FF5 date range: `1963-07-31` to `2026-03-31`

### Build analysis dataset

Run:

```bash
python3 scripts/build_analysis_dataset.py
```

This aligns Open Asset Pricing, FF3, and FF5 to the common monthly FF5/OpenAP
sample:

- sample start: `1963-07-31`
- sample end: `2024-12-31`
- months: `738`

The script writes cleaned files under `data/analysis/`:

- `openap_factors_balanced.csv`: 129 OpenAP long-short factors with no missing
  values.
- `openap_sorted_portfolios_balanced.csv`: 756 sorted portfolio returns with no
  missing values.
- `openap_factors_80pct_available.csv`: 163 factors with at least 80% monthly
  availability.
- `openap_sorted_portfolios_80pct_available.csv`: 1008 sorted portfolios with
  at least 80% monthly availability.
- `fama_french_3_factors_aligned.csv`: FF3 restricted to the common sample.
- `fama_french_5_factors_aligned.csv`: FF5 restricted to the common sample.
- `missingness_report.csv`: missing-value share by source column.
- `data_prep_summary.json`: machine-readable summary of the data-prep run.

Use the balanced files first for baseline regressions, PCA, and the first sparse
autoencoder pass. Use the 80%-available files later if we want a larger universe
with explicit imputation or missing-value masking.

### Paper-specific experiment targets

The experiment scaffold is organized around two paper targets.

Gu, Kelly, and Xiu, `ssrn-3335536.pdf`, is treated as an exact replication
target. The expected input files are:

```text
data/gkx/
  monthly_stock_returns.csv       # permno, date, ret_excess
  monthly_characteristics.csv     # permno, date, 94 lagged characteristics
```

The exact GKX config in `configs/gkx_exact.json` uses the paper's sample split:
1957-03 through 1974-12 for training, 1975-01 through 1986-12 for validation,
and 1987-01 through 2016-12 for testing. It targets FF/PCA/IPCA-style
benchmarks and `CA0` through `CA3` conditional autoencoders with `K = 1..6`.

Validate the exact GKX inputs and build characteristic-managed portfolios with:

```bash
python3 scripts/experiments/run_gkx_exact.py
```

If you have WRDS access, build the CRSP monthly stock return side of the GKX
panel with:

```bash
python3 scripts/pull_gkx_crsp_monthly.py --username YOUR_WRDS_USERNAME
```

The script writes `data/gkx/monthly_stock_returns.csv` with CRSP monthly returns,
delisting-adjusted total returns, local Fama-French `RF`, excess returns, market
equity, exchange code, share code, SIC, and delisting code. The characteristic
side, `data/gkx/monthly_characteristics.csv`, is a separate Compustat/CCM
milestone.

We use the Feng-CityUHK EquityCharacteristics toolkit for the firm-level
characteristic side. Fetch the toolkit with:

```bash
python3 scripts/fetch_equity_characteristics.py
```

Then run its documented WRDS workflow from the checkout:

```bash
python3 scripts/run_equity_characteristics_workflow.py \
  --toolkit-dir external/EquityCharacteristics \
  --workflow-subdir char60 \
  --python .venv/bin/python
```

If you also want the top-level single-characteristic scripts, add
`--include-single-characteristics`. Once the toolkit has produced a rank-imputed
file such as `chars_rank_imputed.feather`, convert it to this project's GKX
schema with:

```bash
python3 scripts/adapt_equity_characteristics.py \
  external/EquityCharacteristics/char60/chars_rank_imputed.feather
```

The adapter writes `data/gkx/monthly_characteristics.csv`, preferring `rank_*`
columns when present and aligning to the `permno,date` keys in
`data/gkx/monthly_stock_returns.csv`.

Ben Chaouch, Lo, Singh, and Xiong, `PDF_Ben.pdf`, is implemented as a
methodology replication using balanced OpenAP long-short factors as the factor
zoo and Global-q monthly 1-way anomaly portfolios as HXZ-style test assets. Its
config lives in `configs/ben_factor_zoo.json`.

Download and concatenate the Global-q monthly 1-way sorts with:

```bash
python3 scripts/pull_global_q_testing_portfolios.py
```

This writes:

```text
data/global_q/global_q_1way_monthly_long.csv
data/global_q/global_q_1way_monthly_all_portfolios.csv
data/global_q/global_q_1way_monthly_low_high.csv
data/global_q/global_q_1way_monthly_metadata.csv
```

Run the first factor-zoo analogue with:

```bash
python3 scripts/experiments/run_ben_factor_zoo.py
```

By default this freezes the linear PCA and recursive PCA baselines over
`K = 1..6`. To run the first sparse-autoencoder grid over hidden dimensions,
activations, and L1 penalties:

```bash
python3 scripts/experiments/run_ben_factor_zoo.py --include-autoencoders
```

### First data check

Run:

```bash
python3 scripts/check_data.py
```

This prints:
- factor and benchmark shapes
- date ranges
- number of overlapping months
- missing-value counts

If your raw files use different names or a different format, update
`src/data_loading.py`.
