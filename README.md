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
