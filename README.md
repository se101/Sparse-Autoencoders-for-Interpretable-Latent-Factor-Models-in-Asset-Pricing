## Sparse Autoencoders for Interpretable Latent Factor Models in Asset Pricing

Data pipeline scaffold for sparse-autoencoder experiments on interpretable
asset-pricing factor returns.

### Data source

The factor zoo is **only** the **150 replicated factor returns** from Feng,
Giglio, and Xiu (2020), *Taming the Factor Zoo* (*Journal of Finance*), as
shipped in the journal’s replication archive (`data/factors.csv` in that package).

- Copy the replication file to **`data/factors.csv`** at the project root
  (see `.gitignore`: the file stays local).
- Columns: **`Date`**, **`RF`**, then **150** factor return columns. Loaders drop
  **`RF`** and scale decimals to **percentage points** so they align with common
  monthly return panels (including test portfolios pulled for this project).

**Test portfolios** (what we price / evaluate on—not the factor zoo) live under
**`data/test_portfolios/`**, for example Global-q monthly 1-way anomaly panels.
Pulled CSVs are gitignored; **`data/test_portfolios/.gitkeep`** keeps the
directory in version control.

Kenneth French’s Dartmouth Data Library (**FF3 factors for GKX / RF merge** only;
not used in the Ben factor-zoo latent pipeline):

- Website: https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html
- After `python3 scripts/pull_fama_french.py`:
  - `data/fama_french_3_factors.csv` (`Mkt-RF`, `SMB`, `HML`, `RF`, …)

Ken French monthly factor returns are stored in **percent** (e.g. `5.07` for
5.07% in a month).

### Expected layout

```text
data/
  factors.csv                          # FGX (2020) replication; not committed
  fama_french_3_factors.csv            # for GKX CRSP pull (RF); optional otherwise
  test_portfolios/
    global_q/
      global_q_1way_monthly_low_high.csv
    ff/
      25_Portfolios_5x5.csv
    ...
  raw/
    global_q/                           # optional cached zips from Global-q
  gkx/                                  # optional: stock-level GKX panel
```

### Refresh Fama-French data

```bash
python3 scripts/pull_fama_french.py
```

### Paper-specific experiment targets

The experiment scaffold is organized around two paper targets.

Gu, Kelly, and Xiu, `ssrn-3335536.pdf`, is treated as an exact replication
target. The expected input files are:

```text
data/gkx/
  monthly_stock_returns.csv       # permno, date, ret_excess
  monthly_characteristics.csv     # permno, date, 94 lagged characteristics
```

The GKX design parameters live in **`src/experiments/config.py`** (`GKX_EXACT`):
training 1957-03 through 1974-12, validation 1975-01 through 1986-12, testing
1987-01 through 2016-12, with `CA0`–`CA3` autoencoder widths and `K = 1..6`.

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
methodology replication using the **FGX 150 factors** as the zoo and **Global-q**
monthly 1-way anomaly portfolios as HXZ-style **test assets**. Config:
`configs/ben_factor_zoo.json` and `src/experiments/config.py` (`BEN_FACTOR_ZOO`).

Download and build Global-q panels into **`data/test_portfolios/global_q/`**:

```bash
python3 scripts/pull_global_q_testing_portfolios.py
```

This writes:

```text
data/test_portfolios/global_q/global_q_1way_monthly_long.csv
data/test_portfolios/global_q/global_q_1way_monthly_all_portfolios.csv
data/test_portfolios/global_q/global_q_1way_monthly_low_high.csv
data/test_portfolios/global_q/global_q_1way_monthly_metadata.csv
```

If you previously used `data/global_q/`, move those CSVs into
`data/test_portfolios/global_q/` (same filenames) or re-run the pull script.

Run the factor-zoo analogue with:

```bash
python3 scripts/experiments/run_ben_factor_zoo.py
```

By default this runs linear PCA and recursive PCA baselines over `K = 1..25`.
It also writes PCA variance diagnostics plus scree and reconstruction-MSE plots:

```text
results/ben_factor_zoo/pca_variance_diagnostics.csv
results/ben_factor_zoo/pca_scree_plot.png
results/ben_factor_zoo/pca_reconstruction_mse_plot.png
```

Sparse autoencoder grid:

```bash
python3 scripts/experiments/run_ben_factor_zoo.py --include-autoencoders
```

### First data check

```bash
python3 scripts/check_data.py
```

This prints factor and test-portfolio shapes, date ranges, overlapping months,
and missing-value counts. If your layouts differ, update `src/data_loading.py`.
