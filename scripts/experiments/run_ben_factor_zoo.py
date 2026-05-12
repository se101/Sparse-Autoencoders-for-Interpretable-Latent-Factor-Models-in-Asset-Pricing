from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from experiments.config import BEN_FACTOR_ZOO, to_jsonable  # noqa: E402
from experiments.factor_zoo import (  # noqa: E402
    pca_latent_factors,
    recursive_latent_factors,
    sparse_autoencoder_latent_factors,
    unexplained_alpha_fraction,
    variance_normalize,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run Ben factor-zoo analogue experiments.")
    parser.add_argument(
        "--include-autoencoders",
        action="store_true",
        help="Also run the sparse autoencoder grid over hidden layers, activations, and L1 penalties.",
    )
    args = parser.parse_args(argv)

    missing = [
        path
        for path in (
            BEN_FACTOR_ZOO.factors_path,
            BEN_FACTOR_ZOO.test_assets_path,
            BEN_FACTOR_ZOO.ff5_path,
        )
        if not path.exists()
    ]
    if missing:
        print("Ben factor-zoo data are not ready.")
        for path in missing:
            print(f"- Missing {path}")
        print("\nRun these first:")
        print("1. python3 scripts/build_analysis_dataset.py")
        print("2. python3 scripts/pull_global_q_testing_portfolios.py")
        return 2

    BEN_FACTOR_ZOO.output_dir.mkdir(parents=True, exist_ok=True)
    factors = _read_monthly(BEN_FACTOR_ZOO.factors_path).set_index("date")
    test_assets = _read_monthly(BEN_FACTOR_ZOO.test_assets_path).set_index("date")
    ff5 = _read_monthly(BEN_FACTOR_ZOO.ff5_path).set_index("date")

    factors = _restrict(factors, BEN_FACTOR_ZOO.sample.start, BEN_FACTOR_ZOO.sample.end)
    test_assets = _restrict(test_assets, BEN_FACTOR_ZOO.sample.start, BEN_FACTOR_ZOO.sample.end)
    ff5 = _restrict(ff5, BEN_FACTOR_ZOO.sample.start, BEN_FACTOR_ZOO.sample.end)

    normalized = variance_normalize(factors)
    market = ff5["Mkt-RF"] if "Mkt-RF" in ff5.columns else None

    latent_models: dict[str, pd.DataFrame] = {}
    model_metadata: dict[str, dict[str, object]] = {}
    diagnostics: dict[str, dict[str, float]] = {}

    for n_factors in BEN_FACTOR_ZOO.latent_factor_grid:
        name = f"linear_pca_proxy_k{n_factors}"
        latent_models[name] = pca_latent_factors(normalized, n_factors, f"linear_k{n_factors}")
        model_metadata[name] = {"family": "linear_pca_proxy", "latent_factors": n_factors}

        name = f"recursive_linear_proxy_k{n_factors}"
        latent_models[name] = recursive_latent_factors(
            normalized,
            n_factors,
            f"recursive_linear_k{n_factors}",
        )
        model_metadata[name] = {"family": "recursive_linear_proxy", "latent_factors": n_factors}

    if args.include_autoencoders:
        for hidden_layers, activation, l1_penalty in itertools.product(
            BEN_FACTOR_ZOO.hidden_layer_grid,
            BEN_FACTOR_ZOO.activation_grid,
            BEN_FACTOR_ZOO.l1_grid,
        ):
            hidden_label = "-".join(str(hidden) for hidden in hidden_layers)
            l1_label = f"{l1_penalty:g}".replace("-", "m").replace(".", "p")
            name = f"sparse_ae_k{BEN_FACTOR_ZOO.latent_factors}_h{hidden_label}_{activation}_l1{l1_label}"
            latent, diagnostic = sparse_autoencoder_latent_factors(
                normalized,
                BEN_FACTOR_ZOO.latent_factors,
                hidden_layers,
                activation,
                l1_penalty,
                BEN_FACTOR_ZOO.autoencoder_epochs,
                BEN_FACTOR_ZOO.autoencoder_learning_rate,
                BEN_FACTOR_ZOO.autoencoder_seed,
                name,
            )
            latent_models[name] = latent
            model_metadata[name] = {
                "family": "sparse_autoencoder",
                "latent_factors": BEN_FACTOR_ZOO.latent_factors,
                "hidden_layers": list(hidden_layers),
                "activation": activation,
                "l1_penalty": l1_penalty,
            }
            diagnostics[name] = diagnostic.__dict__

    summaries = [
        {
            **model_metadata[name],
            **unexplained_alpha_fraction(test_assets, latent, name, market=market).__dict__,
            **diagnostics.get(name, {}),
        }
        for name, latent in latent_models.items()
    ]

    pd.DataFrame(summaries).to_csv(BEN_FACTOR_ZOO.output_dir / "table1_explained_alpha_fraction.csv", index=False)
    for name, latent in latent_models.items():
        latent.to_csv(BEN_FACTOR_ZOO.output_dir / f"{name}_latent_factors.csv")

    manifest = {
        "config": to_jsonable(BEN_FACTOR_ZOO),
        "note": "Uses Global-q monthly 1-way low/high anomaly portfolios as HXZ-style test assets.",
        "input_shape": list(factors.shape),
        "test_asset_shape": list(test_assets.shape),
        "ff5_shape": list(ff5.shape),
        "ran_autoencoders": args.include_autoencoders,
    }
    (BEN_FACTOR_ZOO.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote Ben factor-zoo results to {BEN_FACTOR_ZOO.output_dir}")
    return 0


def _read_monthly(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").drop_duplicates("date")


def _restrict(df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    return df.loc[(df.index >= pd.Timestamp(start)) & (df.index <= pd.Timestamp(end))]


if __name__ == "__main__":
    raise SystemExit(main())
