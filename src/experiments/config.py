from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"


@dataclass(frozen=True)
class DateWindow:
    start: str
    end: str


@dataclass(frozen=True)
class GKXExactConfig:
    """Exact target design from Gu, Kelly, and Xiu."""

    returns_path: Path = DATA_DIR / "gkx" / "monthly_stock_returns.csv"
    characteristics_path: Path = DATA_DIR / "gkx" / "monthly_characteristics.csv"
    output_dir: Path = RESULTS_DIR / "gkx_exact"
    train: DateWindow = DateWindow("1957-03-31", "1974-12-31")
    validation: DateWindow = DateWindow("1975-01-31", "1986-12-31")
    test: DateWindow = DateWindow("1987-01-31", "2016-12-31")
    factor_counts: tuple[int, ...] = (1, 2, 3, 4, 5, 6)
    ca_architectures: dict[str, tuple[int, ...]] | None = None
    ensemble_seeds: tuple[int, ...] = tuple(range(10))
    validation_l1_grid: tuple[float, ...] = (0.0, 1e-6, 1e-5, 1e-4, 1e-3)
    refit_frequency: str = "annual_expanding_train_rolling_12y_validation"
    required_characteristics: int = 94

    def __post_init__(self) -> None:
        if self.ca_architectures is None:
            object.__setattr__(
                self,
                "ca_architectures",
                {
                    "CA0": (),
                    "CA1": (32,),
                    "CA2": (32, 16),
                    "CA3": (32, 16, 8),
                },
            )


@dataclass(frozen=True)
class BenFactorZooConfig:
    """Project-data analogue of Ben Chaouch et al.'s factor-zoo experiments."""

    factors_path: Path = DATA_DIR / "factors.csv"
    test_assets_path: Path = DATA_DIR / "test_portfolios" / "global_q" / "global_q_1way_monthly_low_high.csv"
    output_dir: Path = RESULTS_DIR / "ben_factor_zoo"
    sample: DateWindow = DateWindow("1976-07-31", "2017-12-31")
    latent_factors: int = 6
    latent_factor_grid: tuple[int, ...] = tuple(range(1, 26))
    hidden_layer_grid: tuple[tuple[int, ...], ...] = ((16,), (32,), (64,), (128,), (64, 32))
    activation_grid: tuple[str, ...] = ("relu", "tanh")
    l1_grid: tuple[float, ...] = (0.0, 1e-5, 1e-4, 1e-3)
    #: One hidden layer widths for simple (reconstruction-only) AE test-portfolio runs.
    simple_ae_single_hidden_sizes: tuple[int, ...] = (64,)
    simple_ae_activation_grid: tuple[str, ...] = ("relu", "tanh", "gelu", "elu")
    autoencoder_epochs: int = 250
    autoencoder_learning_rate: float = 1e-3
    autoencoder_seed: int = 2026
    cluster_count: int = 60
    mispricing_gamma: float = 10.0
    models: tuple[str, ...] = (
        "linear_ae",
        "recursive_linear_ae",
        "clustered_linear_ae",
        "recursive_clustered_linear_ae",
        "tanh_ae",
        "recursive_tanh_ae",
        "clustered_tanh_ae",
    )


GKX_EXACT = GKXExactConfig()
BEN_FACTOR_ZOO = BenFactorZooConfig()


def to_jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "__dataclass_fields__"):
        return {key: to_jsonable(item) for key, item in asdict(value).items()}
    if isinstance(value, dict):
        return {key: to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    return value

