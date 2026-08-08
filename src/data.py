"""Dataset loading, validation, and app summary metrics.

load_data - file-level failure
validate_data - frame identity
get_data_summary - metrics for the app's overview tab
"""

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "final.csv"

TARGET = "price"

FEATURES = [
    "year_sold",
    "property_tax",
    "insurance",
    "beds",
    "baths",
    "sqft",
    "year_built",
    "lot_size",
    "basement",
    "popular",
    "recession",
    "property_age",
    "property_type_Condo",
]

REQUIRED_COLUMNS = [TARGET] + FEATURES


def load_data(path=None) -> pd.DataFrame:
    """Read the dataset off disk. File-level failure only.

    The path argument lets tests point at a missing file.
    """
    path = Path(path) if path is not None else DATA_PATH
    logger.info("Loading dataset from %s", path)

    if not path.exists():
        raise FileNotFoundError(f"Real-estate dataset not found at {path}")

    try:
        df = pd.read_csv(path)
    except pd.errors.EmptyDataError as exc:
        # Zero-byte file raised, header-only file raised downstream (validate_data).
        raise ValueError(f"Dataset at {path} is empty or unreadable.") from exc
    except pd.errors.ParserError as exc:
        raise ValueError(f"Dataset at {path} is malformed: {exc}") from exc

    logger.info("Loaded %d rows and %d columns", df.shape[0], df.shape[1])
    return df


def validate_data(df: pd.DataFrame) -> None:
    """Confirm the loaded frame is the intended dataset.

    Checks identity:
    swapped or truncated file is rejected here
    values taken on trust

    """
    # A header-only catch
    if df.empty:
        raise ValueError("Dataset is empty.")

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Shape and column names can match while the contents are wrong type.
    non_numeric = [
        c for c in REQUIRED_COLUMNS if not pd.api.types.is_numeric_dtype(df[c])
    ]
    if non_numeric:
        raise ValueError(f"Expected numeric columns are non-numeric: {non_numeric}")

    logger.info(
        "Validation passed: %d rows, %d columns, %d missing values",
        df.shape[0],
        df.shape[1],
        int(df.isna().sum().sum()),
    )


def get_data_summary(df: pd.DataFrame) -> dict:
    """Return the metrics the app's overview tab displays."""
    return {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "missing_values": int(df.isna().sum().sum()),
        "price_min": float(df[TARGET].min()),
        "price_max": float(df[TARGET].max()),
        "price_mean": float(df[TARGET].mean()),
        "price_median": float(df[TARGET].median()),
    }
