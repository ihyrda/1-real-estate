"""Feature preparation for training and for single-property prediction.

final.csv is already numerically prepared, so this module does no scaling or
encoding. Its job is to keep the feature set in one frozen order and to derive
the four fields a user must not enter independently. Each derived field is
documented on its own function below.

Two callers: split_features_target serves training, from modeling.py. The rest
serve prediction, from the app's property form.
"""

import logging

import pandas as pd

from src.data import FEATURES, TARGET

logger = logging.getLogger(__name__)


def split_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Separate the frozen predictor set from price."""
    X = df[FEATURES].copy()
    y = df[TARGET].copy()
    logger.info("Prepared training data: X=%s, y=%s", X.shape, y.shape)
    return X, y


def derive_property_age(year_sold: int, year_built: int) -> int:
    """Calculate property age and reject impossible years.

    Derived rather than collected: a user-entered age inconsistent with the two
    years would build a row the dataset could never contain.
    """
    age = int(year_sold) - int(year_built)
    if age < 0:
        raise ValueError(
            f"year_built ({year_built}) cannot be later than year_sold ({year_sold})."
        )
    return age


def derive_recession(year_sold: int) -> int:
    """Create the dataset-defined 2010-2013 downturn indicator
    for the predictor form.

    """
    return int(2010 <= int(year_sold) <= 2013)


def derive_basement(property_type: str) -> int:
    """Basement is 1:1 with property type: every non-condo has one, no condo
    does. Derived for the predictor form."""
    return int(str(property_type).strip().lower() != "condo")


def derive_popular(beds: int, baths: int) -> int:
    """The dataset's popular flag marks the two-bed, two-bath configuration.
    Derived for the predictor form."""
    return int(int(beds) == 2 and int(baths) == 2)


def prepare_property_input(values: dict) -> pd.DataFrame:
    """Create one property row in the exact training feature order.

    Values arrive from the app's form, where each widget already constrains its
    own field to the dataset's range. The one rule no single widget can express
    is the relationship between the two years, and derive_property_age enforces
    it.
    """
    row = dict(values)

    property_type = row.pop("property_type", "Single-family")
    row["property_type_Condo"] = int(str(property_type).strip().lower() == "condo")
    row["property_age"] = derive_property_age(row["year_sold"], row["year_built"])
    row["recession"] = derive_recession(row["year_sold"])
    row["basement"] = derive_basement(property_type)
    row["popular"] = derive_popular(row["beds"], row["baths"])

    property_df = pd.DataFrame({c: [row[c]] for c in FEATURES}, columns=FEATURES)
    logger.info("Prepared one property row with %d features", property_df.shape[1])
    return property_df
