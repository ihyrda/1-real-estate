"""Test suite for the real-estate project.

Run from the project root, inside the activated virtual environment
(see README.md)

    python -m pytest -v

property_values is per-test: two tests mutate it.
"""

import pytest

from src.data import (
    FEATURES,
    REQUIRED_COLUMNS,
    TARGET,
    load_data,
    validate_data,
)
from src.modeling import (
    build_models,
    predict_property,
    split_data,
    train_and_evaluate_models,
)
from src.preprocessing import (
    derive_basement,
    derive_popular,
    derive_recession,
    prepare_property_input,
    split_features_target,
)


@pytest.fixture(scope="session")
def df():
    """The validated source dataset."""
    frame = load_data()
    validate_data(frame)
    return frame


@pytest.fixture(scope="session")
def results(df):
    """The full trained result bundle."""
    return train_and_evaluate_models(df)


@pytest.fixture
def property_values():
    """One complete set of user-supplied property inputs."""
    return {
        "year_sold": 2015,
        "property_tax": 300,
        "insurance": 100,
        "beds": 3,
        "baths": 2,
        "sqft": 1800,
        "year_built": 1990,
        "lot_size": 8000,
        "property_type": "Single-family",
    }


# --- data ------------------------------------------------------------------

# The file on disk is the one this project expects: shape and columns
def test_dataset_shape_and_columns(df):
    assert df.shape == (1860, 14)
    for column in REQUIRED_COLUMNS:
        assert column in df.columns


# load_data: absent file raises, no empty frame returned
def test_missing_file_raises_error(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_data(tmp_path / "does_not_exist.csv")


# validate_data: swapped/bad file rejected
def test_missing_column_raises_error(df):
    with pytest.raises(ValueError, match="price"):
        validate_data(df.drop(columns=["price"]))


# --- preparation -----------------------------------------------------------

# Price is the target, never a predictor
def test_target_is_separated(df):
    X, y = split_features_target(df)
    assert TARGET not in X.columns
    assert y.name == TARGET
    assert len(X) == len(y)


# Model fitted on this order, prediction row built to it
def test_feature_order_is_correct(df):
    X, _ = split_features_target(df)
    assert list(X.columns) == FEATURES


# property_age: year_sold minus year_built, never collected
def test_property_age_is_derived(property_values):
    row = prepare_property_input(property_values)
    assert row.loc[0, "property_age"] == 2015 - 1990


# The one rule no widget can enforce: two valid years in an impossible order
def test_impossible_years_are_rejected(property_values):
    property_values["year_built"] = 2020
    with pytest.raises(ValueError, match="year_built"):
        prepare_property_input(property_values)


# The 2010-2013 window, both boundaries and just outside each
def test_recession_is_derived_from_year():
    assert derive_recession(2009) == 0
    assert derive_recession(2010) == 1
    assert derive_recession(2013) == 1
    assert derive_recession(2014) == 0


# Form label becomes the model's binary condo column
def test_property_type_is_encoded(property_values):
    single_family = prepare_property_input(property_values)
    assert single_family.loc[0, "property_type_Condo"] == 0

    property_values["property_type"] = "Condo"
    condo = prepare_property_input(property_values)
    assert condo.loc[0, "property_type_Condo"] == 1


# Derived, not asked: basement is 1:1 with property type
def test_basement_follows_property_type():
    assert derive_basement("Condo") == 0
    assert derive_basement("Single-family") == 1


# Derived, not asked: popular marks the two-bed two-bath configuration
def test_popular_follows_bed_and_bath_count():
    assert derive_popular(2, 2) == 1
    assert derive_popular(3, 2) == 0
    assert derive_popular(2, 1) == 0


# Nine collected fields become the thirteen the model expects
def test_property_input_has_thirteen_features(property_values):
    row = prepare_property_input(property_values)
    assert row.shape == (1, 13)
    assert list(row.columns) == FEATURES


# --- modelling -------------------------------------------------------------

# Stratified split: ~20% condo on both sides
def test_split_is_stratified(df):
    X, y = split_features_target(df)
    X_train, X_test, _, _ = split_data(X, y)
    overall = X["property_type_Condo"].mean()
    assert X_train["property_type_Condo"].mean() == pytest.approx(overall, abs=0.02)
    assert X_test["property_type_Condo"].mean() == pytest.approx(overall, abs=0.02)


# Contract between modeling.py and app.py: app formats on literal names, so a
# rename in one drops formatting in the other [passing names would fix this]
def test_metrics_passing_fidelity(results):
    metrics = results["metrics"]
    assert set(results["models"]) == {"Linear Regression", "Random Forest"}
    assert set(metrics["Model"]) == {"Linear Regression", "Random Forest"}
    for column in ("Train MAE", "Test MAE", "MAE Gap", "Train R2", "Test R2"):
        assert column in metrics.columns
    # Columns existing is not enough: MAE is dollars, R2 a proportion, and the
    # magnitudes cannot overlap. A transposed assignment fails here rather than
    # rendering as $1 on the page.
    assert metrics["Test MAE"].min() > 1000
    assert 0 <= metrics["Test R2"].max() <= 1


# Selection criterion: lowest test MAE wins
def test_selection_criteria_fidelity(results):
    metrics = results["metrics"]
    best = metrics.loc[metrics["Test MAE"].idxmin(), "Model"]
    assert results["selected_model_name"] == best
    assert results["selected_model"] is results["models"][best]


# The app's full path: nine inputs, one price. A row disagreeing with the
# fitted columns raises in sklearn [parts tested above, chained here]
def test_prediction_path_runs_end_to_end(results, property_values):
    row = prepare_property_input(property_values)
    price = predict_property(results["selected_model"], row)
    assert isinstance(price, float)
    assert price > 0


# Both randomised steps seeded: the split, and the forest's bootstrap.
# Linear regression is closed-form, no seed to set
def test_fixed_seed_is_reproducible(df):
    X, y = split_features_target(df)
    first = split_data(X, y)[0].index.tolist()
    second = split_data(X, y)[0].index.tolist()
    assert first == second

    forest = build_models()["Random Forest"]
    X_train, _, y_train, _ = split_data(X, y)
    first_fit = forest.fit(X_train, y_train).predict(X_train[:5])
    second_fit = build_models()["Random Forest"].fit(X_train, y_train).predict(X_train[:5])
    assert (first_fit == second_fit).all()
