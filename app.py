"""Streamlit application for the real-estate price project. 

Three tabs: Overview and Data, Model Results, Price Predictor.

The app calls the modules from /src to execute: 
loading, validation, splitting, training, evaluation, selection;
and price prediction to collect user inputs for a property to predict a price.

Run from the project root:

    streamlit run app.py
"""

import logging
import matplotlib.pyplot as plt
import streamlit as st

from src.data import get_data_summary, load_data, validate_data
from src.logging_config import configure_logging
from src.modeling import predict_property, train_and_evaluate_models
from src.preprocessing import prepare_property_input

configure_logging()
logger = logging.getLogger(__name__)

st.set_page_config(page_title="Real Estate Price Prediction", layout="wide")


# Streamlit re-runs this whole script on every interaction, so both calls are cached. cache_data suits a DataFrame; cache_resource suits the trained models.
@st.cache_data
def get_data():
    """Load and validate the dataset once per session."""
    df = load_data()
    validate_data(df)
    return df


@st.cache_resource
def get_model_results():
    """Train and evaluate both models once per session."""
    return train_and_evaluate_models(get_data())


# start up failure checks mirror existing test coverage; but this block prevents app render  with failures surfacing mid-page tracebacks.
# /src module raises are more robust, critically for repurposing modules outside the streamlit app.
try:
    df = get_data()
    results = get_model_results()
    summary = get_data_summary(df)
except FileNotFoundError:
    logger.exception("Real-estate dataset was not found")
    st.error("The real-estate dataset could not be found.")
    st.stop()
except ValueError as exc:
    logger.exception("Data or input validation failed")
    st.error(str(exc))
    st.stop()
except Exception:
    logger.exception("Unexpected application failure")
    st.error("The application encountered an unexpected error.")
    st.stop()


st.title("Real Estate Price Prediction")
st.caption(
    "A modular rebuild of the Real_Estate notebook: linear regression and random forest models are compared with best test MAE model selected and used for a price predictor form. Regression analysis demonstration only on course data is not presented as real-world market data or used for actual pricing decisions."
)

tab_data, tab_models, tab_predict = st.tabs(
    ["Overview and Data", "Model Results", "Price Predictor"]
)


# --- Tab 1 -----------------------------------------------------------------
# Dataset shape, price summary, preview, distribution plots; figure comes from get_data_summary
with tab_data:
    st.header("Overview and Data")
    st.write(
        "This project predicts residential sale price from property "
        "characteristics."
    )

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Rows", f"{summary['rows']:,}")
    col2.metric("Columns", summary["columns"])
    col3.metric("Missing values", summary["missing_values"])
    col4.metric("Target", "price")

    st.subheader("Price summary")
    col1, col2, col3 = st.columns(3)
    col1.metric("Median", f"\\${summary['price_median']:,.0f}")
    col2.metric("Mean", f"\\${summary['price_mean']:,.0f}")
    col3.metric(
        "Range",
        f"\\${summary['price_min']:,.0f} to \\${summary['price_max']:,.0f}",
    )

    st.subheader("Data preview")
    st.dataframe(df.head(20), width="stretch")


    # Price spread and against the strongest feature.
    st.subheader("Distributions")
    col1, col2 = st.columns(2)

    with col1:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(df["price"], bins=40, color="#4C72B0", edgecolor="white")
        ax.set_xlabel("Sale price")
        ax.set_ylabel("Properties")
        ax.set_title("Price distribution")
        st.pyplot(fig, width="stretch")
        plt.close(fig)

    with col2:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.scatter(df["sqft"], df["price"], s=8, alpha=0.35, color="#4C72B0")
        ax.set_xlabel("Square feet")
        ax.set_ylabel("Sale price")
        ax.set_title("Price versus square feet")
        st.pyplot(fig, width="stretch")
        plt.close(fig)


# --- Tab 2 -----------------
# Metrics table, Selected model with criteria and two diagnostics.
with tab_models:
    st.header("Model Results")
    st.write(
        "Both models are trained on the same 80/20 split, stratified by "
        "property type."
    )

    metrics = results["metrics"]
    st.subheader("Linear regression versus random forest")
    st.dataframe(
        metrics.style.format({
            "Train MAE": "${:,.0f}",
            "Test MAE": "${:,.0f}",
            "MAE Gap": "${:,.0f}",
            "Train R2": "{:.3f}",
            "Test R2": "{:.3f}",
        }),
        width="stretch",
        hide_index=True,
    )
    st.caption(
        "MAE Gap is test MAE minus train MAE — the overfitting check. A small gap means the model performs about as well on unseen properties as on the ones it trained on."
    )

    st.subheader(f"Selected model: {results['selected_model_name']}")
    st.caption("Selected on lowest test MAE.")

    # Both plots use the selected model's held-out predictions. 
    actual = results["test_actual"]
    predicted = results["test_predictions"]
    residuals = actual - predicted

    col1, col2 = st.columns(2)

    with col1:
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.scatter(actual, predicted, s=12, alpha=0.4, color="#4C72B0")
        lo, hi = float(actual.min()), float(actual.max())
        ax.plot([lo, hi], [lo, hi], color="#C44E52", linewidth=1.5)
        ax.set_xlabel("Actual price")
        ax.set_ylabel("Predicted price")
        ax.set_title("Actual versus predicted (test set)")
        st.pyplot(fig, width="stretch")
        plt.close(fig)

    with col2:
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.scatter(predicted, residuals, s=12, alpha=0.4, color="#4C72B0")
        ax.axhline(0, color="#C44E52", linewidth=1.5)
        ax.set_xlabel("Predicted price")
        ax.set_ylabel("Residual (actual - predicted)")
        ax.set_title("Residuals (test set)")
        st.pyplot(fig, width="stretch")
        plt.close(fig)


# --- Tab 3 -----------------------------------------------------------------
# Collects nine values and predicts one price. 
# Remaining 4 variables derived from the 9 inputs (prepare_property_input)
with tab_predict:
    st.header("Price Predictor")
    st.write(
        f"Enter a property below. The prediction uses the selected model "
        f"({results['selected_model_name']})."
    )

    st.caption("Input ranges are constrained to the training dataset.")

    # Predictor feature value constraints are the datasets min/max ranges.
    with st.form("property_form"):
        col1, col2 = st.columns(2)

        with col1:
            year_sold = st.number_input(
                "Year sold", min_value=1993, max_value=2016, value=2015, step=1
            )
            year_built = st.number_input(
                "Year built", min_value=1880, max_value=2014, value=1990, step=1
            )
            sqft = st.number_input(
                "Square feet", min_value=500, max_value=7842, value=1800, step=50
            )
            lot_size = st.number_input(
                "Lot size", min_value=0, max_value=436471, value=8000, step=500
            )

        with col2:
            beds = st.number_input(
                "Bedrooms", min_value=1, max_value=5, value=3, step=1
            )
            baths = st.number_input(
                "Bathrooms", min_value=1, max_value=6, value=2, step=1
            )
            property_tax = st.number_input(
                "Monthly property tax", min_value=88, max_value=4508, value=300, step=10
            )
            insurance = st.number_input(
                "Monthly insurance", min_value=30, max_value=1374, value=100, step=10
            )

        property_type = st.selectbox("Property type", ["Single-family", "Condo"])

        submitted = st.form_submit_button("Predict price")


    # ValueError for the  user so built year later than sold is shown to user.
    if submitted:
        try:
            property_df = prepare_property_input({
                "year_sold": year_sold,
                "property_tax": property_tax,
                "insurance": insurance,
                "beds": beds,
                "baths": baths,
                "sqft": sqft,
                "year_built": year_built,
                "lot_size": lot_size,
                "property_type": property_type,
            })
            price = predict_property(results["selected_model"], property_df)

            st.success(f"Predicted sale price: ${price:,.0f}")


        except ValueError as exc:
            logger.exception("Property input validation failed")
            st.error(str(exc))
        except Exception:
            logger.exception("Unexpected prediction failure")
            st.error("The prediction could not be completed.")
