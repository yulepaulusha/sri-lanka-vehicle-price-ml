"""MSc class activity dashboard. Run: python -m streamlit run app.py"""
import logging
import pandas as pd
import streamlit as st
from vehicle_utils import (
    REFERENCE, REPORTED_RESULTS, encoder_categories, load_brand_models,
    load_models, make_vehicle, predict_prices,
)

st.set_page_config(page_title="Vehicle Price Predictor | Sri Lanka", page_icon="🚗", layout="centered")
st.markdown("""
<style>
.block-container {max-width: 1040px; padding-top: 2.4rem; padding-bottom: 2rem;}
h1 {font-size: 2rem !important; letter-spacing: -.04em;}
h2 {font-size: 1.35rem !important; letter-spacing: -.025em;}
h3 {font-size: 1.1rem !important;}
[data-testid="stMetric"] {background: #fff; border: 1px solid #e2e8f0;
 border-radius: 12px; padding: 16px 20px;}
[data-testid="stMetricValue"] {font-size: 1.65rem;}
div[data-testid="stForm"] {background: #fff; border-radius: 14px;}
.eyebrow {color: #2563eb; font-size: .75rem; font-weight: 650; letter-spacing: .12em;}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def resources():
    pipelines = load_models()
    categories = encoder_categories(pipelines[0])
    return pipelines, categories, load_brand_models(categories)


def category_input(label, column, categories, options=None, key=None):
    choices = categories[column] if options is None else options
    default = REFERENCE[column]
    return st.selectbox(
        label, choices, index=choices.index(default) if default in choices else 0,
        key=key or column,
        format_func=lambda value: value.replace("Not_Available", "Not available"),
    )


def main():
    st.markdown('<div class="eyebrow">MSC APPLIED MACHINE LEARNING · CLASS ACTIVITY</div>', unsafe_allow_html=True)
    st.title("Vehicle Price Predictor")
    st.write("Compare two regression models for **Sri Lankan used vehicle listing prices**.")
    st.caption("December 2024 – February 2025 · [Dataset by Prasad Nirmal on Kaggle](https://www.kaggle.com/datasets/prasadnirmal/srilankan-second-vehiclecar-price-dataset)")
    left, middle, right = st.columns(3)
    left.metric("Clean records", "9,770", help="9,788 original listings, with 18 exact duplicates removed after dropping the exported index.")
    middle.metric("Models", "2", help="Random Forest and XGBoost regression.")
    right.metric("Best reported R²", "0.8119", help="Random Forest, on the original 1,954-record test split.")
    st.info("Educational use only. Predictions estimate historical listing prices, not verified sale prices or current market values. Prices use dataset units; the currency scale is unconfirmed.")
    try:
        pipelines, categories, brand_models = resources()
    except Exception as exc:
        logging.exception("Could not load the saved pipelines")
        st.error(f"Could not load the saved models: {exc}")
        st.caption("Check the files in models/ and install requirements.txt with Python 3.12.")
        return

    st.divider()
    st.subheader("Vehicle details")
    st.caption("The default values are the Suzuki Celerio test example. Choose another vehicle to compare predictions.")
    # Brand sits outside the form so Model options refresh immediately.
    brand = category_input("Brand", "Brand", categories)
    model_options = brand_models.get(brand, categories["Model"])
    if not brand_models:
        st.caption("Brand/model pairs are unavailable. Choose a valid pair from the training categories.")
    with st.form("vehicle_form"):
        left, right = st.columns(2)
        with left:
            model = category_input("Model", "Model", categories, model_options, key=f"model_{brand}")
            year = st.number_input("Year of manufacture", min_value=1956, max_value=2024, value=2014, step=1)
            gear = category_input("Transmission", "Gear", categories)
            town = category_input("Town", "Town", categories)
        with right:
            engine = st.number_input("Engine capacity (cc)", min_value=573.0, max_value=4800.0, value=1000.0, step=1.0)
            fuel = category_input("Fuel type", "Fuel Type", categories)
            condition = category_input("Condition", "Condition", categories)
            leasing = category_input("Leasing", "Leasing", categories)
        st.markdown("**Comfort & equipment**")
        left, right = st.columns(2)
        with left:
            ac = category_input("Air conditioning", "AIR CONDITION", categories)
            steering = category_input("Power steering", "POWER STEERING", categories)
        with right:
            mirror = category_input("Power mirror", "POWER MIRROR", categories)
            window = category_input("Power window", "POWER WINDOW", categories)
        submitted = st.form_submit_button("Predict Price", type="primary", width="stretch")

    if submitted:
        values = {
            "Brand": brand, "Model": model, "YOM": year, "Engine (cc)": engine,
            "Gear": gear, "Fuel Type": fuel, "Town": town, "Leasing": leasing,
            "Condition": condition, "AIR CONDITION": ac, "POWER STEERING": steering,
            "POWER MIRROR": mirror, "POWER WINDOW": window,
        }
        try:
            vehicle = make_vehicle(values, categories, brand_models)
            with st.spinner("Comparing both models…"):
                prices = predict_prices(pipelines, vehicle)
            st.subheader("Estimated listing price")
            left, right = st.columns(2)
            left.metric("Random Forest · primary model", f"{prices[0]:,.2f}")
            right.metric("XGBoost · comparison", f"{prices[1]:,.2f}")
            st.caption("Dataset price units · currency conversion has not been applied.")
            chart = pd.DataFrame({"Model": ["Random Forest", "XGBoost"], "Estimated price": prices}).set_index("Model")
            st.bar_chart(chart, x_label="Model", y_label="Price (dataset units)", color="#2563eb", height=220)
            st.warning("Errors can be large for rare or expensive vehicles. Similar predictions do not guarantee accuracy.")
            with st.expander("Review prediction inputs"):
                st.dataframe(vehicle.T.rename(columns={0: "Value"}).astype(str), width="stretch")
        except ValueError as exc:
            st.error(str(exc))
        except Exception:
            logging.exception("Prediction failed")
            st.error("Prediction failed. Check the model files and selected inputs, then try again.")

    st.divider()
    st.subheader("Model performance")
    st.caption("Final Colab experiment · 80/20 random split · seed 42 · 7,816 training / 1,954 test listings")
    st.dataframe(
        REPORTED_RESULTS.T.style.format({"Random Forest": "{:.4f}", "XGBoost": "{:.4f}"}),
        width="stretch",
    )
    st.write("**Random Forest is the primary model:** it has lower MAE and RMSE and slightly higher R² on this test split. XGBoost trained faster in Colab.")
    with st.expander("Metrics and limitations"):
        st.markdown("""
- **MAE**: average absolute error, in dataset price units. Lower is better.
- **RMSE**: gives large errors more weight, in the same units. Lower is better.
- **R²**: performance relative to predicting the test-set mean. Higher is better; it is not an accuracy percentage.
- **Training time**: seconds on the original Colab hardware; local timing can differ.

Mileage was excluded because every cleaned record follows `(2025 − YOM) × 11,000`.
Related listings appear across the random split, and some test categories were unseen in training.
Future-market performance and prediction intervals have not been assessed.
No hyperparameter tuning study was performed. Further details are in the notebook.
""")
    st.caption("MSc Applied Machine Learning · Class activity · Random Forest + XGBoost")


if __name__ == "__main__":
    main()

