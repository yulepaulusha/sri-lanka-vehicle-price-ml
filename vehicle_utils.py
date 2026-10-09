"""Model loading, input validation and prediction."""
from pathlib import Path
import json
import hashlib
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor
import joblib
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
FEATURES = [
    "Brand", "Model", "YOM", "Engine (cc)", "Gear", "Fuel Type", "Town",
    "Leasing", "Condition", "AIR CONDITION", "POWER STEERING",
    "POWER MIRROR", "POWER WINDOW",
]
# Reference record from the final Colab test split.
REFERENCE = dict(zip(FEATURES, [
    "SUZUKI", "CELERIO ZXI", 2014, 1000.0, "Manual", "Petrol", "Weligama",
    "No Leasing", "USED", "Available", "Available", "Available", "Available",
]))
REPORTED_RESULTS = pd.DataFrame({
    "Model": ["Random Forest", "XGBoost"],
    "MAE": [5.8880, 6.9775], "RMSE": [21.1365, 21.3378],
    "R²": [0.8119, 0.8083], "Training time (s)": [27.07, 1.43],
}).set_index("Model")


def encoder_categories(pipeline):
    """Return categories learned during training."""
    preprocessor = pipeline.named_steps["preprocessor"]
    columns = next(cols for name, _, cols in preprocessor.transformers_ if name == "categorical")
    encoder = preprocessor.named_transformers_["categorical"].named_steps["encoder"]
    return {column: values.tolist() for column, values in zip(columns, encoder.categories_)}


def load_models():
    """Load the saved models and check their inputs and file hashes."""
    rf = joblib.load(BASE_DIR / "models/random_forest_pipeline.pkl")
    # Load the portable XGBoost model with its original fitted preprocessor.
    manifest = json.loads((BASE_DIR / "models/xgboost_recovery.json").read_text(encoding="utf-8"))
    for name, checksum in manifest["files"].items():
        if hashlib.sha256((BASE_DIR / "models" / name).read_bytes()).hexdigest() != checksum:
            raise ValueError(f"Portable XGBoost artifact checksum failed: {name}")
    xgb = XGBRegressor()
    xgb.load_model(BASE_DIR / "models/xgboost_model.ubj")
    xgb_pipeline = Pipeline([
        ("preprocessor", joblib.load(BASE_DIR / "models/xgboost_preprocessor.pkl")),
        ("regressor", xgb),
    ])
    pipelines = (rf, xgb_pipeline)
    for pipeline in pipelines:
        if list(pipeline.feature_names_in_) != FEATURES:
            raise ValueError("Saved model inputs differ from the final 13 features.")
    if encoder_categories(pipelines[0]) != encoder_categories(pipelines[1]):
        raise ValueError("Saved models use different category values.")
    return pipelines


def load_brand_models(categories):
    """Load training brand/model pairs, if available."""
    metadata_path = BASE_DIR / "models" / "input_metadata.json"
    if not metadata_path.exists():
        return {}
    metadata = json.loads(metadata_path.read_text(encoding="utf-8-sig"))
    return {
        brand: [model for model in models if model in categories["Model"]]
        for brand, models in metadata["brand_models"].items()
        if brand in categories["Brand"]
    }


def make_vehicle(values, categories, brand_models=None):
    """Reject unsupported categories and invalid numbers before prediction."""
    if set(values) != set(FEATURES):
        raise ValueError("Provide exactly the 13 vehicle features.")
    for column, allowed in categories.items():
        if values[column] not in allowed:
            raise ValueError(f"{column}: choose a category seen during training.")
    for column, lower, upper in [("YOM", 1956, 2024), ("Engine (cc)", 573, 4800)]:
        value = values[column]
        if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
            raise ValueError(f"{column} must be numeric.")
        if not np.isfinite(value) or not lower <= value <= upper:
            raise ValueError(f"{column} must be between {lower} and {upper} (recorded dataset range).")
    if int(values["YOM"]) != values["YOM"]:
        raise ValueError("Year of manufacture must be a whole year.")
    if brand_models and values["Model"] not in brand_models.get(values["Brand"], []):
        raise ValueError("Choose a model observed for the selected brand in training.")
    return pd.DataFrame([values], columns=FEATURES)


def predict_prices(pipelines, vehicle):
    prices = tuple(float(pipeline.predict(vehicle)[0]) for pipeline in pipelines)
    if not all(np.isfinite(price) and price >= 0 for price in prices):
        raise ValueError("The models returned an invalid price; check the inputs and model files.")
    return prices
