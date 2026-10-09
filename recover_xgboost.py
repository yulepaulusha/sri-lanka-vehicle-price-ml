"""Recover the original Colab XGBoost model for Windows without retraining.

The saved Config section fails to load on Windows. This script checks the source
hash and loads its unchanged native Model section. Existing files are preserved.
"""
import hashlib
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from xgboost import Booster, XGBRegressor
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "models/xgboost_pipeline.pkl"
EXPECTED = "0724bd41808dc73bc55c19680abb1601481446703378e16927706746e8e01cb7"
assert hashlib.sha256(SOURCE.read_bytes()).hexdigest() == EXPECTED
native_path = ROOT / "models/xgboost_model.ubj"
prep_path = ROOT / "models/xgboost_preprocessor.pkl"
if native_path.exists() or prep_path.exists():
    raise FileExistsError("Portable artifacts already exist; this recovery will not overwrite them.")

original_restore = Booster.__setstate__
recovery_details = {}
def restore_native(self, state):
    snapshot = bytes(state["handle"])
    # The verified snapshot contains Config followed by Model.
    # Extract the Model bytes unchanged.
    marker = b"L" + (5).to_bytes(8, "big") + b"Model"
    assert snapshot.count(marker) == 1 and snapshot.endswith(b"}}")
    native = bytearray(snapshot[snapshot.index(marker) + len(marker):-1])
    assert native.startswith(b"{")
    self.__init__()
    self.load_model(native)
    recovery_details["embedded_native_sha256"] = hashlib.sha256(native).hexdigest()

try:
    # Use this restore method only while reading the original pickle.
    Booster.__setstate__ = restore_native
    recovered = joblib.load(SOURCE)
finally:
    Booster.__setstate__ = original_restore

# Check the recorded test results before saving the portable files.
df = pd.read_csv(ROOT / "data/car_price_dataset.csv").drop(columns=["Unnamed: 0"]).drop_duplicates().reset_index(drop=True)
X = df.drop(columns=["Price", "Date", "Millage(KM)"])
_, X_test, _, y_test = train_test_split(X, df["Price"], test_size=0.20, random_state=42)
predictions = recovered.predict(X_test)
metrics = [mean_absolute_error(y_test, predictions), np.sqrt(mean_squared_error(y_test, predictions)), r2_score(y_test, predictions)]
np.testing.assert_allclose(metrics, [6.9775, 21.3378, 0.8083], atol=0.00005, rtol=0)
np.testing.assert_allclose(predictions[5], 30.03, atol=0.005, rtol=0)
recovered.named_steps["regressor"].save_model(native_path)
joblib.dump(recovered.named_steps["preprocessor"], prep_path)
regressor = XGBRegressor()
regressor.load_model(native_path)
portable = Pipeline([("preprocessor", joblib.load(prep_path)), ("regressor", regressor)])
np.testing.assert_array_equal(portable.predict(X_test), predictions)
assert hashlib.sha256(SOURCE.read_bytes()).hexdigest() == EXPECTED
recovery_details.update({
    "source_pickle_sha256": EXPECTED,
    "method": "Extracted unchanged native Model UBJSON section; loaded using Booster.load_model; exported using XGBRegressor.save_model. No fitting or tree modifications.",
    "reason": "Original Colab Config snapshot raises input stream corrupted on Windows XGBoost 3.4.1.",
    "MAE": metrics[0], "RMSE": metrics[1], "R2": metrics[2],
    "reference_prediction": float(predictions[5]),
    "portable_vs_recovered_test_predictions": "exactly equal on all 1,954 test rows",
    "files": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in [native_path, prep_path]},
})
(ROOT / "models/xgboost_recovery.json").write_text(json.dumps(recovery_details, indent=2) + "\n", encoding="utf-8")
print(json.dumps(recovery_details, indent=2))
