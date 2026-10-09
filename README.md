# Sri Lankan Used Vehicle Listing Price Prediction

## 1. Project overview

This MSc Applied Machine Learning class activity compares Random Forest and XGBoost regression for predicting used vehicle listing prices in Sri Lanka. A Streamlit dashboard demonstrates predictions from both saved models. It is an educational experiment, not a current market valuation service.

## 2. Dataset and source

The data comes from Prasad Nirmal's [SriLankan Vehicle(Car) Price Dataset](https://www.kaggle.com/datasets/prasadnirmal/srilankan-second-vehiclecar-price-dataset), **version 3**, updated 9 February 2025. The source describes listings collected from ikman.lk and Riyasewana.com.

- 9,788 original records and 17 columns; listing dates: **3 December 2024 – 5 February 2025**.
- Removing the exported index reveals 18 exact duplicate listings, leaving **9,770 records**.
- Target: `Price`, in the original dataset units.
- Split: **7,816 training / 1,954 test records**, with `test_size=0.20` and `random_state=42`.

Kaggle metadata checked on 9 October 2026 states an **MIT** license and describes prices as LKR, but does not explain the scale of values such as 44.50. No rupee or lakh conversion is applied. The CSV is excluded from Git; underlying marketplace rights should also be checked before redistributing raw data.

## 3. Implementation approach

The final predictors are:

```text
Brand, Model, YOM, Engine (cc), Gear, Fuel Type, Town, Leasing,
Condition, AIR CONDITION, POWER STEERING, POWER MIRROR, POWER WINDOW
```

All 9,770 cleaned records satisfy `Millage(KM) = (2025 - YOM) * 11000`. Mileage was removed because it provides no independent odometer information. The exported index and listing date are also excluded from predictors. Expensive listings are retained; no target transformation or price trimming is applied.

Each sklearn pipeline contains preprocessing and a regressor. Numeric missing values use the training median. Categorical missing values use the training mode, followed by sparse one-hot encoding with `handle_unknown="ignore"`. Preprocessing is fitted on training data only. The dashboard uses fitted categories and training-observed brand/model pairs, validates inputs and compares both predictions. Numeric inputs are limited to the dataset ranges: 1956–2024 and 573–4800 cc.

### XGBoost portability

The original Colab XGBoost pickle failed to load on Windows with `input stream corrupted`, despite matching XGBoost 3.4.1. Its native model was recovered without changing the trees and saved as `models/xgboost_model.ubj`, with the original fitted preprocessor in `models/xgboost_preprocessor.pkl`. The dashboard loads these files; both original pickles are preserved.

Native export/reload gave identical predictions on all 1,954 test records. Recovery details and hashes are recorded in `models/xgboost_recovery.json` and [VERIFICATION.md](VERIFICATION.md). `recover_xgboost.py` documents the one-time procedure and is not needed for normal startup. See [XGBoost's model IO documentation](https://xgboost.readthedocs.io/en/stable/tutorials/saving_model.html) for the limitations of pickle snapshots.

## 4. Models and parameters

Random Forest averages randomized trees to reduce variance. XGBoost adds trees sequentially to reduce remaining errors. Both can model nonlinear relationships and interactions in tabular data.

| Model | Parameters | Reason |
|---|---|---|
| RF | `n_estimators=150` | Average predictions from 150 trees |
| RF | `max_depth=18` | Limit tree complexity |
| RF | `min_samples_split=4`, `min_samples_leaf=2` | Avoid very small partitions |
| RF | `max_features=0.8` | Diversify trees by sampling encoded features at each split |
| XGB | `n_estimators=300`, `learning_rate=0.05` | Use 300 rounds with small contributions per tree |
| XGB | `max_depth=6` | Limit individual tree complexity |
| XGB | `subsample=0.85`, `colsample_bytree=0.85` | Sample rows and encoded features to reduce overfitting |
| XGB | `reg_alpha=0.1`, `reg_lambda=2.0` | Apply L1/L2 penalties to leaf weights |
| XGB | `tree_method="hist"`, `objective="reg:squarederror"` | Use efficient histogram splits and squared-error regression |
| Both | `random_state=42`, `n_jobs=-1` during training | Fix randomness and use available CPU cores |

No tuning or early-stopping study was performed, so these settings are not claimed to be optimal.

## 5. Results

| Metric | Random Forest | XGBoost |
|---|---:|---:|
| MAE | 5.8880 | 6.9775 |
| RMSE | 21.1365 | 21.3378 |
| R² | 0.8119 | 0.8083 |
| Colab training time | 27.07 s | 1.43 s |

MAE is average absolute error; RMSE gives large errors more weight. Both use dataset price units. R² compares predictions with the test-set mean and is not an accuracy percentage. Random Forest is the primary model because it has lower errors and slightly higher R² on this split. XGBoost trained faster in Colab.

Saved-model evaluation reproduced all six rounded metrics locally. Training was not repeated, so the times remain the original Colab measurements. The preliminary mileage-inclusive results are reported separately in the notebook.

The reference record is a Suzuki Celerio ZXI, 2014, 1000 cc, manual, petrol, Weligama, no leasing, used, with all four equipment features available (`X_test.iloc[[5]]`). Its actual price is **44.50**; predictions are **RF 33.86** and **XGBoost 30.03**. Both models underpredict this example.

## 6. How to run the project

Use **Python 3.12**; the tested version is 3.12.10. Run PowerShell from the project root. Reuse an existing `.venv` if available; otherwise create one:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Open [localhost:8501](http://127.0.0.1:8501). Choose the vehicle details and click **Predict Price**. The dashboard does not need the CSV. Use only trusted submitted model files, because pickle/joblib files can execute code. `requirements-lock.txt` contains the full tested environment, including transitive packages and dataset download tools; install it instead of `requirements.txt` when reproducing that environment.

### Notebook dataset and execution

Download Kaggle **version 3** and extract `car_price_dataset.csv` into `data/`, or use the public endpoint:

```powershell
New-Item -ItemType Directory -Path data -Force | Out-Null
Invoke-WebRequest -Uri 'https://www.kaggle.com/api/v1/datasets/download/prasadnirmal/srilankan-second-vehiclecar-price-dataset?datasetVersionNumber=3' -OutFile data\dataset-v3.zip
Expand-Archive -LiteralPath data\dataset-v3.zip -DestinationPath data -Force
Get-FileHash data\car_price_dataset.csv -Algorithm SHA256
```

Expected CSV SHA256: `4208d2673cf4d9c8d3d6042dc8a1b97004671812441975f5879550059fbce1ab`.
If the download endpoint is unavailable, use Kaggle's website. A different file or version needs investigation before comparing results.

Open `ML_activity.ipynb` in VS Code, select `.venv\Scripts\python.exe` as the kernel, set the project root as the working directory, then **Restart Kernel → Run All**. The Python and Jupyter extensions are needed. Keep `RETRAIN=False` and `EXPORT_RETRAINED=False` to evaluate the saved models without modifying them. A new training run is a separate experiment; optional exports go to `models/retrained/`.

The unchanged original notebook is in `archive/ML_activity_colab_original.ipynb`. It contains Colab-specific operations and earlier exports, so use `ML_activity.ipynb` for local execution.

Run the tests with:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

All nine tests, full notebook execution and real Chrome form submissions passed. [VERIFICATION.md](VERIFICATION.md) contains the environment, file inventory, detailed evidence, known warnings and draft AI-use acknowledgement. Adapt the acknowledgement to the university's disclosure requirements.

## 7. Limitations

- Prices are historical advertised listings, not verified transactions. Future-market performance is untested.
- The random split contains **297 test records with the same final predictors as a training record**. Related listings may make results optimistic, although matching attributes do not prove that the same vehicle was listed twice.
- Some test categories were unseen in training: 209 rows with unseen model names, 4 with unseen brands and 1 with an unseen town.
- Rare expensive vehicles have large errors. The notebook includes error tables and residual plots rather than removing these observations.
- Repeated fractional prices are not fully explained by the source. Mileage is derived from year, and the currency scale remains unclear.
- No tuning study, repeated-split uncertainty analysis, significance test or calibrated prediction interval was performed. Similar predictions from both models do not establish confidence.

A grouped or chronological split would require a separate experiment. The current methodology and reported results are unchanged. Moodle upload limits and lecturer-specific declarations or cover sheets should be checked before submission. The submission ZIP excludes the CSV, `.venv`, caches and temporary files; nothing has been submitted externally.
