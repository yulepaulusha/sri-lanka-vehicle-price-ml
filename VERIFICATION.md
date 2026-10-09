# Local verification record

Verified on **9 October 2026** on Windows with the project Python **3.12.10** virtual environment.

## Outcome

The Streamlit dashboard starts and predicts locally. The reorganized notebook executes in full with **retraining and export disabled**. All six corrected, rounded test metrics and both reference predictions are preserved. Both original `.pkl` files are unchanged. No files have been committed, pushed, uploaded, published or deployed.

## Environment

| Dependency | Installed and tested version |
|---|---:|
| Python | 3.12.10 |
| Streamlit | 1.65.0 |
| pandas | 2.2.3 |
| NumPy | 2.1.3 |
| scikit-learn | 1.6.1 |
| XGBoost | 3.4.1 |
| joblib | 1.6.0 |
| Matplotlib | 3.10.0 |
| seaborn | 0.13.2 |
| nbformat | 5.11.1 |
| nbclient | 0.11.0 |
| ipykernel | 7.4.0 |

`pip install` completed inside `.venv`; `pip check` reported **No broken requirements found**. Full transitive versions are in `requirements-lock.txt`. No global packages were installed. Third-party Matplotlib/pyparsing deprecation warnings and Streamlit AppTest's bare-context warning occur during automated checks; they did not prevent execution. The live app log had no prediction error or dataframe serialization traceback after verification.

## Checks and results

| Check | Evidence / outcome |
|---|---|
| Dataset | Public Kaggle version 3 downloaded into ignored `data/`; shape 9,788 × 17; SHA256 matches the value in README |
| Cleaning | 18 duplicates after dropping exported index; 9,770 records; no target trimming |
| Mileage | Formula matches all 9,770 cleaned records; absent from the 13-feature prediction input |
| Split | 7,816 / 1,954 using the original shuffled 80/20 split and seed 42 |
| Preprocessing | Fitted medians and encoder vocabularies match reconstructed training rows; both models use the same input columns |
| Notebook | nbformat validation and AST parsing passed; all 28 cells executed through nbclient using the `.venv` kernel; outputs saved in `ML_activity.ipynb` |
| Saved-model evaluation | RF MAE/RMSE/R² = 5.8880/21.1365/0.8119; XGB = 6.9775/21.3378/0.8083, all matching recorded rounded values |
| Reference inference | RF 33.8631464451; XGB 30.0325775146; rounded 33.86 / 30.03; actual listing price 44.50 |
| XGBoost recovery | Original pickle fails direct Windows deserialization; unchanged native Model section recovered without fitting; native reload exactly matches 1,954 recovered test predictions |
| Automated tests | All **9** unittest checks passed, including both Streamlit AppTest form interactions |
| Unseen categories | Encoders produce finite outputs for an unseen model name; input validation rejects unknown values in every categorical feature |
| Invalid inputs | Nonfinite/out-of-range numeric values, fractional years, extra/missing features and wrong brand/model pairs rejected |
| Startup | Streamlit/Uvicorn running at `http://127.0.0.1:8501`, localhost only |
| Real browser | Installed Chrome controlled with Playwright; actual reference form submitted; 33.86 and 30.03 rendered |
| Brand filtering | Changed brand to TOYOTA, verified Celerio absent from model choices, and submitted a second successful prediction |
| Visual review | Inspected rendered desktop and 390 × 844 mobile screenshots; no horizontal mobile overflow; no browser page errors |
| Repository | Git already initialized; environment, CSV, logs, secrets and temporary artifacts ignored; source/model files remain uncommitted |

The initial in-app browser test runner failed to start its Node process. Browser testing then succeeded using installed Chrome and Playwright. Local screenshots and browser results are in ignored `artifacts/`; these checks were separate from Streamlit AppTest and server startup.

## Original artifact integrity

| Original artifact | SHA256 before/after |
|---|---|
| `random_forest_pipeline.pkl` | `2afdc136dc5beec5e411cac9932b9469ae6c74fcb5d1690a4cee0cb00a66a17c` |
| `xgboost_pipeline.pkl` | `0724bd41808dc73bc55c19680abb1601481446703378e16927706746e8e01cb7` |

The largest original file is 23,087,955 bytes, about 22.02 MiB. It is below GitHub's documented 50 MiB warning threshold, 100 MiB regular Git limit and 25 MiB browser-upload limit. [GitHub size-limit documentation](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github).

## Academic interpretation and remaining decisions

The reconstructed split contains **297** test listings sharing all final predictor values with at least one training listing. The full cleaned file has **858** duplicate predictor rows and **479** identical-predictor groups with differing target prices. Test rows include **209** unseen model names, **4** unseen brands and **1** unseen town. These observations limit generalization claims; they do not prove that each pair represents the same physical vehicle. The original split and results were preserved.

No retraining, hyperparameter tuning, repeated-split validation or future-market evaluation of the activity models was conducted. Training times remain the original Colab measurements, not independently reproduced local timings. A tiny unrelated diagnostic estimator was used only to establish that the installed XGBoost build could restore its own snapshot; it did not use the vehicle dataset or affect activity artifacts.

Kaggle's metadata states MIT and describes Price as LKR, but the scale of the stored numeric values remains unspecified. Currency conversion needs clarification. Raw data stays excluded. Grouped or chronological splits, target cleaning, tuning and new training would be separate experiments.

Lecturer-specific rubric, declaration format, cover sheet and Moodle upload limits were not supplied and remain unchecked. Nothing has been submitted externally.


## Project files

| File | Purpose |
|---|---|
| `ML_activity.ipynb` | Main notebook, including saved-model evaluation and final error analysis |
| `archive/ML_activity_colab_original.ipynb` | Unchanged Colab notebook and original outputs |
| `app.py` | Streamlit dashboard |
| `vehicle_utils.py` | Model loading, validation and prediction |
| `models/random_forest_pipeline.pkl` | Original final Random Forest pipeline |
| `models/xgboost_pipeline.pkl` | Original final Colab XGBoost pickle |
| `models/xgboost_model.ubj`, `models/xgboost_preprocessor.pkl` | Portable XGBoost model and fitted preprocessing used by the dashboard |
| `models/xgboost_recovery.json` | Recovery method, metrics and file hashes |
| `models/input_metadata.json` | Brand/model pairs from the 7,816 training rows |
| `recover_xgboost.py` | One-time recovery procedure; not run during normal startup |
| `tests/test_activity.py` | Nine model, validation and Streamlit tests |
| `requirements.txt`, `requirements-lock.txt` | Direct dependencies and complete tested environment |
| `.streamlit/config.toml`, `.gitignore` | Dashboard theme, localhost settings and repository exclusions |
| `README.md`, `VERIFICATION.md` | Project guide and testing evidence |

The dashboard requires the supplied model files but no CSV. Notebook evaluation also
requires Kaggle version 3. The local submission ZIP excludes `.git/`, `.venv/`, `data/`
and temporary `artifacts/` contents. The original notebook and all model files remain
unchanged during the writing cleanup.

## Academic integrity: draft AI-use acknowledgement

No university AI-use policy or prescribed declaration was supplied. The following
factual draft can be adapted to the required format:

> OpenAI Codex was used to assist with the local dashboard, code and documentation
> review, test setup and XGBoost portability work. Evaluation results were checked
> against the supplied notebook and saved models. The original vehicle models were
> not retrained during this assistance.

This draft describes the assistance recorded for the local project. It does not establish
whether AI tools were used in earlier work or claim that the student has independently
reviewed every change. Any required declaration should reflect the full project history
and the student's own review.


## Final writing review: 9 October 2026

Comments, docstrings, all 15 main-notebook Markdown cells, README wording and dashboard
text were simplified. No unnecessary assistant-credit or promotional text was found.
The remaining `GEMINI` entries are Isuzu model names. Dataset attribution, license
information, the XGBoost loading failure and recovery, material data limitations and
an optional factual AI-use acknowledgement were retained.

After cleanup:

- All nine automated tests passed, including both Streamlit form tests.
- Notebook format and Python syntax checks passed; all 28 cells executed with
  retraining and export disabled.
- Both models loaded through the submitted loading function. Reference predictions
  remained RF 33.8631464451 and XGBoost 30.0325775146.
- All six test metrics matched the recorded rounded results and README table.
- Both training parameter sets matched the original corrected Colab experiment.
- All six files in `models/`, the archived notebook and both requirements files had
  unchanged SHA256 hashes.
- Real Chrome submissions passed for the reference vehicle and a changed brand.
  Desktop and mobile screenshots were reviewed; no browser page errors or horizontal
  overflow were recorded at the 390 × 844 mobile viewport.
- `pip check` reported no broken requirements. The existing third-party deprecation,
  AppTest context and local notebook-kernel warnings did not prevent execution.
- Temporary scripts, logs, screenshots, dataset and environment files remain ignored.
  The 18-file submission ZIP was regenerated and its contents checked against source hashes.

No models were retrained or overwritten during cleanup, and no files were committed,
pushed, deployed, published or submitted. Currency scale, future-market performance,
university disclosure requirements and Moodle-specific submission limits remain unresolved.
