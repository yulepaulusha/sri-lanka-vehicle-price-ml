"""Saved-model, input validation and Streamlit tests."""
import hashlib
import unittest
import warnings
import numpy as np
from sklearn.exceptions import InconsistentVersionWarning
from streamlit.testing.v1 import AppTest
from vehicle_utils import (
    BASE_DIR, FEATURES, REFERENCE, encoder_categories, load_models,
    load_brand_models, make_vehicle, predict_prices,
)


class VehicleActivityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with warnings.catch_warnings():
            warnings.simplefilter("error", InconsistentVersionWarning)
            cls.models = load_models()
        cls.categories = encoder_categories(cls.models[0])
        cls.pairs = load_brand_models(cls.categories)

    def test_model_contract_and_reference(self):
        frame = make_vehicle(REFERENCE, self.categories, self.pairs)
        self.assertEqual(list(frame.columns), FEATURES)
        self.assertEqual(frame.shape, (1, 13))
        np.testing.assert_allclose(predict_prices(self.models, frame), [33.86, 30.03], atol=0.005, rtol=0)

    def test_unknown_categories_rejected_by_dashboard_contract(self):
        for column in self.categories:
            values = dict(REFERENCE, **{column: "__UNSEEN__"})
            with self.subTest(column=column), self.assertRaises(ValueError):
                make_vehicle(values, self.categories, self.pairs)

    def test_encoder_handles_novel_categories(self):
        frame = make_vehicle(REFERENCE, self.categories, self.pairs)
        frame["Model"] = "__UNSEEN__"
        self.assertTrue(all(np.isfinite(model.predict(frame)).all() for model in self.models))

    def test_bad_numeric_values_rejected(self):
        for column, invalid in [("YOM", 2014.5), ("YOM", 2026), ("YOM", float("nan")),
                                ("YOM", True), ("Engine (cc)", 0), ("Engine (cc)", float("inf")),
                                ("Engine (cc)", "1000")]:
            with self.subTest(column=column, invalid=invalid), self.assertRaises(ValueError):
                make_vehicle(dict(REFERENCE, **{column: invalid}), self.categories, self.pairs)

    def test_missing_or_extra_features_rejected(self):
        extra = dict(REFERENCE, **{"Millage(KM)": 121000})
        missing = dict(REFERENCE)
        missing.pop("Town")
        for values in [extra, missing]:
            with self.assertRaises(ValueError):
                make_vehicle(values, self.categories, self.pairs)

    def test_invalid_brand_model_pair_rejected(self):
        self.assertTrue(self.pairs, "Training-derived model pairing metadata is required for the submitted app.")
        wrong_brand = next(brand for brand, models in self.pairs.items() if "CELERIO ZXI" not in models)
        with self.assertRaises(ValueError):
            make_vehicle(dict(REFERENCE, Brand=wrong_brand), self.categories, self.pairs)

    def test_streamlit_reference_form_submission(self):
        app = AppTest.from_file(str(BASE_DIR / "app.py"), default_timeout=45).run()
        self.assertFalse(app.exception)
        self.assertEqual(len(app.selectbox), 11)
        self.assertEqual(len(app.number_input), 2)
        app.button[0].click().run()
        self.assertFalse(app.exception)
        self.assertFalse(app.error)
        self.assertEqual([metric.value for metric in app.metric][-2:], ["33.86", "30.03"])
        self.assertEqual(app.dataframe[0].value.shape, (13, 1))

    def test_streamlit_brand_change_filters_models(self):
        app = AppTest.from_file(str(BASE_DIR / "app.py"), default_timeout=45).run()
        app.selectbox(key="Brand").select("TOYOTA").run()
        self.assertFalse(app.exception)
        selected = app.selectbox(key="model_TOYOTA")
        self.assertEqual(selected.options, self.pairs["TOYOTA"])
        self.assertNotIn("CELERIO ZXI", selected.options)
        app.button[0].click().run()
        self.assertFalse(app.exception)
        self.assertFalse(app.error)
        self.assertEqual(len(app.metric), 5)

    def test_accepted_artifact_hashes(self):
        expected = {"random_forest_pipeline.pkl": "2afdc136dc5beec5e411cac9932b9469ae6c74fcb5d1690a4cee0cb00a66a17c",
                    "xgboost_pipeline.pkl": "0724bd41808dc73bc55c19680abb1601481446703378e16927706746e8e01cb7"}
        for name, checksum in expected.items():
            self.assertEqual(hashlib.sha256((BASE_DIR / "models" / name).read_bytes()).hexdigest(), checksum)


if __name__ == "__main__":
    unittest.main()
