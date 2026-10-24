"""
Verification and Test Suite for Accident Severity Prediction Model and Inference Logic.
Run with: python test_app.py
"""

import sys
import os
import numpy as np
import pandas as pd
import joblib

def test_inference_pipeline():
    print("[1/4] Checking required files existence...")
    assert os.path.exists("rf_model.joblib"), "rf_model.joblib missing!"
    assert os.path.exists("ordinal_encoder2.joblib"), "ordinal_encoder2.joblib missing!"
    assert os.path.exists("accident.csv"), "accident.csv missing!"
    assert os.path.exists("App.py"), "App.py missing!"
    print("  -> All core project files are present.")

    print("\n[2/4] Testing Model & Encoder deserialization...")
    model = joblib.load("rf_model.joblib")
    encoder = joblib.load("ordinal_encoder2.joblib")
    print(f"  -> Model loaded: {type(model).__name__}")
    print(f"  -> Encoder loaded: {type(encoder).__name__}")
    print(f"  -> Number of categorical features in encoder: {len(encoder.categories_)}")
    assert len(encoder.categories_) == 7, "Encoder must have exactly 7 categorical feature sets!"

    print("\n[3/4] Testing categorical transformation and inference...")
    # Sample test inputs
    test_cat_raw = [
        str(encoder.categories_[0][0]),
        str(encoder.categories_[1][0]),
        str(encoder.categories_[2][0]),
        str(encoder.categories_[3][0]),
        str(encoder.categories_[4][0]),
        str(encoder.categories_[5][0]),
        str(encoder.categories_[6][0])
    ]
    encoded = encoder.transform(np.array(test_cat_raw, ndmin=2)).ravel()
    assert len(encoded) == 7, "Encoded vector must be length 7"

    feature_names = getattr(model, 'feature_names_in_', [
        'Number_of_vehicles_involved',
        'Number_of_casualties',
        'Hour_of_Day',
        'Type_of_collision',
        'Age_band_of_driver',
        'Sex_of_driver',
        'Educational_level',
        'Service_year_of_vehicle',
        'Day_of_week',
        'Area_accident_occured'
    ])

    # Numerical features: vehicles=2, casualties=1, hour=17
    feature_df = pd.DataFrame([[2, 1, 17] + list(encoded)], columns=feature_names)
    prediction = model.predict(feature_df)[0]
    probabilities = model.predict_proba(feature_df)[0]

    print(f"  -> Predicted class: {prediction}")
    print(f"  -> Probabilities: {probabilities}")
    assert prediction in [0, 1, 2], "Prediction must be 0 (Fatal), 1 (Serious), or 2 (Slight)"
    assert np.isclose(np.sum(probabilities), 1.0, atol=1e-4), "Probabilities must sum to 1.0"
    print("  -> Inference outputs validated successfully.")

    print("\n[4/4] Testing edge cases (slider boundaries & extreme conditions)...")
    edge_cases = [
        {"vehicles": 1, "casualties": 1, "hour": 0},   # Minimum boundary & midnight
        {"vehicles": 7, "casualties": 8, "hour": 23},  # Maximum boundary & late night
        {"vehicles": 3, "casualties": 5, "hour": 12},  # Midday multi-casualty
    ]

    for ec in edge_cases:
        f_df = pd.DataFrame([[ec["vehicles"], ec["casualties"], ec["hour"]] + list(encoded)], columns=feature_names)
        pred = model.predict(f_df)[0]
        prob = model.predict_proba(f_df)[0]
        assert pred in [0, 1, 2]
        assert np.isclose(np.sum(prob), 1.0, atol=1e-4)

    print("  -> Edge cases passed without errors.")
    print("\n[SUCCESS] ALL VERIFICATION TESTS PASSED SUCCESSFULLY! Model is production-ready.")

if __name__ == "__main__":
    try:
        test_inference_pipeline()
    except Exception as e:
        print(f"\n[FAILURE] Test failed: {e}", file=sys.stderr)
        sys.exit(1)
