"""
Model Training & Evaluation Pipeline for Road Traffic Accident Severity Prediction.

Features:
- Loads Addis Ababa Police Department accident dataset.
- Cleans and extracts hour from timestamp.
- Encodes categorical predictors with scikit-learn OrdinalEncoder.
- Trains Random Forest Classifier with class-weight balancing.
- Generates comprehensive evaluation metrics (F1-score, Precision, Recall, Confusion Matrix).
- Serializes trained model and encoder for production inference in Streamlit.
"""

import os
import argparse
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import classification_report, confusion_matrix, f1_score


def load_and_preprocess_data(csv_path="accident.csv"):
    """Loads CSV, extracts Hour_of_Day, handles missing values."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset not found at {csv_path}")

    print(f"[*] Loading dataset from {csv_path}...")
    df = pd.read_csv(csv_path)
    print(f"[*] Loaded {len(df)} records with {df.shape[1]} columns.")

    # Extract Hour of Day
    if 'Time' in df.columns:
        parsed_time = pd.to_datetime(df['Time'], errors='coerce')
        df['Hour_of_Day'] = parsed_time.dt.hour.fillna(12).astype(int)
    elif 'Hour_of_Day' not in df.columns:
        df['Hour_of_Day'] = 12

    # Fill missing values for core categorical features
    cat_features = [
        'Type_of_collision',
        'Age_band_of_driver',
        'Sex_of_driver',
        'Educational_level',
        'Service_year_of_vehicle',
        'Day_of_week',
        'Area_accident_occured'
    ]
    for col in cat_features:
        if col in df.columns:
            df[col] = df[col].fillna('Unknown').astype(str).str.strip()
        else:
            df[col] = 'Unknown'

    num_features = [
        'Number_of_vehicles_involved',
        'Number_of_casualties',
        'Hour_of_Day'
    ]
    for col in num_features:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(1)

    # Encode Target:
    # 0: Fatal injury, 1: Serious Injury, 2: Slight Injury
    target_mapping = {
        'Fatal injury': 0,
        'Serious Injury': 1,
        'Slight Injury': 2
    }
    y = df['Accident_severity'].map(target_mapping)
    valid_mask = y.notna()

    df_clean = df[valid_mask].copy()
    y_clean = y[valid_mask].astype(int)

    return df_clean, y_clean, cat_features, num_features


def train_pipeline(data_path="accident.csv", n_estimators=700, max_depth=20, test_size=0.2, random_state=42):
    """Executes the full training, evaluation, and serialization workflow."""
    df, y, cat_cols, num_cols = load_and_preprocess_data(data_path)

    # Fit OrdinalEncoder on categorical features
    print("[*] Fitting OrdinalEncoder...")
    encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    encoded_cats = pd.DataFrame(
        encoder.fit_transform(df[cat_cols]),
        columns=cat_cols,
        index=df.index
    )

    # Combine numerical and encoded categorical features
    X = pd.concat([df[num_cols], encoded_cats], axis=1)
    feature_order = num_cols + cat_cols
    X = X[feature_order]

    print(f"[*] Total features: {X.shape[1]}")
    print(f"[*] Class distribution:\n{y.value_counts(normalize=True).rename(index={0:'Fatal', 1:'Serious', 2:'Slight'})}")

    # Train / Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    print(f"[*] Training RandomForestClassifier (n_estimators={n_estimators}, max_depth={max_depth})...")
    # Using class_weight='balanced' to address extreme imbalance without data leakage
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        class_weight='balanced_subsample',
        random_state=random_state,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    # Evaluation
    print("\n" + "="*50)
    print("           MODEL EVALUATION ON TEST SET")
    print("="*50)
    preds = model.predict(X_test)
    class_names = ['Fatal injury', 'Serious Injury', 'Slight Injury']
    report = classification_report(y_test, preds, target_names=class_names, digits=4)
    print(report)

    weighted_f1 = f1_score(y_test, preds, average='weighted')
    macro_f1 = f1_score(y_test, preds, average='macro')
    print(f"[*] Weighted F1-Score: {weighted_f1:.4f}")
    print(f"[*] Macro F1-Score:    {macro_f1:.4f}")

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, preds))
    print("="*50 + "\n")

    # Serialize Artifacts
    print("[*] Serializing model and encoder artifacts...")
    joblib.dump(encoder, "ordinal_encoder2.joblib")
    joblib.dump(model, "rf_model.joblib", compress=9)
    print("[+] Successfully saved `ordinal_encoder2.joblib` and `rf_model.joblib`.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Road Traffic Accident Severity Model")
    parser.add_argument("--data", default="accident.csv", help="Path to accident dataset CSV")
    parser.add_argument("--trees", type=int, default=700, help="Number of trees in Random Forest")
    parser.add_argument("--depth", type=int, default=20, help="Max depth of decision trees")
    args = parser.parse_args()

    train_pipeline(data_path=args.data, n_estimators=args.trees, max_depth=args.depth)
