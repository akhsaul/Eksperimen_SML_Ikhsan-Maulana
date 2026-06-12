import os
import argparse
import joblib
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor


RANDOM_STATE = 42
TARGET_COLUMN = "num"

DROP_COLUMNS = ["id", "dataset"]

STRING_CATEGORICAL_FEATURES = ["sex", "cp", "restecg", "slope", "thal"]
NUMERIC_FEATURES = ["age", "trestbps", "chol", "thalch", "oldpeak", "ca"]
BOOL_FEATURES = ["fbs", "exang"]


def build_preprocessor():
    numeric_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                IterativeImputer(
                    estimator=RandomForestRegressor(
                        n_estimators=100,
                        max_depth=None,
                        min_samples_leaf=5,
                        max_features=1.0,
                        bootstrap=True,
                        max_samples=0.5,
                        random_state=RANDOM_STATE,
                        n_jobs=-1,
                    ),
                    max_iter=20,
                    tol=0.001,
                    initial_strategy="median",
                    imputation_order="ascending",
                    random_state=RANDOM_STATE,
                ),
            ),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
            ),
        ]
    )

    bool_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                    drop="if_binary",
                ),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, NUMERIC_FEATURES),
            ("cat", categorical_pipeline, STRING_CATEGORICAL_FEATURES),
            ("bool", bool_pipeline, BOOL_FEATURES),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    return preprocessor


def clean_raw_data(df):
    df = df.copy()

    df = df.drop(columns=DROP_COLUMNS, errors="ignore")

    for col in NUMERIC_FEATURES:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    for col in BOOL_FEATURES:
        df[col] = (
            df[col]
            .astype(str)
            .str.strip()
            .str.lower()
            .replace(
                {
                    "true": True,
                    "false": False,
                    "1": True,
                    "0": False,
                    "yes": True,
                    "no": False,
                    "nan": np.nan,
                    "none": np.nan,
                }
            )
        )

    df[TARGET_COLUMN] = pd.to_numeric(df[TARGET_COLUMN], errors="coerce")
    df = df.dropna(subset=[TARGET_COLUMN])

    # Binary classification:
    # 0 = tidak terkena heart disease
    # 1 = terkena heart disease
    df[TARGET_COLUMN] = (df[TARGET_COLUMN] > 0).astype(int)

    return df


def preprocess_data(input_path, output_dir, test_size=0.2):
    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_csv(input_path)
    df = clean_raw_data(df)

    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    # Split dulu sebelum fit preprocessing.
    # Ini mencegah data leakage.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    preprocessor = build_preprocessor()

    # Fit hanya pada X_train.
    X_train_processed = preprocessor.fit_transform(X_train)

    # X_test hanya transform, tidak ikut fit.
    X_test_processed = preprocessor.transform(X_test)

    feature_names = preprocessor.get_feature_names_out()

    X_train_df = pd.DataFrame(X_train_processed, columns=feature_names)
    X_test_df = pd.DataFrame(X_test_processed, columns=feature_names)

    y_train_df = pd.DataFrame({TARGET_COLUMN: y_train.reset_index(drop=True)})
    y_test_df = pd.DataFrame({TARGET_COLUMN: y_test.reset_index(drop=True)})

    X_train_df.to_csv(os.path.join(output_dir, "X_train.csv"), index=False)
    X_test_df.to_csv(os.path.join(output_dir, "X_test.csv"), index=False)
    y_train_df.to_csv(os.path.join(output_dir, "y_train.csv"), index=False)
    y_test_df.to_csv(os.path.join(output_dir, "y_test.csv"), index=False)

    joblib.dump(preprocessor, os.path.join(output_dir, "preprocessor.joblib"))

    print("Preprocessing selesai.")
    print(f"X_train shape: {X_train_df.shape}")
    print(f"X_test shape : {X_test_df.shape}")
    print(f"y_train shape: {y_train_df.shape}")
    print(f"y_test shape : {y_test_df.shape}")

    return X_train_df, X_test_df, y_train_df, y_test_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        type=str,
        default="../namadataset_raw/heart_disease.csv",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="namadataset_preprocessing",
    )

    parser.add_argument(
        "--test_size",
        type=float,
        default=0.2,
    )

    args = parser.parse_args()

    preprocess_data(
        input_path=args.input,
        output_dir=args.output_dir,
        test_size=args.test_size,
    )
