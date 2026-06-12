import warnings
import os
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import kagglehub
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore", category=ConvergenceWarning)


def download_dataset_from_kagglehub(
    kaggle_dataset: str,
    kaggle_csv_filename: str,
    dataset_path: str,
) -> str:
    csv_path = kagglehub.dataset_download(
        kaggle_dataset,
        path=kaggle_csv_filename,
        output_dir=dataset_path,
        force_download=False,
    )
    return csv_path


def load_dataset(csv_path: str) -> pd.DataFrame:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset tidak ditemukan: {csv_path}")

    return pd.read_csv(csv_path)


def save_and_show_plot(filename: str, output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename), dpi=150, bbox_inches="tight")
    if "agg" not in plt.get_backend().lower():
        plt.show()
    plt.close()


def check_dataset(df: pd.DataFrame) -> None:
    n_rows, n_cols = df.shape
    print(f"Dataset shape: {n_rows} Rows x {n_cols} Columns", end="\n\n")

    print("Describe dataset:")
    print(df.describe(include="all"), end="\n\n")

    missing_counts = df.isnull().sum()
    columns_with_missing = missing_counts[missing_counts > 0].sort_values(
        ascending=False
    )
    columns_without_missing = missing_counts[missing_counts == 0]

    print("Kolom yang punya missing values:")
    if columns_with_missing.empty:
        print("Tidak ada kolom yang punya missing values.")
    else:
        missing_table = columns_with_missing.rename("missing_count").to_frame()
        missing_table["missing_pct"] = (
            missing_table["missing_count"] / len(df) * 100
        ).round(2)
        print(missing_table)
    print()

    print("Kolom yang tidak punya missing values:")
    if columns_without_missing.empty:
        print("Tidak ada kolom yang bebas missing values.")
    else:
        print(columns_without_missing.index.tolist())
    print()

    duplicate_rows = df[df.duplicated(keep=False)]
    print(f"Jumlah baris duplikat: {df.duplicated().sum()}")
    print("Baris yang duplikat:")
    if duplicate_rows.empty:
        print("Tidak ada baris duplikat.")
    else:
        print(duplicate_rows)
    print()

    print("Lowest value, highest value, dan total unique value semua kolom:")
    print(build_column_summary(df), end="\n\n")


def build_column_summary(df: pd.DataFrame) -> pd.DataFrame:
    summary_rows = []

    for col in df.columns:
        series = df[col]
        non_null_series = series.dropna()

        if non_null_series.empty:
            lowest_value = np.nan
            highest_value = np.nan
        elif pd.api.types.is_numeric_dtype(series):
            lowest_value = series.min(skipna=True)
            highest_value = series.max(skipna=True)
        else:
            string_series = non_null_series.astype(str)
            lowest_value = string_series.min()
            highest_value = string_series.max()

        summary_rows.append(
            {
                "column": col,
                "dtype": series.dtype,
                "missing_count": series.isna().sum(),
                "lowest_value": lowest_value,
                "highest_value": highest_value,
                "total_unique_value": series.nunique(dropna=True),
            }
        )

    return pd.DataFrame(summary_rows)


def print_num_analysis(df: pd.DataFrame, target_column: str) -> None:
    health_status = np.where(df[target_column] > 0, "Sakit", "Sehat")
    health_counts = pd.Series(health_status, name="health_status").value_counts()
    health_summary = pd.DataFrame(
        {
            "count": health_counts,
            "percentage": (health_counts / len(df) * 100).round(2),
        }
    )

    stage_counts = df[target_column].value_counts().sort_index()
    stage_summary = pd.DataFrame(
        {
            "count": stage_counts,
            "percentage": (stage_counts / len(df) * 100).round(2),
        }
    )

    print("Kolom num - jumlah dan persentase orang sakit dan sehat:")
    print(health_summary, end="\n\n")

    print("Kolom num - jumlah dan persentase berdasarkan stage:")
    print(stage_summary, end="\n\n")


def plot_all_column_distributions(df: pd.DataFrame, output_dir: str) -> None:
    for col in df.columns:
        plt.figure(figsize=(10, 5))

        if pd.api.types.is_numeric_dtype(df[col]):
            sns.histplot(data=df, x=col, bins=30, kde=True)
            plt.title(f"Distribusi Kolom {col}")
            plt.xlabel(col)
            plt.ylabel("Frekuensi")
            save_and_show_plot(f"distribution_{col}.png", output_dir)
        else:
            order = df[col].value_counts(dropna=False).index
            sns.countplot(data=df, x=col, order=order)
            plt.title(f"Distribusi Kolom {col}")
            plt.xlabel(col)
            plt.ylabel("Jumlah")
            plt.xticks(rotation=30, ha="right")
            save_and_show_plot(f"distribution_{col}.png", output_dir)

    # Analisis: distribusi semua kolom menunjukkan dataset didominasi pasien Male
    # sebanyak 726 baris, chest pain asymptomatic sebanyak 496 baris, dan stage num
    # 0/1. Kolom trestbps dan chol memiliki nilai minimum 0 sehingga perlu ditinjau
    # sebagai nilai yang tidak wajar untuk tekanan darah dan kolesterol.


def plot_relationships(df: pd.DataFrame, output_dir: str) -> None:
    plt.figure(figsize=(10, 5))
    sns.histplot(data=df, x="age", hue="sex", bins=25, kde=True, multiple="layer")
    plt.title("Distribusi Age Berdasarkan Sex")
    plt.xlabel("Age")
    plt.ylabel("Frekuensi")
    save_and_show_plot("relationship_sex_age.png", output_dir)
    # Analisis: distribusi umur Male dan Female relatif mirip. Median umur Male
    # adalah 55 tahun, sedikit lebih tinggi daripada Female yaitu 53 tahun, dan
    # jumlah Male jauh lebih banyak daripada Female.

    plt.figure(figsize=(10, 5))
    sns.countplot(data=df, x="cp", hue="sex", order=df["cp"].value_counts().index)
    plt.title("Hubungan Chest Pain dan Sex")
    plt.xlabel("Chest Pain")
    plt.ylabel("Jumlah")
    plt.xticks(rotation=25, ha="right")
    save_and_show_plot("relationship_chest_pain_sex.png", output_dir)
    # Analisis: kategori chest pain asymptomatic paling dominan, terutama pada Male
    # sebanyak 426 baris. Pada Female, atypical angina muncul 61 baris dan jauh lebih
    # banyak daripada typical angina yang hanya 10 baris.

    plt.figure(figsize=(10, 5))
    chest_pain_order = df.groupby("cp")["age"].median().sort_values().index
    sns.boxplot(data=df, x="cp", y="age", order=chest_pain_order)
    plt.title("Hubungan Chest Pain dan Age")
    plt.xlabel("Chest Pain")
    plt.ylabel("Age")
    plt.xticks(rotation=25, ha="right")
    save_and_show_plot("relationship_chest_pain_age.png", output_dir)
    # Analisis: asymptomatic dan typical angina cenderung muncul pada umur yang lebih
    # tinggi dengan rata-rata sekitar 55 tahun, sedangkan atypical angina memiliki
    # rata-rata umur lebih muda yaitu sekitar 49 tahun.

    heatmap_df = df[["age", "trestbps", "fbs", "thalch", "num"]].copy()
    heatmap_df["fbs"] = heatmap_df["fbs"].map({True: 1, False: 0})
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        heatmap_df.corr(numeric_only=True), annot=True, cmap="coolwarm", fmt=".2f"
    )
    plt.title("Heatmap Korelasi Age, Trestbps, Fbs, Thalch, dan Num")
    save_and_show_plot("heatmap_age_trestbps_fbs_thalch_num.png", output_dir)
    # Analisis: age berkorelasi positif dengan num sekitar 0.34, thalch berkorelasi
    # negatif dengan num sekitar -0.37, dan age juga berkorelasi negatif dengan
    # thalch sekitar -0.37.

    plt.figure(figsize=(8, 5))
    sns.countplot(data=df, x="sex", hue="fbs")
    plt.title("Hubungan Sex dan Fbs")
    plt.xlabel("Sex")
    plt.ylabel("Jumlah")
    save_and_show_plot("relationship_sex_fbs.png", output_dir)
    # Analisis: fbs=True lebih banyak ditemukan pada Male sebanyak 119 baris
    # dibanding Female sebanyak 19 baris, tetapi pembacaan perlu hati-hati karena
    # jumlah pasien Male jauh lebih besar.

    plt.figure(figsize=(8, 5))
    sns.scatterplot(data=df, x="age", y="thalch", hue="sex", alpha=0.75)
    plt.title("Hubungan Age dan Thalch")
    plt.xlabel("Age")
    plt.ylabel("Thalch")
    save_and_show_plot("relationship_age_thalch.png", output_dir)
    # Analisis: thalch cenderung menurun saat age meningkat, sesuai korelasi negatif
    # age dan thalch sekitar -0.37 pada heatmap.


def plot_num_distribution(
    df: pd.DataFrame, target_column: str, output_dir: str
) -> None:
    health_df = df.assign(
        health_status=np.where(df[target_column] > 0, "Sakit", "Sehat")
    )

    plt.figure(figsize=(7, 5))
    sns.countplot(data=health_df, x="health_status", order=["Sehat", "Sakit"])
    plt.title("Distribusi Orang Sehat dan Sakit Berdasarkan Num")
    plt.xlabel("Status")
    plt.ylabel("Jumlah")
    save_and_show_plot("num_healthy_sick_distribution.png", output_dir)
    # Analisis: kelompok Sakit lebih banyak daripada Sehat, yaitu 509 orang atau
    # 55.33% dibanding 411 orang atau 44.67%.

    plt.figure(figsize=(8, 5))
    sns.countplot(
        data=df, x=target_column, order=sorted(df[target_column].dropna().unique())
    )
    plt.title("Distribusi Stage Penyakit Berdasarkan Num")
    plt.xlabel("Stage Num")
    plt.ylabel("Jumlah")
    save_and_show_plot("num_stage_distribution.png", output_dir)
    # Analisis: stage num 0 adalah kategori terbanyak, diikuti stage 1. Stage 4 adalah
    # kategori paling sedikit sehingga distribusi stage tidak seimbang.


def run_eda(df: pd.DataFrame, target_column: str, output_dir: str) -> None:
    sns.set_theme(style="whitegrid")

    print("# Exploratory Data Analysis")
    check_dataset(df)
    print_num_analysis(df, target_column)

    plot_all_column_distributions(df, output_dir)
    plot_relationships(df, output_dir)
    plot_num_distribution(df, target_column, output_dir)


def clean_raw_data(
    df: pd.DataFrame,
    target_column: str,
    drop_columns: list[str],
    numeric_features: list[str],
    bool_features: list[str],
) -> pd.DataFrame:
    df = df.copy()

    df = df.drop(columns=drop_columns, errors="ignore")

    for col in bool_features:
        if col in df.columns:
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

    for col in numeric_features:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df[target_column] = pd.to_numeric(df[target_column], errors="coerce")

    df = df.drop_duplicates()
    df = df.dropna(subset=[target_column])

    df[target_column] = (df[target_column] > 0).astype(int)

    return df


def build_preprocessor(
    numeric_features: list[str],
    string_categorical_features: list[str],
    bool_features: list[str],
    random_state: int,
) -> ColumnTransformer:
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
                        random_state=random_state,
                        n_jobs=-1,
                    ),
                    max_iter=20,
                    tol=0.001,
                    initial_strategy="median",
                    imputation_order="ascending",
                    random_state=random_state,
                ),
            ),
            ("scaler", StandardScaler()),
        ]
    )

    string_categorical_pipeline = Pipeline(
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
            ("num", numeric_pipeline, numeric_features),
            ("cat", string_categorical_pipeline, string_categorical_features),
            ("bool", bool_pipeline, bool_features),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    return preprocessor


def preprocess_data(
    df: pd.DataFrame,
    target_column: str,
    drop_columns: list[str],
    numeric_features: list[str],
    string_categorical_features: list[str],
    bool_features: list[str],
    test_size: float,
    random_state: int,
    output_dir: str,
) -> None:
    data = clean_raw_data(
        df, target_column, drop_columns, numeric_features, bool_features
    )

    X = data.drop(columns=[target_column])
    y = data[target_column].astype(int)

    print("# Dataset setelah preprocessing awal")
    check_dataset(data)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    preprocessor = build_preprocessor(
        numeric_features,
        string_categorical_features,
        bool_features,
        random_state,
    )
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    feature_names = preprocessor.get_feature_names_out()

    X_train_df = pd.DataFrame(X_train_processed, columns=feature_names)
    X_test_df = pd.DataFrame(X_test_processed, columns=feature_names)
    y_train_df = pd.DataFrame({target_column: y_train.reset_index(drop=True)})
    y_test_df = pd.DataFrame({target_column: y_test.reset_index(drop=True)})

    print("# Dataset X_train setelah preprocessing pipeline")
    check_dataset(X_train_df)

    os.makedirs(output_dir, exist_ok=True)
    X_train_df.to_csv(os.path.join(output_dir, "X_train.csv"), index=False)
    X_test_df.to_csv(os.path.join(output_dir, "X_test.csv"), index=False)
    y_train_df.to_csv(os.path.join(output_dir, "y_train.csv"), index=False)
    y_test_df.to_csv(os.path.join(output_dir, "y_test.csv"), index=False)

    joblib.dump(preprocessor, os.path.join(output_dir, "preprocessor.joblib"))


def main() -> None:
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    current_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.getenv("DATASET_PATH", parent_dir)
    kaggle_dataset = os.getenv("KAGGLE_DATASET", "redwankarimsony/heart-disease-data")
    kaggle_csv_filename = os.getenv("KAGGLE_CSV_FILENAME", "heart_disease_uci.csv")
    eda_output_dir = os.getenv("EDA_OUTPUT_DIR", os.path.join(current_dir, "eda_outputs"))
    output_dir = os.getenv(
        "OUTPUT_DIR", os.path.join(current_dir, "heartdisease_preprocessing")
    )

    random_state = 42
    target_column = "num"
    drop_columns = ["id", "dataset"]

    string_categorical_features = ["sex", "cp", "restecg", "slope", "thal"]
    numeric_features = ["age", "trestbps", "chol", "thalch", "oldpeak", "ca"]
    bool_features = ["fbs", "exang"]
    test_size = 0.2

    csv_path = download_dataset_from_kagglehub(
        kaggle_dataset,
        kaggle_csv_filename,
        dataset_path,
    )
    df = load_dataset(csv_path)
    run_eda(df, target_column, eda_output_dir)
    preprocess_data(
        df,
        target_column,
        drop_columns,
        numeric_features,
        string_categorical_features,
        bool_features,
        test_size,
        random_state,
        output_dir,
    )


if __name__ == "__main__":
    main()
