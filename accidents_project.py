# accidents_project.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.class_weight import compute_class_weight


# ============================================================
# 1. CONFIG
# ============================================================

CSV_PATH = "data/US_Accidents_March23.csv"

# To avoid killing your laptop, you can optionally subsample
MAX_ROWS_FOR_PREPROCESS = None  # e.g., 2_000_000 or None for full
MAX_ROWS_FOR_CLUSTERING = None
MAX_ROWS_FOR_DBSCAN = None
MAX_ROWS_FOR_CLASSIFICATION = None

RANDOM_STATE = 42


# ============================================================
# 2. LOADING + BASE PREPROCESSING
# ============================================================

def load_data(path: str, max_rows: int | None = None) -> pd.DataFrame:
    """
    Load only the columns we need for clustering and classification.
    """
    usecols = [
        "ID",
        "Severity",
        "Start_Time",
        "State",
        # Weather
        "Temperature(F)",
        "Humidity(%)",
        "Visibility(mi)",
        "Wind_Speed(mph)",
        "Precipitation(in)",
        # POI
        "Amenity",
        "Bump",
        "Crossing",
        "Give_Way",
        "Junction",
        "No_Exit",
        "Railway",
        "Roundabout",
        "Station",
        "Stop",
        "Traffic_Calming",
        "Traffic_Signal",
        "Turning_Loop",
        # Twilight
        "Sunrise_Sunset",
        "Civil_Twilight",
        "Nautical_Twilight",
        "Astronomical_Twilight",
    ]

    df = pd.read_csv(
        path,
        usecols=usecols,
        nrows=max_rows,
        low_memory=True,
    )

    # Rename to simpler column names
    df = df.rename(
        columns={
            "Temperature(F)": "Temperature",
            "Humidity(%)": "Humidity",
            "Visibility(mi)": "Visibility",
            "Wind_Speed(mph)": "Wind_Speed",
            "Precipitation(in)": "Precipitation",
        }
    )

    return df


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert Start_Time to datetime and extract Hour and Month.
    """
    df = df.copy()
    df["Start_Time"] = pd.to_datetime(df["Start_Time"], errors="coerce")
    df = df.dropna(subset=["Start_Time"])

    df["Hour"] = df["Start_Time"].dt.hour
    df["Month"] = df["Start_Time"].dt.month

    return df


def encode_twilight(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode twilight features as binary 1=Day, 0=Night.
    Missing values are imputed with column mode.
    """
    df = df.copy()
    twilight_cols = [
        "Sunrise_Sunset",
        "Civil_Twilight",
        "Nautical_Twilight",
        "Astronomical_Twilight",
    ]

    for col in twilight_cols:
        # Fill missing with mode first
        if df[col].isna().any():
            mode_val = df[col].mode(dropna=True)
            if len(mode_val) > 0:
                df[col] = df[col].fillna(mode_val.iloc[0])

        df[col] = df[col].map({"Day": 1, "Night": 0})

        # If there are unexpected values, fill with column median
        df[col] = df[col].fillna(df[col].median())

    return df


def impute_weather(df: pd.DataFrame) -> pd.DataFrame:
    """
    Impute missing weather features using group-wise median by (State, Month, Hour),
    then fall back to global median.
    """
    df = df.copy()

    weather_cols = ["Temperature", "Humidity", "Visibility", "Wind_Speed", "Precipitation"]
    group_cols = ["State", "Month", "Hour"]

    for col in weather_cols:
        # group-wise median
        group_medians = df.groupby(group_cols)[col].transform("median")
        df[col] = df[col].fillna(group_medians)

        # global median fallback
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())

    return df


def cast_bool_to_int(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert POI boolean features to int8 (0/1).
    """
    df = df.copy()
    poi_cols = [
        "Amenity",
        "Bump",
        "Crossing",
        "Give_Way",
        "Junction",
        "No_Exit",
        "Railway",
        "Roundabout",
        "Station",
        "Stop",
        "Traffic_Calming",
        "Traffic_Signal",
        "Turning_Loop",
    ]

    for col in poi_cols:
        df[col] = df[col].astype("int8")

    return df


def base_preprocess(path: str, max_rows: int | None = None) -> pd.DataFrame:
    """
    Full preprocessing pipeline:
    - Load subset of columns
    - Add temporal features
    - Encode twilight
    - Impute weather
    - Cast booleans to ints
    """
    print("Loading data...")
    df = load_data(path, max_rows=max_rows)
    print(f"Loaded {len(df):,} rows.")

    print("Adding time features (Hour, Month)...")
    df = add_time_features(df)

    print("Encoding twilight features...")
    df = encode_twilight(df)

    print("Imputing missing weather values...")
    df = impute_weather(df)

    print("Casting POI booleans to ints...")
    df = cast_bool_to_int(df)

    # Keep only rows with non-null Severity (should be all)
    df = df.dropna(subset=["Severity"])

    print(f"Preprocessing done. Rows after cleaning: {len(df):,}")
    return df


# ============================================================
# 3. CLUSTERING (K-means + DBSCAN)
# ============================================================

def prepare_clustering_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Select features for clustering (weather + Hour + twilight).
    """
    cluster_features = [
        "Temperature",
        "Humidity",
        "Visibility",
        "Precipitation",
        "Wind_Speed",
        "Hour",
        "Sunrise_Sunset",
        "Civil_Twilight",
        "Nautical_Twilight",
        "Astronomical_Twilight",
    ]
    return df[cluster_features].copy()


def run_kmeans_clustering(
    df: pd.DataFrame,
    n_clusters: int = 5,
    max_rows: int | None = None,
    random_state: int = 42,
):
    """
    Run K-means on a (possibly sampled) subset of df and evaluate.
    """
    X = prepare_clustering_matrix(df)

    if max_rows is not None and len(X) > max_rows:
        X = X.sample(n=max_rows, random_state=random_state)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print(f"Running KMeans with k={n_clusters} on {X.shape[0]:,} points...")
    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        n_init="auto",
    )
    labels = kmeans.fit_predict(X_scaled)

    sil = silhouette_score(X_scaled, labels)
    db = davies_bouldin_score(X_scaled, labels)
    print(f"KMeans silhouette score: {sil:.4f}")
    print(f"KMeans Davies-Bouldin index: {db:.4f}")

    # Attach labels back to df for cluster interpretation
    clustered_df = df.loc[X.index].copy()
    clustered_df["cluster_kmeans"] = labels

    return kmeans, scaler, clustered_df, sil, db


def run_dbscan_clustering(
    df: pd.DataFrame,
    eps: float = 0.7,
    min_samples: int = 200,
    max_rows: int | None = None,
    random_state: int = 42,
):
    """
    Run DBSCAN on a (smaller) subset of df and evaluate.
    """
    X = prepare_clustering_matrix(df)

    if max_rows is not None and len(X) > max_rows:
        X = X.sample(n=max_rows, random_state=random_state)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print(f"Running DBSCAN on {X.shape[0]:,} points...")
    dbscan = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1)
    labels = dbscan.fit_predict(X_scaled)

    # DBSCAN labels: -1 = noise
    unique_labels = np.unique(labels)
    print(f"DBSCAN found clusters: {unique_labels}")

    # Filter noise for internal metrics
    mask = labels != -1
    if mask.sum() > 1 and len(np.unique(labels[mask])) > 1:
        sil = silhouette_score(X_scaled[mask], labels[mask])
        db = davies_bouldin_score(X_scaled[mask], labels[mask])
        print(f"DBSCAN silhouette (non-noise): {sil:.4f}")
        print(f"DBSCAN Davies-Bouldin (non-noise): {db:.4f}")
    else:
        sil = None
        db = None
        print("Not enough non-noise clusters to compute internal metrics.")

    clustered_df = df.loc[X.index].copy()
    clustered_df["cluster_dbscan"] = labels

    return dbscan, scaler, clustered_df, sil, db


def summarize_clusters_by_severity(clustered_df: pd.DataFrame, cluster_col: str):
    """
    For each cluster, show how Severity is distributed.
    """
    print(f"\nSeverity distribution by {cluster_col}:")
    tab = (
        clustered_df
        .groupby(cluster_col)["Severity"]
        .value_counts(normalize=True)
        .unstack(fill_value=0)
        .sort_index()
    )
    print(tab)
    return tab


# ============================================================
# 4. CLASSIFICATION (Decision Tree baseline)
# ============================================================

def prepare_classification_matrix(df: pd.DataFrame):
    """
    Select features and target for severity classification.
    """
    feature_cols = [
        # Weather
        "Temperature",
        "Humidity",
        "Visibility",
        "Precipitation",
        "Wind_Speed",
        # Time
        "Hour",
        # Twilight
        "Sunrise_Sunset",
        "Civil_Twilight",
        "Nautical_Twilight",
        "Astronomical_Twilight",
        # POI
        "Amenity",
        "Bump",
        "Crossing",
        "Give_Way",
        "Junction",
        "No_Exit",
        "Railway",
        "Roundabout",
        "Station",
        "Stop",
        "Traffic_Calming",
        "Traffic_Signal",
        "Turning_Loop",
    ]

    X = df[feature_cols].copy()
    y = df["Severity"].astype(int)

    return X, y


def run_decision_tree_classification(
    df: pd.DataFrame,
    max_rows: int | None = None,
    random_state: int = 42,
):
    """
    Train and evaluate a Decision Tree classifier to predict Severity.
    Uses class_weight='balanced' to handle class imbalance.
    """
    X, y = prepare_classification_matrix(df)

    if max_rows is not None and len(X) > max_rows:
        X, _, y, _ = train_test_split(
            X, y, train_size=max_rows,
            stratify=y,
            random_state=random_state,
        )

    # Train/val/test split: 70/15/15
    X_temp, X_test, y_temp, y_test = train_test_split(
        X,
        y,
        test_size=0.15,
        stratify=y,
        random_state=random_state,
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_temp,
        y_temp,
        test_size=0.1765,  # 0.1765 * 0.85 ≈ 0.15 => 70/15/15
        stratify=y_temp,
        random_state=random_state,
    )

    # Compute class weights manually to be explicit in the report
    classes = np.unique(y_train)
    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=y_train,
    )
    class_weight_dict = {c: w for c, w in zip(classes, class_weights)}
    print("Class weights:", class_weight_dict)

    clf = DecisionTreeClassifier(
        max_depth=15,
        min_samples_leaf=200,
        class_weight=class_weight_dict,
        random_state=random_state,
    )

    print("Training Decision Tree...")
    clf.fit(X_train, y_train)

    print("\nValidation performance:")
    y_val_pred = clf.predict(X_val)
    print(classification_report(y_val, y_val_pred, digits=4))
    print("Confusion matrix (val):")
    print(confusion_matrix(y_val, y_val_pred))

    print("\nTest performance:")
    y_test_pred = clf.predict(X_test)
    print(classification_report(y_test, y_test_pred, digits=4))
    print("Confusion matrix (test):")
    print(confusion_matrix(y_test, y_test_pred))

    return clf, (X_train, y_train, X_val, y_val, X_test, y_test)

# ============================================================
# 4. PLOTTING RESULTS
# ============================================================

def plot_all_results(kmeans_df, clf, X_train, y_test, y_test_pred):
    """Generate all required plots in one function"""
    import os
    os.makedirs('outputs', exist_ok=True)

    # 1. Confusion Matrix
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_test, y_test_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=[1, 2, 3, 4], yticklabels=[1, 2, 3, 4])
    plt.title('Test Confusion Matrix', fontweight='bold')
    plt.ylabel('True Severity')
    plt.xlabel('Predicted Severity')
    plt.savefig('outputs/confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved: outputs/confusion_matrix.png")

    # 2. Feature Importance
    importances = clf.feature_importances_
    indices = np.argsort(importances)[::-1][:15]
    plt.figure(figsize=(10, 6))
    plt.barh(range(15), importances[indices])
    plt.yticks(range(15), [X_train.columns[i] for i in indices])
    plt.xlabel('Importance')
    plt.title('Top 15 Features', fontweight='bold')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig('outputs/feature_importance.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved: outputs/feature_importance.png")

    # 3. Cluster characteristics (simplified - just 3 key features)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for idx, feature in enumerate(['Temperature', 'Visibility', 'Hour']):
        kmeans_df.boxplot(column=feature, by='cluster_kmeans', ax=axes[idx])
        axes[idx].set_title(f'{feature} by Cluster')
    plt.tight_layout()
    plt.savefig('outputs/cluster_chars.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved: outputs/cluster_chars.png")


# ============================================================
# 6. MAIN SCRIPT
# ============================================================

if __name__ == "__main__":
    # 1) Preprocess
    df_all = base_preprocess(
        CSV_PATH,
        max_rows=MAX_ROWS_FOR_PREPROCESS,
    )

    # 2) K-means clustering
    kmeans, kmeans_scaler, kmeans_df, k_sil, k_db = run_kmeans_clustering(
        df_all,
        n_clusters=5,  # you can sweep this for your elbow plot
        max_rows=MAX_ROWS_FOR_CLUSTERING,
        random_state=RANDOM_STATE,
    )
    summarize_clusters_by_severity(kmeans_df, "cluster_kmeans")

    # 3) DBSCAN clustering
    dbscan, dbscan_scaler, dbscan_df, d_sil, d_db = run_dbscan_clustering(
        df_all,
        eps=0.7,            # tune based on your experiments
        min_samples=200,    # tune as needed
        max_rows=MAX_ROWS_FOR_DBSCAN,
        random_state=RANDOM_STATE,
    )
    summarize_clusters_by_severity(dbscan_df, "cluster_dbscan")

    # 4) Classification (Decision Tree baseline)
    clf, splits = run_decision_tree_classification(
        df_all,
        max_rows=MAX_ROWS_FOR_CLASSIFICATION,
        random_state=RANDOM_STATE,
    )

    print("\nPipeline finished.")
