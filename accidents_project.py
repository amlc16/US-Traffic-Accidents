from pathlib import Path
import time
import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    classification_report,
    confusion_matrix,
    mutual_info_score
)
from sklearn.utils import resample

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.utils.class_weight import compute_class_weight

# ============================================================
# 1. CONFIG
# ============================================================

CSV_PATH = "data/US_Accidents_March23.csv"

MAX_ROWS_FOR_PREPROCESS = None
MAX_ROWS_FOR_CLUSTERING = 1_000 #1_000_000
MAX_ROWS_FOR_DBSCAN = 1_000 #100_000
MAX_ROWS_FOR_CLASSIFICATION = None

RANDOM_STATE = 42

# ============================================================
# UTILITIES
# ============================================================
def log_time(start: float, label: str):
    elapsed = time.perf_counter() - start
    print(f"{label} time: {elapsed:.2f} seconds")
    return elapsed


def compute_entropy(counts: np.ndarray, base: float = 2.0) -> float:
    """Entropy H(X) from counts."""
    probs = counts / counts.sum()
    probs = probs[probs > 0]
    return float(-(probs * np.log(probs) / np.log(base)).sum())


def compute_mi_conditional_entropy(y_true: np.ndarray, y_cluster: np.ndarray):
    """
    Compute:
    - Mutual information I(Severity; Cluster)
    - H(Severity), H(Severity | Cluster)
    from severity labels and cluster labels.
    """
    # Mutual information
    mi = mutual_info_score(y_true, y_cluster)

    # Joint contingency table: rows = severity, cols = cluster
    contingency = pd.crosstab(y_true, y_cluster)
    joint_counts = contingency.values

    # Marginals
    severity_counts = joint_counts.sum(axis=1)
    cluster_counts = joint_counts.sum(axis=0)

    # Entropies
    H_severity = compute_entropy(severity_counts)
    H_cluster = compute_entropy(cluster_counts)
    H_joint = compute_entropy(joint_counts.ravel())

    # H(S | C) = H(S, C) - H(C)
    H_severity_given_cluster = H_joint - H_cluster

    return mi, H_severity, H_severity_given_cluster

def balance_training_data(X_train, y_train, random_state=42):
    """
    Create a rebalanced training set:
    - Oversample minority classes (1, 3, 4)
    - Downsample majority class (2)
    Returns new X_train_bal, y_train_bal.
    """
    df_train = X_train.copy()
    df_train["Severity"] = y_train.values

    # Split by class
    dfs = {c: df_train[df_train["Severity"] == c] for c in sorted(df_train["Severity"].unique())}

    # Choose a target size for minority classes
    minority_sizes = [len(dfs[c]) for c in dfs if c != 2]
    target_minority = int(max(minority_sizes) * 0.5)

    balanced_parts = []

    for c, df_c in dfs.items():
        if c == 2:
            # downsample majority to 1.5 * target_minority
            n_samples = min(len(df_c), int(1.5 * target_minority))
            df_c_bal = resample(
                df_c, replace=False, n_samples=n_samples, random_state=random_state
            )
        else:
            # oversample minority up to target_minority
            df_c_bal = resample(
                df_c, replace=True, n_samples=target_minority, random_state=random_state
            )
        balanced_parts.append(df_c_bal)

    df_bal = pd.concat(balanced_parts).sample(frac=1, random_state=random_state)
    X_bal = df_bal.drop(columns=["Severity"])
    y_bal = df_bal["Severity"]

    print("Balanced training class counts:")
    print(y_bal.value_counts())

    return X_bal, y_bal


def plot_severity_distribution(df: pd.DataFrame, out_path: Path):
    """
    Plot severity distribution as a bar chart and save to out_path.
    """
    
    # plot severity distribution
    sev_counts = df["Severity"].value_counts().sort_index()

    plt.figure()
    sev_counts.plot(kind="bar")
    plt.xlabel("Severity")
    plt.ylabel("Count")
    plt.title("Severity Distribution")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"Saved severity distribution plot to {out_path}")

def plot_cluster_sizes(labels: np.ndarray, out_path: Path, title: str):
    """
    Optional helper to plot cluster sizes if matplotlib is installed.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping cluster size plot.")
        return

    unique, counts = np.unique(labels, return_counts=True)
    # For DBSCAN, cluster -1 is noise
    order = np.argsort(unique)
    unique = unique[order]
    counts = counts[order]

    plt.figure()
    plt.bar(unique.astype(str), counts)
    plt.xlabel("Cluster Label")
    plt.ylabel("Count")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"Saved cluster size plot to {out_path}")


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
    start = time.perf_counter()
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

    print(f"Preprocessing done. Rows after cleaning: {len(df):,}\n")
    log_time(start, "Preprocessing")
    return df


# ============================================================
# DATA DESCRIPTION HELPERS
# ============================================================
def describe_data(df: pd.DataFrame):
    """
    Print some basic statistics for the Data section.
    """
    print("\n=== HEAD (first 5 rows) ===")
    print(df.head())

    print("\n=== Numeric describe ===")
    print(df[["Temperature", "Humidity", "Visibility", "Wind_Speed", "Precipitation"]].describe())

    print("\n=== Severity distribution ===")
    print(df["Severity"].value_counts().sort_index())

    print("\n=== Severity distribution (normalized) ===")
    print(df["Severity"].value_counts(normalize=True).sort_index())

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
    # start timer
    start = time.perf_counter()

    X = prepare_clustering_matrix(df)

    # Sample if needed
    if max_rows is not None and len(X) > max_rows:
        X = X.sample(n=max_rows, random_state=random_state)

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Run K-means
    print(f"Running KMeans with k={n_clusters} on {X.shape[0]:,} points...")
    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        n_init="auto",
    )
    # Fit and predict
    labels = kmeans.fit_predict(X_scaled)
    

    # Evaluate with internal metrics
    sil = silhouette_score(X_scaled, labels)
    db = davies_bouldin_score(X_scaled, labels)
    

    # Attach labels back to df for cluster interpretation
    clustered_df = df.loc[X.index].copy()
    clustered_df["cluster_kmeans"] = labels

    # External metrics: MI & conditional entropy
    mi, H_sev, H_sev_given_cluster = compute_mi_conditional_entropy(
        clustered_df["Severity"].values,
        clustered_df["cluster_kmeans"].values,
    )
    kmeans_time = time.perf_counter() - start

    print(f"KMeans total time: {kmeans_time:.2f} seconds \n")
    print(f"KMeans silhouette score: {sil:.4f}")
    print(f"KMeans Davies-Bouldin index: {db:.4f}")
    print(f"KMeans mutual information I(Severity; Cluster): {mi:.4f} bits")
    print(f"H(Severity): {H_sev:.4f} bits")
    print(f"H(Severity | Cluster): {H_sev_given_cluster:.4f} bits")

    plot_cluster_sizes(
        labels,
        "kmeans_cluster_sizes.png",
        title=f"KMeans Cluster Sizes (k={n_clusters})",
    )

    return kmeans, scaler, clustered_df, {
        "time": kmeans_time,
        "silhouette": sil,
        "davies_bouldin": db,
        "mi": mi,
        "H_severity": H_sev,
        "H_severity_given_cluster": H_sev_given_cluster,
    }


def run_dbscan_clustering(
    df: pd.DataFrame,
    eps: float = 0.7,
    min_samples: int = 200,
    max_rows: int | None = None,
    random_state: int = 42,
):
    """
    Run DBSCAN on a subset of df and evaluate.
    """
    start = time.perf_counter()

    X = prepare_clustering_matrix(df)

    # Sample if needed
    if max_rows is not None and len(X) > max_rows:
        X = X.sample(n=max_rows, random_state=random_state)

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Run DBSCAN
    print(f"\nRunning DBSCAN on {X.shape[0]:,} points...")
    dbscan = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1)
    labels = dbscan.fit_predict(X_scaled)

    dbscan_time = log_time(start, "DBSCAN")

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

    # External metrics: MI & conditional entropy (ignore noise or keep? -> keep)
    mi, H_sev, H_sev_given_cluster = compute_mi_conditional_entropy(
        clustered_df["Severity"].values,
        clustered_df["cluster_dbscan"].values,
    )
    print(f"DBSCAN mutual information I(Severity; Cluster): {mi:.4f} bits")
    print(f"H(Severity): {H_sev:.4f} bits")
    print(f"H(Severity | Cluster): {H_sev_given_cluster:.4f} bits")

    plot_cluster_sizes(
        labels,
        "dbscan_cluster_sizes.png",
        title="DBSCAN Cluster Sizes (including noise)",
    )

    return dbscan, scaler, clustered_df, {
        "time": dbscan_time,
        "silhouette": sil,
        "davies_bouldin": db,
        "mi": mi,
        "H_severity": H_sev,
        "H_severity_given_cluster": H_sev_given_cluster,
    }


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
# 4. CLASSIFICATION (Decision Tree and Neural Networks baseline)
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

def split_for_classification(
    df: pd.DataFrame,
    max_rows: int | None = None,
    random_state: int = 42,
):
    X, y = prepare_classification_matrix(df)

    if max_rows is not None and len(X) > max_rows:
        X, _, y, _ = train_test_split(
            X, y, train_size=max_rows,
            stratify=y,
            random_state=random_state,
        )

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
        test_size=0.1765,  # 0.1765 * 0.85 ≈ 0.15
        stratify=y_temp,
        random_state=random_state,
    )

    return X_train, y_train, X_val, y_val, X_test, y_test


def run_decision_tree_classification(
    df: pd.DataFrame,
    max_rows: int | None = None,
    random_state: int = 42,
):
    """
    Train and evaluate a Decision Tree classifier to predict Severity.
    Uses class_weight='balanced' to handle class imbalance.
    """

    X_train, y_train, X_val, y_val, X_test, y_test = split_for_classification(
        df,
        max_rows=max_rows,
        random_state=random_state,
    )
    

    # Compute class weights manually to handle imbalance
    classes = np.unique(y_train)
    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=y_train,
    )
    class_weight_dict = {c: w for c, w in zip(classes, class_weights)}
    print("\nDecision Tree class weights:", class_weight_dict)

    # Initialize Decision Tree classifier
    clf = DecisionTreeClassifier(
        max_depth=15,
        min_samples_leaf=200,
        class_weight=class_weight_dict,
        random_state=random_state,
    )

    start = time.perf_counter()
    print("\nTraining Decision Tree...")
    clf.fit(X_train, y_train)
    dt_time = log_time(start, "Decision Tree training")

    print("\nDecision Tree – Validation performance:")
    y_val_pred = clf.predict(X_val)
    print(classification_report(y_val, y_val_pred, digits=4))
    print("Confusion matrix (val):")
    print(confusion_matrix(y_val, y_val_pred))

    print("\nDecision Tree – Test performance:")
    y_test_pred = clf.predict(X_test)
    print(classification_report(y_test, y_test_pred, digits=4))
    print("Confusion matrix (test):")
    print(confusion_matrix(y_test, y_test_pred))


    from sklearn.metrics import accuracy_score, f1_score

    val_acc = accuracy_score(y_val, y_val_pred)
    val_f1_macro = f1_score(y_val, y_val_pred, average="macro")
    test_acc = accuracy_score(y_test, y_test_pred)
    test_f1_macro = f1_score(y_test, y_test_pred, average="macro")

    metrics = {
        "time": dt_time,
        "val_accuracy": val_acc,
        "val_f1_macro": val_f1_macro,
        "test_accuracy": test_acc,
        "test_f1_macro": test_f1_macro,
    }

    print("Decision Tree metrics:", metrics)
    return clf, metrics

#unbalanced version
def run_mlp_classification(
    df: pd.DataFrame,
    max_rows: int | None = None,
    random_state: int = 42,
):
    """
    Train and evaluate an MLP (neural network) classifier to predict Severity.
    Uses StandardScaler and class_weight='balanced'.
    """
    X_train, y_train, X_val, y_val, X_test, y_test = split_for_classification(
        df,
        max_rows=max_rows,
        random_state=random_state,
    )

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    classes = np.unique(y_train)
    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=y_train,
    )
    class_weight_dict = {c: w for c, w in zip(classes, class_weights)}
    print("\nMLP class weights:", class_weight_dict)

    mlp = MLPClassifier(
        hidden_layer_sizes=(64, 32),
        activation="relu",
        solver="adam",
        alpha=1e-4,
        batch_size=256,
        learning_rate="adaptive",
        max_iter=60,
        random_state=random_state,
        verbose=True,
    )

    start = time.perf_counter()
    print("\nTraining MLP...")
    mlp.fit(X_train_scaled, y_train)
    mlp_time = log_time(start, "MLP training")

    from sklearn.metrics import accuracy_score, f1_score

    print("\nMLP – Validation performance:")
    y_val_pred = mlp.predict(X_val_scaled)
    print(classification_report(y_val, y_val_pred, digits=4))
    print("Confusion matrix (val):")
    print(confusion_matrix(y_val, y_val_pred))

    print("\nMLP – Test performance:")
    y_test_pred = mlp.predict(X_test_scaled)
    print(classification_report(y_test, y_test_pred, digits=4))
    print("Confusion matrix (test):")
    print(confusion_matrix(y_test, y_test_pred))

    val_acc = accuracy_score(y_val, y_val_pred)
    val_f1_macro = f1_score(y_val, y_val_pred, average="macro")
    test_acc = accuracy_score(y_test, y_test_pred)
    test_f1_macro = f1_score(y_test, y_test_pred, average="macro")

    metrics = {
        "time": mlp_time,
        "val_accuracy": val_acc,
        "val_f1_macro": val_f1_macro,
        "test_accuracy": test_acc,
        "test_f1_macro": test_f1_macro,
    }

    print("MLP metrics:", metrics)
    return mlp, scaler, metrics


#balanced version
# def run_mlp_classification(
#     df: pd.DataFrame,
#     max_rows: int | None = None,
#     random_state: int = 42,
# ):
#     """
#     Train and evaluate an MLP (neural network) classifier to predict Severity.
#     Uses StandardScaler and a rebalanced training set.
#     """
#     X_train, y_train, X_val, y_val, X_test, y_test = split_for_classification(
#         df,
#         max_rows=max_rows,
#         random_state=random_state,
#     )

#     # Rebalance training data
#     X_train_bal, y_train_bal = balance_training_data(X_train, y_train, random_state=random_state)

#     # Scale features (fit on balanced train only)
#     scaler = StandardScaler()
#     X_train_scaled = scaler.fit_transform(X_train_bal)
#     X_val_scaled = scaler.transform(X_val)
#     X_test_scaled = scaler.transform(X_test)

#     mlp = MLPClassifier(
#         hidden_layer_sizes=(64, 32),
#         activation="relu",
#         solver="adam",
#         alpha=1e-4,
#         batch_size=256,
#         learning_rate="adaptive",
#         max_iter=60,
#         random_state=random_state,
#         verbose=True,
#     )

#     start = time.perf_counter()
#     print("\nTraining MLP (balanced data)...")
#     mlp.fit(X_train_scaled, y_train_bal)
#     mlp_time = log_time(start, "MLP training")

#     from sklearn.metrics import accuracy_score, f1_score

#     print("\nMLP – Validation performance:")
#     y_val_pred = mlp.predict(X_val_scaled)
#     print(classification_report(y_val, y_val_pred, digits=4))
#     print("Confusion matrix (val):")
#     print(confusion_matrix(y_val, y_val_pred))

#     print("\nMLP – Test performance:")
#     y_test_pred = mlp.predict(X_test_scaled)
#     print(classification_report(y_test, y_test_pred, digits=4))
#     print("Confusion matrix (test):")
#     print(confusion_matrix(y_test, y_test_pred))

#     val_acc = accuracy_score(y_val, y_val_pred)
#     val_f1_macro = f1_score(y_val, y_val_pred, average="macro")
#     test_acc = accuracy_score(y_test, y_test_pred)
#     test_f1_macro = f1_score(y_test, y_test_pred, average="macro")

#     metrics = {
#         "time": mlp_time,
#         "val_accuracy": val_acc,
#         "val_f1_macro": val_f1_macro,
#         "test_accuracy": test_acc,
#         "test_f1_macro": test_f1_macro,
#     }

#     print("MLP metrics:", metrics)
#     return mlp, scaler, metrics


# ============================================================
# 5. MAIN SCRIPT
# ============================================================

if __name__ == "__main__":
    # 1) Preprocess

    df_all = base_preprocess(
        CSV_PATH,
        max_rows=MAX_ROWS_FOR_PREPROCESS,
    )

    # Basic data description
    describe_data(df_all)
    plot_severity_distribution(df_all, "severity_distribution.png")

    

    # 2) K-means clustering
    kmeans, kmeans_scaler, kmeans_df, kmeans_metrics = run_kmeans_clustering(
        df_all,
        n_clusters=5,  # variable to tune
        max_rows=MAX_ROWS_FOR_CLUSTERING,
        random_state=RANDOM_STATE,
    )
    summarize_clusters_by_severity(kmeans_df, "cluster_kmeans")


    # 3) DBSCAN clustering
    dbscan, dbscan_scaler, dbscan_df, dbscan_metrics = run_dbscan_clustering(
        df_all,
        eps=0.7,            # variable to tune
        min_samples=200,    # variable to tune
        max_rows=MAX_ROWS_FOR_DBSCAN,
        random_state=RANDOM_STATE,
    )
    summarize_clusters_by_severity(dbscan_df, "cluster_dbscan")

    # 4) Classification (Decision Tree baseline)
    dtree, dt_metrics = run_decision_tree_classification(
        df_all,
        max_rows=MAX_ROWS_FOR_CLASSIFICATION,
        random_state=RANDOM_STATE,
    )
    # 5) Classification (MLP baseline)
    mlp, mlp_scaler, mlp_metrics = run_mlp_classification(
        df_all,
        max_rows=MAX_ROWS_FOR_CLASSIFICATION,
        random_state=RANDOM_STATE,
    )
    print("\n=== SUMMARY OF KEY METRICS ===")
    print("KMeans:", kmeans_metrics)
    print("DBSCAN:", dbscan_metrics)
    print("Decision Tree:", dt_metrics)
    print("MLP:", mlp_metrics)

    print("\nPipeline finished.")
