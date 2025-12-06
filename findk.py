"""
Sweep different k values for K-Means and report:
- runtime
- silhouette score
- Davies–Bouldin index
- mutual information I(Severity; Cluster)
- conditional entropy H(Severity | Cluster)
- per-cluster severity distributions

Requires accidents_project.py in the same directory.
"""

from accidents_project import (
    base_preprocess,
    run_kmeans_clustering,
    summarize_clusters_by_severity,
    CSV_PATH,
    RANDOM_STATE,
)

# ------------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------------

# k values to test
KS = [3, 4, 5, 6]

# How many rows to use for the sweep (subset for speed)
MAX_ROWS_FOR_PREPROCESS = None      # None = preprocess full dataset
MAX_ROWS_FOR_KMEANS_SWEEP = 300_000 # subset for KMeans sweep


# ------------------------------------------------------------------
# MAIN
# ------------------------------------------------------------------

def main():
    # 1) Preprocess data (you can also reuse an already-preprocessed subset)
    df_all = base_preprocess(
        CSV_PATH,
        max_rows=MAX_ROWS_FOR_PREPROCESS,
    )

    # 2) Sweep over k
    results = []
    for k in KS:
        print(f"\n==============================")
        print(f"=== KMeans sweep: k = {k} ===")
        print(f"==============================")

        kmeans, scaler, clustered_df, metrics = run_kmeans_clustering(
            df_all,
            n_clusters=k,
            max_rows=MAX_ROWS_FOR_KMEANS_SWEEP,
            random_state=RANDOM_STATE,
        )

        # Per-cluster severity distribution (for interpretation)
        summarize_clusters_by_severity(clustered_df, "cluster_kmeans")

        metrics["k"] = k
        results.append(metrics)

    # 3) Print summary line for each k (nice for the report)
    print("\n=== KMeans sweep summary ===")
    for r in results:
        print(
            f"k={r['k']}: "
            f"time={r['time']:.1f}s, "
            f"silhouette={r['silhouette']:.3f}, "
            f"Davies-Bouldin={r['davies_bouldin']:.3f}, "
            f"MI={r['mi']:.3f}, "
            f"H(S|C)={r['H_severity_given_cluster']:.3f}"
        )


if __name__ == "__main__":
    main()