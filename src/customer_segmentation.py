"""Reproducible RFM customer-segmentation pipeline.

This script reproduces the final model selected in the original academic project:
K-Means clustering on Recency, Frequency and Monetary (RFM) features with K=4.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, davies_bouldin_score, silhouette_score
from sklearn.preprocessing import MinMaxScaler


REFERENCE_DATE = pd.Timestamp("2024-12-31")
SEGMENT_NAMES = {
    0: "High-Value Repeat Customers",
    1: "Frequent Low-Spend Customers",
    2: "Lapsed High-Spend Customers",
    3: "Recent Occasional Customers",
}


def load_data(path: Path) -> pd.DataFrame:
    """Load the source CSV."""
    return pd.read_csv(path)


def build_rfm(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create raw and min-max-scaled RFM features."""
    monetary = (
        df["Purchase_Amount"]
        .astype(str)
        .str.replace("$", "", regex=False)
        .str.replace(",", "", regex=False)
        .str.strip()
        .astype(float)
    )

    purchase_date = pd.to_datetime(
        df["Time_of_Purchase"],
        format="mixed",
        errors="raise",
    )

    raw = pd.DataFrame(
        {
            "RECENCY": (REFERENCE_DATE - purchase_date).dt.days.clip(lower=0),
            "FREQUENCY": df["Frequency_of_Purchase"].astype(float),
            "MONETARY": monetary,
        },
        index=df.index,
    )

    scaler = MinMaxScaler()
    scaled = pd.DataFrame(
        scaler.fit_transform(raw),
        columns=raw.columns,
        index=raw.index,
    )
    return raw, scaled


def evaluate_k_range(x: pd.DataFrame, k_min: int = 2, k_max: int = 8) -> pd.DataFrame:
    """Evaluate candidate K values using common internal clustering metrics."""
    rows: list[dict[str, float]] = []
    for k in range(k_min, k_max + 1):
        model = KMeans(n_clusters=k, random_state=42, n_init=50)
        labels = model.fit_predict(x)
        rows.append(
            {
                "k": k,
                "inertia": float(model.inertia_),
                "silhouette": float(silhouette_score(x, labels)),
                "davies_bouldin": float(davies_bouldin_score(x, labels)),
            }
        )
    return pd.DataFrame(rows)


def fit_final_model(x: pd.DataFrame) -> tuple[KMeans, np.ndarray]:
    """Fit the final four-cluster RFM model."""
    model = KMeans(n_clusters=4, random_state=42, n_init=50)
    labels = model.fit_predict(x)
    return model, labels


def stability_ari(x: pd.DataFrame, seeds=range(5)) -> tuple[float, float]:
    """Estimate solution stability across random seeds using pairwise ARI."""
    labelings = [
        KMeans(n_clusters=4, random_state=seed, n_init=20).fit_predict(x)
        for seed in seeds
    ]
    aris = [
        adjusted_rand_score(labelings[i], labelings[j])
        for i in range(len(labelings))
        for j in range(i + 1, len(labelings))
    ]
    return float(np.mean(aris)), float(np.std(aris))


def posthoc_external_checks(df: pd.DataFrame, labels: np.ndarray) -> dict[str, float]:
    """Check whether clusters are associated with held-out descriptive variables.

    These variables are used only after clustering and do not influence cluster formation.
    """
    evaluation = pd.DataFrame(
        {
            "cluster": labels,
            "Purchase_Intent": df["Purchase_Intent"],
            "Customer_Satisfaction": df["Customer_Satisfaction"],
        }
    )

    p_intent = chi2_contingency(
        pd.crosstab(evaluation["cluster"], evaluation["Purchase_Intent"])
    )[1]
    p_satisfaction = chi2_contingency(
        pd.crosstab(evaluation["cluster"], evaluation["Customer_Satisfaction"])
    )[1]

    return {
        "chi2_p_purchase_intent": float(p_intent),
        "chi2_p_customer_satisfaction": float(p_satisfaction),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data/ecommerce_consumer_behavior.csv"),
        help="Path to the source CSV.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results"),
        help="Directory for generated CSV results.",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    df = load_data(args.data)
    raw_rfm, scaled_rfm = build_rfm(df)

    k_metrics = evaluate_k_range(scaled_rfm)
    model, labels = fit_final_model(scaled_rfm)

    ari_mean, ari_std = stability_ari(scaled_rfm)
    external = posthoc_external_checks(df, labels)

    final_metrics = pd.DataFrame(
        [
            {
                "model": "KMeans — RFM only (final)",
                "k": 4,
                "inertia": float(model.inertia_),
                "silhouette": float(silhouette_score(scaled_rfm, labels)),
                "davies_bouldin": float(davies_bouldin_score(scaled_rfm, labels)),
                "ari_mean": ari_mean,
                "ari_std": ari_std,
                **external,
            }
        ]
    )

    profile = raw_rfm.assign(cluster=labels).groupby("cluster").mean().reset_index()
    profile.insert(1, "segment", profile["cluster"].map(SEGMENT_NAMES))

    k_metrics.to_csv(args.output_dir / "k_selection_metrics.csv", index=False)
    final_metrics.to_csv(args.output_dir / "final_model_metrics.csv", index=False)
    profile.to_csv(args.output_dir / "rfm_cluster_profile.csv", index=False)

    print(final_metrics.to_string(index=False))
    print()
    print(profile.to_string(index=False))


if __name__ == "__main__":
    main()
