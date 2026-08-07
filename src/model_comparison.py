"""Selected exploratory model comparisons from the original research workflow.

The final portfolio model is RFM + K-Means. This file preserves the main comparison
logic without the duplicated notebook/Colab experimentation.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score


DATA_PATH = Path("data/ecommerce_consumer_behavior.csv")


def _map_ordinal(series: pd.Series, mapping: dict[str, float]) -> pd.Series:
    return (
        series.astype(str)
        .str.strip()
        .str.upper()
        .map(mapping)
        .fillna(0.0)
    )


def prepare_numeric_ordinal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Reproduce the numeric/ordinal K-Means comparison feature set."""
    work = pd.DataFrame(index=df.index)

    work["MONETARY"] = (
        df["Purchase_Amount"]
        .astype(str)
        .str.replace("$", "", regex=False)
        .str.replace(",", "", regex=False)
        .str.strip()
        .astype(float)
    )
    work["FREQUENCY"] = df["Frequency_of_Purchase"].astype(float)

    dates = pd.to_datetime(df["Time_of_Purchase"], format="mixed", errors="raise")
    work["RECENCY"] = (pd.Timestamp("2024-12-31") - dates).dt.days.clip(lower=0)

    work["Research_Time"] = df["Time_Spent_on_Product_Research(hours)"].astype(float)
    work["Return_Rate"] = df["Return_Rate"].astype(float)
    work["Time_to_Decision"] = df["Time_to_Decision"].astype(float)
    work["Discount_Used"] = df["Discount_Used"].astype(float)
    work["Loyalty_Program"] = df["Customer_Loyalty_Program_Member"].astype(float)

    social_map = {"LOW": 0.33, "MEDIUM": 0.66, "HIGH": 1.0}
    discount_map = {
        "NOT SENSITIVE": 0.0,
        "SOMEWHAT SENSITIVE": 0.5,
        "VERY SENSITIVE": 1.0,
    }
    work["Social_Media_Influence"] = _map_ordinal(df["Social_Media_Influence"], social_map)
    work["Engagement_with_Ads"] = _map_ordinal(df["Engagement_with_Ads"], social_map)
    work["Discount_Sensitivity"] = _map_ordinal(df["Discount_Sensitivity"], discount_map)

    scaler = MinMaxScaler()
    return pd.DataFrame(
        scaler.fit_transform(work),
        columns=work.columns,
        index=work.index,
    )


def main() -> None:
    df = pd.read_csv(DATA_PATH)
    x_full = prepare_numeric_ordinal_features(df)

    full_model = KMeans(n_clusters=4, random_state=42, n_init=50)
    full_labels = full_model.fit_predict(x_full)

    rfm = x_full[["RECENCY", "FREQUENCY", "MONETARY"]]
    rfm_model = KMeans(n_clusters=4, random_state=42, n_init=50)
    rfm_labels = rfm_model.fit_predict(rfm)

    # GMM comparison after PCA, following the exploratory project direction.
    pca = PCA(n_components=0.90, random_state=42)
    x_pca = pca.fit_transform(x_full)
    gmm_rows = []
    for k in range(2, 9):
        gmm = GaussianMixture(n_components=k, covariance_type="full", random_state=42)
        gmm.fit(x_pca)
        gmm_rows.append({"k": k, "aic": gmm.aic(x_pca), "bic": gmm.bic(x_pca)})

    comparison = pd.DataFrame(
        [
            {
                "model": "KMeans — numeric/ordinal feature set",
                "k": 4,
                "silhouette": silhouette_score(x_full, full_labels),
                "davies_bouldin": davies_bouldin_score(x_full, full_labels),
            },
            {
                "model": "KMeans — RFM only",
                "k": 4,
                "silhouette": silhouette_score(rfm, rfm_labels),
                "davies_bouldin": davies_bouldin_score(rfm, rfm_labels),
            },
        ]
    )

    print(comparison.to_string(index=False))
    print()
    print("GMM + PCA model-selection table:")
    print(pd.DataFrame(gmm_rows).to_string(index=False))


if __name__ == "__main__":
    main()
