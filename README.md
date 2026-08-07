# E-commerce Customer Segmentation with RFM

Customer segmentation using **RFM (Recency, Frequency, Monetary) analysis** and **K-Means clustering** to identify distinct purchasing-value profiles and translate them into targeted customer strategies.

## Business objective

The goal is to group customers according to purchasing value and timing so that different segments can support decisions around retention, re-engagement, cross-sell, upsell, average order value and repeat-purchase activation.

## Modeling approach

The analysis evaluated multiple representations before selecting the final model:

1. broader behavioral K-Means feature sets;
2. sensitivity analysis around discount and loyalty variables;
3. Isolation Forest as an outlier diagnostic;
4. probabilistic clustering with GMM after PCA;
5. final **RFM-only K-Means** segmentation.

The final model uses RFM because it produced the clearest combination of interpretability, quantitative separation and stability.

For the full modeling path, see [`notebooks/exploratory_modeling.ipynb`](notebooks/exploratory_modeling.ipynb).

## Dataset

The repository uses the public **Ecommerce Consumer Behavior Analysis Data** dataset from Kaggle:

- **1,000 customers**
- **28 variables**
- demographics, purchase behavior, engagement, loyalty, satisfaction and purchase-intent fields
- license: **CC BY 4.0**

See [`DATASET.md`](DATASET.md) for attribution and usage details.

## Final model

**K-Means on min-max-scaled RFM features, K=4**

| Metric | Result |
|---|---:|
| Silhouette score | **0.289** |
| Davies-Bouldin Index | **1.128** |
| Mean pairwise ARI across seeds | **0.980** |
| Customers | **1,000** |

The evaluated silhouette curve peaks at **K=4**.

![Silhouette analysis](assets/kmeans_silhouette.png)

## Customer segments

| Segment | Customers | Avg. recency | Avg. frequency | Avg. monetary | Recommended action |
|---|---:|---:|---:|---:|---|
| **High-Value Repeat Customers** | 261 | 156.3 days | 9.1 | $406.58 | Retention, loyalty recognition, cross-sell and selective upsell. |
| **Frequent Low-Spend Customers** | 264 | 213.5 days | 9.7 | $160.71 | Increase basket size with bundles, cross-sell and spend-threshold offers. |
| **Lapsed High-Spend Customers** | 226 | 278.3 days | 4.4 | $328.50 | Win-back and re-engagement; investigate the reasons for inactivity. |
| **Recent Occasional Customers** | 249 | 99.2 days | 4.1 | $209.95 | Encourage the next purchase and build repeat behavior. |

![RFM profiles](assets/rfm_profile.png)

### High-Value Repeat Customers
High purchase frequency and the highest average monetary value.  
**Focus:** retention, loyalty recognition, relevant cross-sell and selective upsell.

### Frequent Low-Spend Customers
Very frequent purchasing but the lowest average monetary value.  
**Focus:** increase basket size through bundles, complementary products and spend-threshold mechanics.

### Lapsed High-Spend Customers
Meaningful historical spend but the oldest average recency and lower purchase frequency.  
**Focus:** win-back and re-engagement.

### Recent Occasional Customers
The most recent purchases but relatively low purchase frequency.  
**Focus:** encourage the next purchase and build repeat behavior.

> Recommended actions are derived from the segment profiles and were not tested as causal interventions.

## Post-hoc validation

`Customer_Satisfaction` and `Purchase_Intent` were excluded from clustering and analyzed only after the segments were formed.

- Purchase Intent chi-square p-value: **0.795**
- Customer Satisfaction chi-square p-value: **0.914**

The lack of significant association suggests that the RFM clusters primarily capture **transaction value and timing**, rather than all attitudinal dimensions in the source data.

## Repository structure

```text
ecommerce-customer-segmentation/
├── README.md
├── DATASET.md
├── LICENSE
├── requirements.txt
├── data/
│   └── ecommerce_consumer_behavior.csv
├── notebooks/
│   ├── ecommerce_customer_segmentation.ipynb
│   └── exploratory_modeling.ipynb
├── src/
│   ├── customer_segmentation.py
│   └── model_comparison.py
├── results/
│   ├── final_model_metrics.csv
│   ├── k_selection_metrics.csv
│   ├── cluster_summary.csv
│   └── post_analysis_means.csv
└── assets/
    ├── kmeans_elbow.png
    ├── kmeans_silhouette.png
    ├── cluster_sizes.png
    ├── rfm_profile.png
    └── rfm_pca.png
```

## Run locally

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate

pip install -r requirements.txt
python src/customer_segmentation.py
```

## Methodological choices

- Technical identifiers are excluded from modeling.
- Purchase amount is converted from currency text to numeric.
- Recency uses a fixed observation date (`2024-12-31`) for reproducibility.
- RFM features are Min-Max scaled before K-Means.
- `K=4` is selected using internal metrics and interpretability.
- Stability is tested across multiple random seeds using Adjusted Rand Index.
- Satisfaction and purchase intent are held out from cluster formation.
- Demographic and additional behavioral fields are used for profiling rather than defining the final RFM segments.

## Limitations

- The dataset contains 1,000 records and is not a live production customer table.
- The fixed Recency reference date is useful for reproducibility; a production system would use a dynamic observation date.
- RFM intentionally omits product-, channel- and attitude-level behavior from the final clustering space.
- Cluster names are descriptive interpretations, not ground-truth classes.
- Recommended marketing actions require experimental validation before deployment.
