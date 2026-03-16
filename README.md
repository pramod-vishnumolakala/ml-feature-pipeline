# ML Feature Engineering Pipeline

Production-grade ML feature engineering pipeline on AWS that prepares fraud, risk and customer propensity features for data science teams. Reduced model deployment timelines by **30%** by delivering clean, validated, ML-ready feature datasets from enterprise data assets.

## Architecture

```
Source Data (Redshift + S3)
┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│  Transaction │  │  Customer    │  │   Account    │
│  Features    │  │  360 Profile │  │   History    │
└──────┬───────┘  └──────┬───────┘  └──────┬───────┘
       │                 │                  │
       └─────────────────┼──────────────────┘
                         │
              ┌──────────▼──────────┐
              │   Apache Airflow    │  ← Pipeline orchestration
              │   (feature DAGs)    │
              └──────────┬──────────┘
                         │
              ┌──────────▼──────────┐
              │   AWS Glue PySpark  │  ← Feature computation
              │   + Feature Store   │
              └──────────┬──────────┘
                         │
              ┌──────────▼──────────┐
              │  Great Expectations │  ← Feature validation
              └──────────┬──────────┘
                         │
               ┌─────────┴──────────┐
               │                    │
    ┌──────────▼──────┐   ┌─────────▼─────────┐
    │  Amazon S3      │   │  Amazon Redshift   │
    │  Feature Store  │   │  (model training)  │
    └─────────────────┘   └───────────────────┘
```

## Feature Groups

| Feature Group | Features | Target Models |
|---|---|---|
| **Transaction Velocity** | txn_count_5m/1h/24h, amount_sum/avg, velocity_breach | Fraud detection |
| **Account Risk** | risk_score, historical_fraud_flag, dispute_count | Credit risk |
| **Customer Behaviour** | renewal_rate, claim_frequency, tenure_years | Churn prediction |
| **Geo Anomaly** | country_risk_score, distance_from_home, is_foreign_txn | Fraud detection |
| **Temporal** | is_off_hours, is_weekend, days_since_last_txn | All models |
| **Merchant** | category_risk, merchant_fraud_rate, is_new_merchant | Fraud detection |

## Key Features

- **30% faster** model deployment via standardised feature contracts
- **Automated validation** with Great Expectations on every feature run
- **Reusable feature library** — 80+ features across 6 feature groups
- **Point-in-time correct** feature joins to prevent data leakage
- **Airflow DAGs** — automated daily/hourly feature refresh
- **Feature registry** — versioned feature definitions with lineage

## Tech Stack

| Layer | Technology |
|---|---|
| Orchestration | Apache Airflow |
| Processing | AWS Glue, PySpark |
| Feature store | Amazon S3 (Parquet), Amazon Redshift |
| Validation | Great Expectations, Deequ |
| Storage | Amazon S3, Amazon Redshift |
| Security | AWS KMS, IAM |
| IaC | Terraform |

## Author

**Pramod Vishnumolakala** — Senior Data Engineer  
[pramodvishnumolakala@gmail.com](mailto:pramodvishnumolakala@gmail.com) · [LinkedIn](https://linkedin.com/in/pramod-vishnumolakala)
