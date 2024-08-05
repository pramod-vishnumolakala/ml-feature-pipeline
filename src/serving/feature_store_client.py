"""
Feature Serving — retrieves ML-ready features from S3 feature store
for model training and real-time inference.
Pramod Vishnumolakala — github.com/pramod-vishnumolakala
"""

import logging
import boto3
import pandas as pd
import pyarrow.parquet as pq
import s3fs
from datetime import datetime, timedelta

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

FEATURE_BUCKET = "pramod-ml-features"
FEATURE_PREFIX = "feature_store"
REGION         = "us-east-1"


class FeatureStore:
    """
    Reads ML features from S3 feature store.
    Supports point-in-time correct feature retrieval to prevent data leakage.
    """

    def __init__(self):
        self.s3  = s3fs.S3FileSystem()
        self.bucket = FEATURE_BUCKET

    def get_features_for_date(self, feature_date: str, feature_groups: list[str] = None) -> pd.DataFrame:
        """
        Load feature set for a specific date.
        Used by model training jobs.
        """
        path = f"s3://{self.bucket}/{FEATURE_PREFIX}/{feature_date}/"
        logger.info(f"Loading features from {path}")

        df = pd.read_parquet(path, filesystem=self.s3)

        if feature_groups:
            cols = self._get_columns_for_groups(feature_groups)
            available = [c for c in cols if c in df.columns]
            df = df[["transaction_id", "account_id"] + available]

        logger.info(f"Loaded {len(df):,} feature records for {feature_date}")
        return df

    def get_features_for_account(self, account_id: str, as_of_date: str) -> pd.DataFrame:
        """
        Point-in-time correct feature retrieval for a single account.
        Prevents data leakage in model training.
        """
        df = self.get_features_for_date(as_of_date)
        account_features = df[df["account_id"] == account_id]
        if account_features.empty:
            logger.warning(f"No features found for account {account_id} on {as_of_date}")
        return account_features

    def get_training_dataset(
        self,
        start_date: str,
        end_date: str,
        feature_groups: list[str] = None,
        sample_rate: float = 1.0,
    ) -> pd.DataFrame:
        """
        Build a training dataset across a date range.
        Concatenates daily feature snapshots.
        """
        start = datetime.strptime(start_date, "%Y-%m-%d")
        end   = datetime.strptime(end_date,   "%Y-%m-%d")
        dfs   = []

        current = start
        while current <= end:
            date_str = current.strftime("%Y-%m-%d")
            try:
                df = self.get_features_for_date(date_str, feature_groups)
                if sample_rate < 1.0:
                    df = df.sample(frac=sample_rate, random_state=42)
                dfs.append(df)
            except Exception as exc:
                logger.warning(f"No features for {date_str}: {exc}")
            current += timedelta(days=1)

        if not dfs:
            raise ValueError(f"No feature data found between {start_date} and {end_date}")

        result = pd.concat(dfs, ignore_index=True)
        logger.info(f"Training dataset: {len(result):,} rows across {len(dfs)} days")
        return result

    def list_available_dates(self, last_n_days: int = 30) -> list[str]:
        """List dates for which features are available in S3."""
        prefix = f"{self.bucket}/{FEATURE_PREFIX}/"
        try:
            paths = self.s3.ls(f"s3://{prefix}")
            dates = [p.split("/")[-1] for p in paths if "/" in p]
            return sorted(dates)[-last_n_days:]
        except Exception as exc:
            logger.error(f"Failed to list feature dates: {exc}")
            return []

    @staticmethod
    def _get_columns_for_groups(groups: list[str]) -> list[str]:
        group_cols = {
            "velocity":     ["txn_count_5m", "txn_count_1h", "txn_count_24h",
                             "amount_sum_5m", "amount_sum_1h", "amount_sum_24h",
                             "amount_z_score_24h", "velocity_breach_5m"],
            "account_risk": ["total_lifetime_txns", "fraud_rate_lifetime",
                             "has_fraud_history", "account_age_days", "is_new_account"],
            "geo":          ["is_high_risk_country", "is_foreign_txn",
                             "country_risk_score", "merchant_risk_score"],
            "temporal":     ["txn_hour", "is_weekend", "is_off_hours", "is_business_hours"],
            "amount":       ["amount_log", "is_round_amount", "is_high_amount", "amount_bucket"],
        }
        cols = []
        for group in groups:
            cols.extend(group_cols.get(group, []))
        return list(dict.fromkeys(cols))   # deduplicate preserving order


if __name__ == "__main__":
    store = FeatureStore()
    dates = store.list_available_dates(last_n_days=7)
    logger.info(f"Available feature dates: {dates}")

    if dates:
        df = store.get_features_for_date(dates[-1], feature_groups=["velocity", "geo"])
        logger.info(f"Sample features:\n{df.head()}")
