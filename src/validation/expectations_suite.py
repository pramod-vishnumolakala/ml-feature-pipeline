"""
Great Expectations validation suite for ML feature pipeline outputs.
Validates 80+ features across all feature groups before they reach
the model training environment.
Pramod Vishnumolakala — github.com/pramod-vishnumolakala
"""

import great_expectations as ge
from great_expectations.core.batch import BatchRequest
from great_expectations.checkpoint import SimpleCheckpoint
import logging

logger = logging.getLogger(__name__)


def build_feature_expectation_suite(context) -> str:
    """
    Build and save the feature expectation suite.
    Returns suite name.
    """
    suite_name = "ml_feature_pipeline_suite"
    suite = context.create_expectation_suite(suite_name, overwrite_existing=True)
    batch_request = BatchRequest(
        datasource_name="s3_feature_store",
        data_connector_name="default_inferred_data_connector",
        data_asset_name="feature_store/",
    )
    validator = context.get_validator(
        batch_request=batch_request,
        expectation_suite_name=suite_name,
    )

    # ── Column presence ───────────────────────────────────────────────
    required_columns = [
        "transaction_id", "account_id", "amount", "event_timestamp",
        # Velocity
        "txn_count_5m", "txn_count_1h", "txn_count_24h",
        "amount_sum_5m", "amount_sum_1h", "amount_sum_24h",
        "amount_avg_24h", "amount_std_24h", "amount_z_score_24h",
        "velocity_breach_5m", "velocity_breach_1h",
        # Account risk
        "total_lifetime_txns", "historical_fraud_flags",
        "fraud_rate_lifetime", "has_fraud_history",
        "account_age_days", "days_since_last_txn",
        # Geo
        "is_high_risk_country", "is_foreign_txn",
        "country_risk_score", "merchant_risk_score",
        # Temporal
        "txn_hour", "is_weekend", "is_off_hours", "is_business_hours",
        # Amount
        "amount_log", "is_round_amount", "is_high_amount", "amount_bucket",
    ]
    for col in required_columns:
        validator.expect_column_to_exist(col)

    # ── Null rate expectations ────────────────────────────────────────
    zero_null_cols = [
        "transaction_id", "account_id", "amount",
        "txn_count_5m", "txn_count_24h",
        "is_high_risk_country", "is_off_hours",
    ]
    for col in zero_null_cols:
        validator.expect_column_values_to_not_be_null(col)

    low_null_cols = [
        "amount_z_score_24h", "amount_std_24h",
        "fraud_rate_lifetime", "days_since_last_txn",
    ]
    for col in low_null_cols:
        validator.expect_column_values_to_not_be_null(col, mostly=0.95)

    # ── Range expectations ────────────────────────────────────────────
    # Binary flags must be 0 or 1
    binary_cols = [
        "is_weekend", "is_off_hours", "is_business_hours",
        "is_high_risk_country", "is_foreign_txn",
        "is_round_amount", "is_high_amount",
        "velocity_breach_5m", "velocity_breach_1h",
        "has_fraud_history",
    ]
    for col in binary_cols:
        validator.expect_column_values_to_be_in_set(col, [0, 1])

    # Amount must be positive
    validator.expect_column_values_to_be_between("amount", min_value=0.01, max_value=1_000_000)
    validator.expect_column_values_to_be_between("amount_log", min_value=0)
    validator.expect_column_values_to_be_between("amount_z_score_24h", min_value=-20, max_value=20)

    # Fraud rate must be between 0 and 1
    validator.expect_column_values_to_be_between("fraud_rate_lifetime", min_value=0, max_value=1)

    # Count features must be non-negative
    count_cols = [
        "txn_count_5m", "txn_count_1h", "txn_count_24h",
        "total_lifetime_txns", "historical_fraud_flags",
        "account_age_days",
    ]
    for col in count_cols:
        validator.expect_column_values_to_be_between(col, min_value=0)

    # Hour must be 0–23
    validator.expect_column_values_to_be_between("txn_hour", min_value=0, max_value=23)

    # Risk scores must be 0–3
    validator.expect_column_values_to_be_between("country_risk_score",  min_value=0, max_value=3)
    validator.expect_column_values_to_be_between("merchant_risk_score", min_value=0, max_value=3)

    # ── Cardinality expectations ──────────────────────────────────────
    validator.expect_column_values_to_be_in_set(
        "amount_bucket", ["MICRO", "SMALL", "MEDIUM", "LARGE", "VERY_LARGE"]
    )

    # ── Volume expectations ───────────────────────────────────────────
    validator.expect_table_row_count_to_be_between(
        min_value=100_000,
        max_value=50_000_000,
    )

    # ── Distribution expectations (data drift detection) ─────────────
    validator.expect_column_mean_to_be_between("amount", min_value=50, max_value=2000)
    validator.expect_column_median_to_be_between("amount", min_value=20, max_value=500)
    validator.expect_column_stdev_to_be_between("amount", min_value=10, max_value=5000)

    # Fraud rate sanity check — should be < 5% of transactions
    validator.expect_column_mean_to_be_between(
        "has_fraud_history", min_value=0, max_value=0.05
    )

    validator.save_expectation_suite(discard_failed_expectations=False)
    logger.info(f"Expectation suite '{suite_name}' saved with {len(suite.expectations)} expectations")
    return suite_name


def run_validation_checkpoint(feature_date: str) -> bool:
    """
    Run the full validation checkpoint for a given feature date.
    Returns True if all expectations pass.
    """
    context = ge.data_context.DataContext()
    suite_name = "ml_feature_pipeline_suite"

    checkpoint_config = {
        "name": "ml_feature_checkpoint",
        "config_version": 1,
        "class_name": "SimpleCheckpoint",
        "validations": [{
            "batch_request": {
                "datasource_name": "s3_feature_store",
                "data_connector_name": "default_inferred_data_connector",
                "data_asset_name": f"feature_store/{feature_date}/",
            },
            "expectation_suite_name": suite_name,
        }],
        "action_list": [
            {"name": "store_validation_result",  "action": {"class_name": "StoreValidationResultAction"}},
            {"name": "store_evaluation_params",  "action": {"class_name": "StoreEvaluationParametersAction"}},
            {"name": "update_data_docs",         "action": {"class_name": "UpdateDataDocsAction"}},
        ],
    }

    checkpoint = SimpleCheckpoint(**checkpoint_config, data_context=context)
    result = checkpoint.run()

    if result["success"]:
        logger.info(f"All feature validations PASSED for {feature_date}")
    else:
        failed = [
            r["expectation_config"]["expectation_type"]
            for suite_result in result["run_results"].values()
            for r in suite_result["validation_result"]["results"]
            if not r["success"]
        ]
        logger.error(f"Feature validations FAILED for {feature_date}: {failed}")

    return result["success"]


if __name__ == "__main__":
    from datetime import datetime
    today = datetime.utcnow().strftime("%Y-%m-%d")
    success = run_validation_checkpoint(today)
    exit(0 if success else 1)
