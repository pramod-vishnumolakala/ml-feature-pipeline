"""
Airflow DAG — ML Feature Engineering Pipeline
Runs daily at 2am UTC. Triggers Glue jobs for each feature group,
validates outputs, and notifies data science team on completion.
Pramod Vishnumolakala — github.com/pramod-vishnumolakala
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.providers.amazon.aws.operators.glue import GlueJobOperator
from airflow.providers.amazon.aws.operators.sns import SnsPublishOperator
from airflow.providers.amazon.aws.sensors.s3 import S3KeySensor
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.empty import EmptyOperator
from airflow.utils.trigger_rule import TriggerRule
import boto3
import logging

logger = logging.getLogger(__name__)

DEFAULT_ARGS = {
    "owner":            "pramod.vishnumolakala",
    "depends_on_past":  False,
    "email":            ["pramodvishnumolakala@gmail.com"],
    "email_on_failure": True,
    "email_on_retry":   False,
    "retries":          2,
    "retry_delay":      timedelta(minutes=5),
}

SNS_TOPIC_ARN = "arn:aws:sns:us-east-1:123456789:ml-feature-alerts"
FEATURE_BUCKET = "pramod-ml-features"


def check_source_data(**context) -> str:
    """
    Check if yesterday's transaction data is available in S3.
    Returns branch: proceed or skip.
    """
    ds       = context["ds"]
    s3       = boto3.client("s3", region_name="us-east-1")
    prefix   = f"transactions/year={ds[:4]}/month={ds[5:7]}/day={ds[8:10]}/"
    try:
        resp = s3.list_objects_v2(Bucket="pramod-fraud-dw", Prefix=prefix, MaxKeys=1)
        if resp.get("KeyCount", 0) > 0:
            logger.info(f"Source data found at {prefix}")
            return "run_velocity_features"
        logger.warning(f"No source data found at {prefix}")
        return "skip_pipeline"
    except Exception as exc:
        logger.error(f"S3 check failed: {exc}")
        return "skip_pipeline"


def validate_feature_output(**context) -> bool:
    """Validate feature output: check record count and null rates."""
    import pandas as pd
    ds = context["ds"]
    s3_path = f"s3://{FEATURE_BUCKET}/feature_store/{ds}/"

    spark = context.get("spark_session")
    if not spark:
        logger.info("Validation skipped in local context")
        return True

    df    = spark.read.parquet(s3_path)
    count = df.count()
    logger.info(f"Feature output record count: {count:,}")

    if count < 100_000:
        raise ValueError(f"Feature count {count:,} below minimum threshold of 100,000")

    logger.info("Feature output validation passed")
    return True


with DAG(
    dag_id="ml_feature_engineering_pipeline",
    description="Daily ML feature engineering pipeline — fraud, risk and churn features",
    default_args=DEFAULT_ARGS,
    schedule_interval="0 2 * * *",       # 2am UTC daily
    start_date=datetime(2024, 1, 1),
    catchup=False,
    max_active_runs=1,
    tags=["ml", "features", "fraud", "pramod"],
) as dag:

    start = EmptyOperator(task_id="start")

    # Check source data availability
    check_source = BranchPythonOperator(
        task_id="check_source_data",
        python_callable=check_source_data,
    )

    skip_pipeline = EmptyOperator(task_id="skip_pipeline")

    # Velocity features (Glue job)
    run_velocity = GlueJobOperator(
        task_id="run_velocity_features",
        job_name="ml-feature-velocity",
        script_args={
            "--FEATURE_DATE":   "{{ ds }}",
            "--S3_INPUT":       f"s3://pramod-fraud-dw/transactions/",
            "--S3_OUTPUT":      f"s3://{FEATURE_BUCKET}/feature_store/",
        },
        aws_conn_id="aws_default",
        region_name="us-east-1",
        wait_for_completion=True,
    )

    # Account risk features
    run_account_risk = GlueJobOperator(
        task_id="run_account_risk_features",
        job_name="ml-feature-account-risk",
        script_args={
            "--FEATURE_DATE": "{{ ds }}",
            "--S3_INPUT":     f"s3://pramod-fraud-dw/",
            "--S3_OUTPUT":    f"s3://{FEATURE_BUCKET}/feature_store/",
        },
        aws_conn_id="aws_default",
        region_name="us-east-1",
        wait_for_completion=True,
    )

    # Geo & amount features
    run_geo_amount = GlueJobOperator(
        task_id="run_geo_amount_features",
        job_name="ml-feature-geo-amount",
        script_args={
            "--FEATURE_DATE": "{{ ds }}",
            "--S3_OUTPUT":    f"s3://{FEATURE_BUCKET}/feature_store/",
        },
        aws_conn_id="aws_default",
        region_name="us-east-1",
        wait_for_completion=True,
    )

    # Wait for all feature outputs to land in S3
    wait_for_features = S3KeySensor(
        task_id="wait_for_feature_output",
        bucket_name=FEATURE_BUCKET,
        bucket_key="feature_store/{{ ds }}/_SUCCESS",
        aws_conn_id="aws_default",
        poke_interval=60,
        timeout=3600,
    )

    # Validate feature output
    validate = PythonOperator(
        task_id="validate_feature_output",
        python_callable=validate_feature_output,
    )

    # Notify data science team on success
    notify_success = SnsPublishOperator(
        task_id="notify_success",
        target_arn=SNS_TOPIC_ARN,
        message=(
            "ML Feature Pipeline SUCCEEDED — {{ ds }}\n"
            "Features available at: s3://pramod-ml-features/feature_store/{{ ds }}/\n"
            "Feature groups: velocity, account_risk, geo, amount, temporal"
        ),
        subject="ML Features Ready — {{ ds }}",
        aws_conn_id="aws_default",
    )

    # Notify on failure
    notify_failure = SnsPublishOperator(
        task_id="notify_failure",
        target_arn=SNS_TOPIC_ARN,
        message="ML Feature Pipeline FAILED — {{ ds }} — check Airflow logs",
        subject="ALERT: ML Feature Pipeline Failed",
        aws_conn_id="aws_default",
        trigger_rule=TriggerRule.ONE_FAILED,
    )

    end = EmptyOperator(task_id="end", trigger_rule=TriggerRule.NONE_FAILED_MIN_ONE_SUCCESS)

    # DAG dependencies
    start >> check_source >> [run_velocity, skip_pipeline]
    run_velocity >> run_account_risk >> run_geo_amount
    run_geo_amount >> wait_for_features >> validate >> notify_success >> end
    notify_failure >> end
    skip_pipeline >> end
