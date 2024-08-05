# ============================================================
# ML Feature Engineering Pipeline — Terraform Infrastructure (AWS)
# Author: Pramod Vishnumolakala
# ============================================================

terraform {
  required_version = ">= 1.4"
  required_providers {
    aws = { source = "hashicorp/aws", version = "~> 5.0" }
  }
  backend "s3" {
    bucket = "pramod-terraform-state"
    key    = "ml-feature-pipeline/terraform.tfstate"
    region = "us-east-1"
  }
}

provider "aws" { region = "us-east-1" }

variable "environment"  { default = "production" }
variable "project_name" { default = "ml-feature-pipeline" }


# ── S3 — feature store ────────────────────────────────────────────────
resource "aws_s3_bucket" "feature_store" {
  bucket = "pramod-ml-features"
  tags   = { Project = var.project_name }
}

resource "aws_s3_bucket_versioning" "feature_store" {
  bucket = aws_s3_bucket.feature_store.id
  versioning_configuration { status = "Enabled" }
}

resource "aws_s3_bucket_lifecycle_configuration" "feature_store" {
  bucket = aws_s3_bucket.feature_store.id
  rule {
    id     = "expire-old-features"
    status = "Enabled"
    expiration { days = 90 }   # keep 90 days of feature snapshots
  }
}


# ── Glue jobs — one per feature group ────────────────────────────────
resource "aws_iam_role" "glue_role" {
  name = "${var.project_name}-glue-role"
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Action    = "sts:AssumeRole"
      Effect    = "Allow"
      Principal = { Service = "glue.amazonaws.com" }
    }]
  })
}

resource "aws_iam_role_policy_attachment" "glue_service" {
  role       = aws_iam_role.glue_role.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSGlueServiceRole"
}

locals {
  glue_jobs = {
    velocity     = "ml-feature-velocity"
    account_risk = "ml-feature-account-risk"
    geo_amount   = "ml-feature-geo-amount"
  }
}

resource "aws_glue_job" "feature_jobs" {
  for_each     = local.glue_jobs
  name         = each.value
  role_arn     = aws_iam_role.glue_role.arn
  glue_version = "4.0"

  command {
    name            = "glueetl"
    script_location = "s3://pramod-glue-scripts/${each.value}.py"
    python_version  = "3"
  }

  default_arguments = {
    "--job-language"    = "python"
    "--enable-metrics"  = ""
    "--S3_OUTPUT"       = "s3://${aws_s3_bucket.feature_store.bucket}/feature_store/"
  }

  number_of_workers = 8
  worker_type       = "G.1X"
  timeout           = 60
}


# ── MWAA (Managed Airflow) ────────────────────────────────────────────
resource "aws_s3_bucket" "airflow_dags" {
  bucket = "pramod-ml-airflow-dags"
  tags   = { Project = var.project_name }
}

resource "aws_mwaa_environment" "ml_pipeline" {
  name               = "${var.project_name}-airflow"
  airflow_version    = "2.8.1"
  environment_class  = "mw1.medium"
  max_workers        = 5
  min_workers        = 1
  dag_s3_path        = "dags/"
  source_bucket_arn  = aws_s3_bucket.airflow_dags.arn

  execution_role_arn = aws_iam_role.glue_role.arn

  logging_configuration {
    dag_processing_logs { enabled   = true; log_level = "INFO" }
    scheduler_logs      { enabled   = true; log_level = "WARNING" }
    task_logs           { enabled   = true; log_level = "INFO" }
    webserver_logs      { enabled   = true; log_level = "ERROR" }
    worker_logs         { enabled   = true; log_level = "INFO" }
  }

  tags = { Project = var.project_name }
}


# ── SNS — pipeline alerts ─────────────────────────────────────────────
resource "aws_sns_topic" "ml_alerts" {
  name = "ml-feature-alerts"
}

resource "aws_sns_topic_subscription" "email" {
  topic_arn = aws_sns_topic.ml_alerts.arn
  protocol  = "email"
  endpoint  = "pramodvishnumolakala@gmail.com"
}


# ── Outputs ───────────────────────────────────────────────────────────
output "feature_store_bucket" { value = aws_s3_bucket.feature_store.bucket }
output "airflow_url"          { value = aws_mwaa_environment.ml_pipeline.webserver_url }
output "glue_job_names"       { value = [for j in aws_glue_job.feature_jobs : j.name] }
