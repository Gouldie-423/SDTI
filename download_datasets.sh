#!/bin/bash
# Download SDTI datasets and large result files from S3
#
# Usage:
#   ./download_datasets.sh              # Download everything
#   ./download_datasets.sh --datasets   # Download only datasets
#   ./download_datasets.sh --results    # Download only CostResults.csv
#
# Prerequisites:
#   - AWS CLI installed (https://aws.amazon.com/cli/)
#   - AWS credentials configured (aws configure) or appropriate IAM role
#
# S3 Bucket: sagemaker-us-east-1-505863485541
# Prefix:    sdti-datasets/

S3_BUCKET="sagemaker-us-east-1-505863485541"
S3_PREFIX="sdti-datasets"

download_datasets() {
    echo "Downloading datasets to datasets/ ..."
    mkdir -p datasets/cifar-10-batches-py datasets/digit_mnist
    aws s3 sync "s3://${S3_BUCKET}/${S3_PREFIX}/" datasets/ \
        --exclude "results/*"
    echo "Datasets downloaded."
}

download_results() {
    echo "Downloading CostResults.csv ..."
    aws s3 cp "s3://${S3_BUCKET}/${S3_PREFIX}/results/CostResults.csv" CostResults.csv
    echo "CostResults.csv downloaded."
}

case "${1:-all}" in
    --datasets) download_datasets ;;
    --results)  download_results ;;
    all|*)      download_datasets; download_results ;;
esac

echo ""
echo "Download complete. Dataset files:"
ls -lh datasets/ 2>/dev/null
echo ""
ls -lh CostResults.csv 2>/dev/null
