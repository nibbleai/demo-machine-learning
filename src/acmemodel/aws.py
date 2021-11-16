from logging import getLogger
from functools import wraps
from typing import Callable

import boto3

from .config import (AWS_S3_BUCKET_NAME, AWS_SECRET_ACCESS_KEY,
                     AWS_ACCESS_KEY_ID)

SESSION = boto3.Session(AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY)
S3 = SESSION.client('s3')
REQUIRED_ENV_VARS = (
    AWS_S3_BUCKET_NAME, AWS_SECRET_ACCESS_KEY, AWS_ACCESS_KEY_ID
)

logger = getLogger(__name__)


def upload_data_to_s3(data: bytes, key: str,
                      bucket: str = AWS_S3_BUCKET_NAME) -> None:
    if any(key is None for key in REQUIRED_ENV_VARS):
        raise ValueError(
            "AWS credentials and default bucket name must be "
            "provided as enviroment variables. Check 'config.py' for info."
        )
    logger.info(f'Uploading {len(data) / 1000:.2f} kb of data to S3...')
    S3.put_object(
        ACL='private',
        Bucket=bucket,
        Body=data,
        Key=key
    )


def download_data_from_s3(key: str, bucket: str = AWS_S3_BUCKET_NAME) -> bytes:
    if any(key is None for key in REQUIRED_ENV_VARS):
        raise ValueError(
            "AWS credentials and default bucket name must be "
            "provided as enviroment variables. Check 'config.py' for info."
        )
    logger.info(f"Downloading '{key}' from S3 bucket...")
    resp = S3.get_object(
        Bucket=bucket,
        Key=key
    )
    # Response's body is a bytes-stream object that we need to read in order
    # to get the binary object's content. Depending on the object size, the
    # reading action can take time, and this simple implementation may not be
    # appropriate.
    return resp['Body'].read()
