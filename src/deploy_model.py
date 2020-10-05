from pathlib import Path
import re

from .aws import upload_data_to_s3
from .config import ( MODEL_S3_STORAGE_KEY, DATASET_S3_STORAGE_KEY,
                     TRAINING_REPORT_S3_STORAGE_KEY)
from .train.log import MODEL_FILENAME, REPORT_FILENAME, DATASET_FILENAME
from .utils import read_binary_data_from_file, find_experiment_directory

FILENAME_TO_S3_KEY = {
    # Map local file names to S3 keys
    MODEL_FILENAME: MODEL_S3_STORAGE_KEY,
    REPORT_FILENAME: TRAINING_REPORT_S3_STORAGE_KEY,
    DATASET_FILENAME: DATASET_S3_STORAGE_KEY
}


def main() -> None:
    model_id = input('Enter the ID of the model to be deployed:\n>  ')
    if not re.match(r'^[a-f0-9]{40}$', model_id):
        raise ValueError('Model id must be a valid sha1 hexdigest.')

    directory = find_experiment_directory(model_id)
    deploy(directory)
    print('---\nModel successfully deployed!')


def deploy(experiment_directory: Path) -> None:
    """Upload a fitted model along with its experiment artifacts to S3."""
    for name in [MODEL_FILENAME, REPORT_FILENAME, DATASET_FILENAME]:
        data = read_binary_data_from_file(f'{experiment_directory}/{name}')
        upload_data_to_s3(data, FILENAME_TO_S3_KEY[name])


if __name__ == '__main__':
    main()
