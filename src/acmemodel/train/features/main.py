import logging

from . import features_generator
from ..data import load_raw_data


logger = logging.getLogger(__name__)


def main():
    data = load_raw_data()
    features_generator.fit_transform(data)
    logger.debug("All features were successfully generated.")


if __name__ == '__main__':
    main()
