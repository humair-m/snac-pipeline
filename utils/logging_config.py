"""
Logging configuration to suppress verbose output from libraries
and show only pipeline-relevant logs.
"""

import logging
import warnings
import os


def setup_logging():
    """
    Configure logging to suppress verbose library output.
    Shows only: tqdm progress bars, dataset loading, and reader/worker logs.
    """

    warnings.filterwarnings("ignore")

    os.environ["TRANSFORMERS_VERBOSITY"] = "error"
    os.environ["DATASETS_VERBOSITY"] = "error"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    os.environ["HF_HUB_VERBOSITY"] = "error"

    logging.basicConfig(
        level=logging.CRITICAL,
        format='%(message)s',
        force=True
    )

    for logger_name in ["pytorch_lightning", "lightning", "lightning_fabric"]:
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.CRITICAL)
        logger.propagate = False

    for logger_name in ["datasets", "datasets.builder", "datasets.info"]:
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.CRITICAL)
        logger.propagate = False

    logging.getLogger("transformers").setLevel(logging.CRITICAL)
    logging.getLogger("filelock").setLevel(logging.CRITICAL)

    for logger_name in ["urllib3", "requests", "fsspec", "asyncio"]:
        logging.getLogger(logger_name).setLevel(logging.CRITICAL)
