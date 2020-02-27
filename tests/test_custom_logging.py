"""
Tests are only done for the added color_info method, as the rest of the
logging is vanilla python logging.
"""

import pytest
from edflow.custom_logging import log
import os
import shutil
import logging


def test_color_info():

    try:
        os.makedirs("color_logs")
        logger = log._create_logger("test_logger", "./color_logs", logging.DEBUG)

        logger.info("msg", color=1)
        logger.info("msg", color="green")
        logger.info("msg", color="y")
        logger.debug("msg", color="c")
        logger.error("msg", color="m")
        logger.critical("msg", color="blue")
        logger.info("msg", color="white")
        logger.info("msg", color="black")

        logger.info("nocol")
        logger.debug("nocol")

        logger.info("test", extra={"color": "red"})
        logger.debug("test", extra={"color": "green"})
        logger.critical("test", extra={"color": "yellow"})

        with pytest.raises(ValueError):
            logger.info("msg", color="wrong")

    finally:
        shutil.rmtree("color_logs")


if __name__ == "__main__":
    test_color_info()
