"""
Unit tests for the logger module.
"""
import logging
import os
from src.logger import get_logger

def test_get_logger():
    """
    Test the get_logger function to ensure it creates a logger,
    and subsequent calls return the same instance without adding handlers.
    """
    logger_name = "ACCORD_TEST_SINGLE"

    # Ensure the logger is clean before the test
    logger_instance = logging.getLogger(logger_name)
    logger_instance.handlers.clear()
    logger_instance.propagate = False

    # First call should create and configure the logger
    logger1 = get_logger(name=logger_name)
    assert isinstance(logger1, logging.Logger)
    assert logger1.name == logger_name
    initial_handler_count = len(logger1.handlers)
    assert initial_handler_count > 0

    # Second call with the same name should return the same logger instance
    logger2 = get_logger(name=logger_name)
    assert logger2 is logger1
    # And it should not have added more handlers
    assert len(logger2.handlers) == initial_handler_count

    # Clean up handlers after test
    for handler in logger1.handlers:
        handler.close()
    logger_instance.handlers.clear()

def test_get_logger_truncation():
    """
    Test that calling get_logger multiple times with the same log_file
    does not truncate the file after it has been written to.
    """
    logger_name = "ACCORD_TEST_TRUNCATION"
    log_file = "test_truncation.log"

    # Clean up before test
    if os.path.exists(log_file):
        os.remove(log_file)

    logger_instance = logging.getLogger(logger_name)
    logger_instance.handlers.clear()

    try:
        # 1. Initialise logger
        logger = get_logger(name=logger_name, log_file=log_file)

        # 2. Write something
        test_msg = "Initial log line"
        logger.info(test_msg)

        # Ensure it is flushed
        for handler in logger.handlers:
            handler.flush()

        # Verify it's there
        with open(log_file, 'r', encoding='utf-8') as f:
            content = f.read()
        assert test_msg in content

        # 3. Call get_logger again with same file
        logger2 = get_logger(name=logger_name, log_file=log_file)
        assert logger2 is logger

        # 4. Verify file still contains the original message (no truncation)
        with open(log_file, 'r', encoding='utf-8') as f:
            content = f.read()
        assert test_msg in content, "Log file was truncated when calling get_logger again!"

    finally:
        # Clean up
        for handler in logger_instance.handlers:
            handler.close()
        logger_instance.handlers.clear()
        if os.path.exists(log_file):
            try:
                os.remove(log_file)
            except PermissionError:
                pass
