"""
Unit tests for the logger module.
"""
import logging
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
    logger_instance.handlers.clear()
