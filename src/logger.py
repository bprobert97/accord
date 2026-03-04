"""
The Autonomous Cooperative Consensus Orbit Determination (ACCORD) framework.
Author: Beth Probert
Email: beth.probert@strath.ac.uk

Copyright (C) 2025 Applied Space Technology Laboratory

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""

import logging
from typing import Optional
from logging import Logger

def get_logger(name: str = "ACCORD", log_file: Optional[str] = None) -> Logger:
    """
    Returns a logger that is safe to use across multiple modules.
    Configures the logger to write to a file.

    Args:
    - name: The name of the logger.
    - log_file: The file to write logs to. If None, defaults to "app.log" for new loggers.

    Returns:
    - A configured Logger instance.
    """
    logger = logging.getLogger(name)
    
    # If no handlers exist, do initial configuration
    if not logger.hasHandlers():
        logger.setLevel(logging.DEBUG)
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )

        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(formatter)
        logger.addHandler(ch)

        # File handler
        actual_log_file = log_file if log_file else "app.log"
        fh = logging.FileHandler(actual_log_file, mode='w')
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

        # Optional: prevent messages from propagating to the root logger
        logger.propagate = False
    elif log_file is not None:
        # If handlers exist but a specific log_file is requested, 
        # update the FileHandler to point to the new file.
        # This is useful for Monte Carlo runs where we want to redirect the same logger.
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        
        # Remove old file handlers
        for handler in logger.handlers[:]:
            if isinstance(handler, logging.FileHandler):
                logger.removeHandler(handler)
        
        # Add new file handler
        fh = logging.FileHandler(log_file, mode='w')
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger
