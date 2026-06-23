#!/usr/bin/env python3
"""Logging configuration."""

from __future__ import annotations

import logging
import sys

_CONFIGURED = False


def get_logger(name: str = "talent_flow") -> logging.Logger:
    global _CONFIGURED
    logger = logging.getLogger(name)
    if not _CONFIGURED:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
        )
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        _CONFIGURED = True
    return logger
