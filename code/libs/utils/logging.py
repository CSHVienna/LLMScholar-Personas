"""Logging helpers used by every script."""

import logging

import pandas as pd


def setup_logging(level: int = logging.INFO) -> logging.Logger:
    """Configure root logging and return a module-level logger.

    Usage at the top of a script:

        from libs.utils.logging import setup_logging
        logger = setup_logging()
    """
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    return logging.getLogger()


def log_value_counts(
    df: pd.DataFrame,
    column: str,
    *,
    title: str | None = None,
    width: int = 20,
    logger: logging.Logger | None = None,
) -> None:
    """Log the value-count distribution of a column with percentages.

    Replaces the boilerplate at the end of every factuality script:

        logger.info("Status distribution:")
        for status, count in df["xxx_status"].value_counts().items():
            logger.info("  %-20s %6d  (%.1f%%)", status, count, 100 * count / n)
    """
    log = logger or logging.getLogger()
    n = len(df)
    log.info("%s:", title or f"{column} distribution")
    for value, count in df[column].value_counts().items():
        pct = 100 * count / n if n else 0.0
        log.info("  %-*s %6d  (%.1f%%)", width, str(value), count, pct)
