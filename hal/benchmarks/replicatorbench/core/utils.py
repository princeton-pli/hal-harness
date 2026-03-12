# core/utils.py
import os
import sys
import json
import logging

# Markers so we can detect our own handlers
_CONSOLE_MARK = "_core_console_handler"
_FILE_MARK = "_core_file_handler"

def get_logger(name: str = "replication"):
    """
    Return a shared logger. Adds ONE console handler (to stdout) the first time only.
    - Use the same `name` ("replication") in all modules to avoid duplicates.
    - Sets propagate=False so logs don't bubble to root and double-print.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # Create a single formatter (returned for your convenience)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Add our console handler once
    has_console = any(
        isinstance(h, logging.StreamHandler) and getattr(h, _CONSOLE_MARK, False)
        for h in logger.handlers
    )
    if not has_console:
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.INFO)
        ch.setFormatter(formatter)
        setattr(ch, _CONSOLE_MARK, True)
        logger.addHandler(ch)

    # Prevent duplication via root logger handlers
    logger.propagate = False

    return logger, formatter

def build_file_description(available_files, file_path):
    desc = ""
    for file_id, (file_name, file_desc) in enumerate(available_files.items(), start=1):
        desc += f"{file_id}. {os.path.join(file_path, file_name)}: {file_desc}\n"
    return desc

def configure_file_logging(logger: logging.Logger, study_path: str, log_file_name: str):
    """
    Ensure exactly ONE file handler for this logger (replace any prior core file handler).
    Safe to call multiple times (idempotent).
    """
    # Remove our prior file handlers (keep third-party ones if any)
    for h in list(logger.handlers):
        if isinstance(h, logging.FileHandler) and getattr(h, _FILE_MARK, False):
            logger.removeHandler(h)
            try:
                h.close()
            except Exception:
                pass
    log_path = os.path.join(study_path, "_log")
    os.makedirs(log_path, exist_ok=True)
    log_file_full_path = os.path.join(log_path, log_file_name)
    fh = logging.FileHandler(log_file_full_path, mode="a", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    # Reuse the same human-friendly format
    fh.setFormatter(logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    ))
    setattr(fh, _FILE_MARK, True)
    logger.addHandler(fh)

    logger.debug(f"[core.utils] File logging configured: {log_file_full_path}")
