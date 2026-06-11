import os, logging

def setup_logger(log_file: str) -> logging.Logger:
    logger = logging.getLogger("generate")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    fh = logging.FileHandler(log_file)
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    return logger


