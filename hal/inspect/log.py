import logging

logger = logging.getLogger("agent_eval")


# FIXME: remove this file
def log_start(msg: str) -> None:
    logger.info(f"====={msg}=====")


def log(msg: str = "") -> None:
    logger.info(msg)


def log_end(msg: str | None = None) -> None:
    if msg:
        logger.info(msg)
    logger.info("===============\n")
