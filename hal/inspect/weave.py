from contextlib import contextmanager
from typing import Any, Iterator
from .log import log, log_end, log_start
import weave  # type: ignore


@contextmanager
def weave_tracing(run_id: str) -> Iterator[Any]:
    """
    Context manager for initializing and finalizing the weave client for logging.

    Args:
        run_id (str): Unique identifier for the current run.

    Yields:
        weave_client (Any): The initialized weave client.
    """

    try:
        log()
        log_start("Weave Init")
        weave_client = weave.init(run_id)
        log(f"run_id: {run_id}")
        log_end()
        yield weave_client
    finally:
        weave.finish()
