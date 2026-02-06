"""Utility functions for v2."""

import logging
import subprocess


logger = logging.getLogger(__name__)


def run_command(
    cmd: list[str], check: bool = True
) -> subprocess.CompletedProcess:
    """Run a subprocess command with logging.

    Args:
        cmd: Command and arguments as list
        check: If True, raise on non-zero exit code

    Returns:
        CompletedProcess with stdout/stderr captured

    Raises:
        subprocess.CalledProcessError: If check=True and command fails
    """
    logger.debug(f"Running command: {' '.join(cmd)}")

    result = subprocess.run(cmd, capture_output=True, text=True, check=False)

    if result.stdout:
        logger.debug(f"Command stdout: {result.stdout}")
    if result.stderr:
        logger.debug(f"Command stderr: {result.stderr}")

    if check and result.returncode != 0:
        logger.error(f"Command failed with exit code {result.returncode}: {' '.join(cmd)}")
        raise subprocess.CalledProcessError(
            result.returncode, cmd, result.stdout, result.stderr
        )
    elif result.returncode != 0:
        logger.warning(f"Command failed (ignoring): {' '.join(cmd)}")

    return result
