"""CLI command for hal-test"""

import click
from .test_command import run_hal_test
from .utils.logging_utils import print_step


@click.command()
@click.option('--verbose', '-v', 
              is_flag=True,
              help='Show detailed output and errors')
@click.option('--no-cleanup', 
              is_flag=True,
              help='Skip cleanup of test artifacts (useful for debugging)')
def test_command(verbose, no_cleanup):
    """Run quick validation test on HAL installation with corebench_easy.
    
    This command runs a minimal test using a deterministic test agent to verify 
    that the system is working correctly. It uses corebench_easy which only requires
    reading local files - no Docker or external APIs needed.
    
    The test automatically handles all setup including decrypting test files and 
    installing dependencies. First run will download one test capsule.
    
    Examples:
        hal-test                    # Run test
        hal-test -v                 # Verbose output  
        hal-test --no-cleanup       # Keep test artifacts for debugging
    """
    
    # Run the test
    success = run_hal_test(
        verbose=verbose,
        cleanup=not no_cleanup
    )
    
    # Exit with appropriate code
    exit(0 if success else 1)


if __name__ == '__main__':
    test_command()