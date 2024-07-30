import importlib
import weave
from .benchmark_manager import load_benchmark
from .logging_manager import initialize_logging
from .result_manager import store_and_upload_results
from .utils.validation import validate_agent_output

def run_agent_evaluation(agent_path, benchmark_name, model, config):
    # Initialize logging
    # weave.init(config['weave_project_name'])
    
    # Load the agent function
    module_name, function_name = agent_path.rsplit('.', 1)
    module = importlib.import_module(module_name)
    agent_function = getattr(module, function_name)

    # Load the benchmark
    benchmark = load_benchmark(config, benchmark_name)

    # Run the test task
    print("Running test task...")
    test_passed = benchmark.test_run(agent_function)
    if test_passed:
        print("Test task passed!")

    # Run the full evaluation
    if test_passed:
        result = benchmark.run(agent_function)

    # # Store and upload results
    store_and_upload_results(result, benchmark_name, agent_path, model, config)

    # return result