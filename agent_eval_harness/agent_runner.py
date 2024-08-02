import importlib
from .benchmark_manager import load_benchmark
from .utils.weave_utils import get_total_cost
import weave
import time
from dotenv import load_dotenv
load_dotenv()

def run_agent_evaluation(agent_function, benchmark, model, config, **kwargs):
    # Initialize logging
    weave_client = weave.init(f"{benchmark}_{int(time.time())}")
    
    # Load the agent function
    module_name, function_name = agent_function.rsplit('.', 1)
    module = importlib.import_module(module_name)
    agent_function = getattr(module, function_name)

    # Load the benchmark
    benchmark = load_benchmark(config, benchmark)

    run_id = f"{benchmark.benchmark_name}_{kwargs['agent_name']}_{int(time.time())}"

    # Run the test task
    print("Running test task...")
    test_passed = benchmark.test_run(agent_function, weave_client)
    if test_passed:
        print("Test task passed!")

    # Run the full evaluation
    if test_passed:
        result = benchmark.run(agent_function, run_id)

    # get total cost TODO add more refined preprocessing for logs
    processed_logs = get_total_cost(weave_client)

    # Process and upload results
    benchmark.process_and_upload_results(kwargs["agent_name"], run_id, eval_results=result, logging_results={'total_cost': processed_logs}, config=config, upload=kwargs['upload'])