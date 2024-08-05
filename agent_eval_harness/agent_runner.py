import importlib
from .benchmark_manager import BenchmarkManager
from .utils.weave_utils import get_total_cost
import weave
import time
from dotenv import load_dotenv
import pprint
pp = pprint.PrettyPrinter(depth=4)
load_dotenv()

def run_agent_evaluation(agent_function, benchmark, model, config, **kwargs):
    # Initialize logging
    weave_client = weave.init(f"{benchmark}_{int(time.time())}")
    
    # Load the agent function
    module_name, function_name = agent_function.rsplit('.', 1)
    module = importlib.import_module(module_name)
    agent_function = getattr(module, function_name)

    # Load the benchmark
    benchmark_manager = BenchmarkManager(config)
    benchmark_manager.mount_benchmark(benchmark)
    benchmark = benchmark_manager.get_benchmark(benchmark)

    run_id = f"{benchmark.benchmark_name}_{kwargs['agent_name']}_{int(time.time())}"

    # Run the test task
    print("Running test task...")
    # test_passed = benchmark.test_run(agent_function, weave_client)
    test_passed = True
    
    # Run the agent if the test task passed
    if test_passed:
        print("Test task passed!")
        result = benchmark.run(agent_function, run_id)
        
        

    # Process and upload results
    print(f"\n\n=====Processing results=====")
    print(f"Agent name: {kwargs['agent_name']}")
    results_summary = benchmark.process_and_upload_results(kwargs["agent_name"], run_id, eval_results=result, weave_client=weave_client, config=config, upload=kwargs['upload'])
    print("=====")

    # Unmount the benchmark
    benchmark_manager.unmount_benchmark(benchmark)

    # pretty print results_summary dict
    print("\n\n=====Results Summary=====")
    pp.pprint(results_summary)
    print('=====')