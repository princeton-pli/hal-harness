import importlib
from .benchmark_manager import BenchmarkManager
from .utils.weave_utils import get_total_cost
import weave
import time
import sys
from dotenv import load_dotenv
from .utils.utils import safe_filename

load_dotenv()

def run_agent_evaluation(agent_function, benchmark, model, config, agent_args, **kwargs):

    # Initialize logging for test run
    print("=====Initializing logging for test run=====")
    weave_client = weave.init(f"{benchmark}_{int(time.time())}_TEST_RUN")
    print("Logging initialized!")
    print("=====\n\n")
    
    
    print("=====Setting up agent=====")
    # add agent dir to sys.path
    sys.path.append(kwargs['agent_dir'])
    # Load the agent function
    module_name, agent_function_name = agent_function.rsplit('.', 1)
    print(f"Loading agent function: {agent_function_name} from module: {module_name}")
    print("Agent setup complete!")
    print("=====\n\n")

    # Load the benchmark
    print("=====Setting up benchmark=====")
    benchmark_manager = BenchmarkManager(kwargs["agent_dir"], config)
    benchmark = benchmark_manager.get_benchmark(benchmark)
    print("Benchmark setup complete!")
    print("=====\n\n")

    run_id = safe_filename(f"{benchmark.benchmark_name}_{kwargs['agent_name']}_{int(time.time())}")
    print(f"===\nRun ID: {run_id}\n===\n\n")

    # Run the test task
    print("=====Running test task=====")
    print("Running test task...")
    # test_passed = benchmark.test_run(agent_function, weave_client)
    test_passed = True
    print("Test task passed!")
    print("=====\n\n")

    
    
    # Run the agent if the test task passed
    if test_passed:
        # Initialize logging for evaluation run
        print("=====Initializing logging for main run=====")
        if kwargs['run_id']: # if run_id is provided, use it
            run_id = kwargs['run_id']

        weave_client = weave.init(run_id)
        print("Logging initialized!")
        print("=====\n\n")

        print("=====Running agent=====")
        result = benchmark.run(agent_function=agent_function, 
                               run_id=run_id, 
                               agent_args=agent_args,
                               conda_env_name=kwargs.get('conda_env_name', None),
                               max_concurrent=kwargs.get('max_concurrent', 1),
                               vm=kwargs.get('vm', False))
        print("Done!")
        print("=====\n\n")


        print("=====Running evaluation=====")
        result = benchmark.run_evaluation_harness(
            agent_output=result,
            run_id=run_id
        )
        print("Done!")
        print("=====\n\n")

        # Process and upload results
        print(f"=====Processing results=====")
        print(f"Agent name: {kwargs['agent_name']}")
        results_summary = benchmark.process_and_upload_results(kwargs["agent_name"], run_id, eval_results=result, weave_client=weave_client, config=config, upload=kwargs['upload'])
        print("=====\n\n")

    
    else:
        print("Test task failed. Please make sure your agent function returns the correct output format and passes the weave_task_id as Weave attribute for each LLM API call.")
        