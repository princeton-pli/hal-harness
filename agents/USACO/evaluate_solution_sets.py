import argparse
import pickle
from collections import Counter
import os

from USACOBench.evaluation import evaluate_solution_sets, print_metrics
import json

parser = argparse.ArgumentParser()
parser.add_argument(
    "-s", "--ss", help="file path to solution sets", default="solution_sets.pickle"
)
parser.add_argument(
    "-d", "--dataset_name", help="name of problem dataset", default="usaco_subset307"
)
parser.add_argument(
    "-v", "--verbose", action="store_true", help="whether to print result metrics"
)
parser.add_argument("-m", "--mode", help="fail_fast or eval_all", default="eval_all")
parser.add_argument(
    "-r",
    "--rs",
    help="file path to save results (default is results.pickle)",
    default="result_sets.pickle",
)
args = parser.parse_args()

# eval
with open(args.ss, "rb") as f:
    solution_sets = pickle.load(f)

abs = os.path.abspath(__file__).replace("/evaluate_solution_sets.py", "")
with open("{}/data/datasets/{}_dict.json".format(abs, args.dataset_name), "r") as f:
    problem_dict = json.load(f)

result_sets = evaluate_solution_sets(solution_sets, problem_dict, mode=args.mode)

# print
if args.verbose:
    print_metrics(result_sets)
    print("Result summary:")
    result_types = [
        result["result_type"] for result_set in result_sets for result in result_set
    ]
    print(Counter(result_types))
    print()

# save
fname = args.rs
print("Saving results at {}...".format(fname))
with open(fname, "wb") as f:
    pickle.dump(result_sets, f)
