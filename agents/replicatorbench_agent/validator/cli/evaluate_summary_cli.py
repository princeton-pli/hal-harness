import argparse
import json

from validator.aggregate_eval_scores import summarize_eval_scores


def main():
    parser = argparse.ArgumentParser(description="Validator: Aggregate and summarize evaluation scores")
    parser.add_argument('--study_path', type=str, required=True, help='Path to study with agents output for evaluation')
    args = parser.parse_args()

    summarize_eval_scores(args.study_path)


if __name__ == "__main__":
    main()
