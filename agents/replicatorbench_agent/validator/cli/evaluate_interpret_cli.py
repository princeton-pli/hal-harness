import argparse
import json

from validator.evaluate_interpret import extract_from_human_replication_study


def extract_human_replication_info():
    parser = argparse.ArgumentParser(description="Validator: Evaluate interpret report based on human replication report")
    parser.add_argument('--reference_report_path', type=str, required=True, help='Path to the human replication path')
    parser.add_argument('--study_path', type=str, required=True, help='Path to study with agents output for evaluation')
    args = parser.parse_args()

    extract_from_human_replication_study(args.reference_report_path, args.study_path)


if __name__ == "__main__":
    extract_human_replication_info()
