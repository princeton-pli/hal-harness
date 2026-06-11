"""
evaluate_extracted_info.py
Extract and evaluate from human replication study at the same time. 
"""

import argparse
import json

from validator.extract_and_evaluate_from_human_rep import extract_from_human_replication_study


def extract_human_replication_info():
    parser = argparse.ArgumentParser(description="Validator: Extract expected replication info from SCORE reports")
    parser.add_argument('--extracted_json_path', type=str, required=True, help='Path to the extracted replication_info.json')
    parser.add_argument('--expected_json_path', type=str, required=True, help='Path to the extracted replication_info.json')
    parser.add_argument('--output_path', type=str, required=True, help='Path to save the evaluation info')
    args = parser.parse_args()

    extract_from_human_replication_study(args.extracted_json_path, args.expected_json_path, args.output_path )


if __name__ == "__main__":
    extract_human_replication_info()
