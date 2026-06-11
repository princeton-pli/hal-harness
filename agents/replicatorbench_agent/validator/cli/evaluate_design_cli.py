"""
evaluate_design_cli.py
Evaluate Generate-design using human preregistration document.
"""

import argparse
import json

from validator.evaluate_replication_preregistration import extract_from_human_replication_study


def extract_human_replication_info():
    parser = argparse.ArgumentParser(description="Validator: Extract expected replication info from SCORE reports")
    parser.add_argument('--extracted_json_path', type=str, required=True, help='Path to the extracted replication_info.json')
    parser.add_argument('--reference_doc_path', type=str, required=True, help='Path to human preregistration document')
    parser.add_argument('--output_path', type=str, required=True, help='Path to save the evaluation info')
    args = parser.parse_args()

    extract_from_human_replication_study(args.extracted_json_path, args.reference_doc_path, args.output_path)


if __name__ == "__main__":
    extract_human_replication_info()
