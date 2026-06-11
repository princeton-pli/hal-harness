"""
evaluate_extracted_info.py
Extract and evaluate from human replication study at the same time. 
"""

import argparse
import json

from validator.evaluate_execute import run_evaluate_execute


def main():
    parser = argparse.ArgumentParser(description="Validator: Extract expected replication info from SCORE reports")
    parser.add_argument('--study_path', type=str, required=True, help='Path to save the evaluation info')
    args = parser.parse_args()

    run_evaluate_execute(args.study_path )


if __name__ == "__main__":
    main()
