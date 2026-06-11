"""
LLM_Benchmarking__
|
extract_human_replication_info.py
Created on Wed Jun 18 00:32:19 2025
@author: Rochana Obadage
"""

import argparse
import json

from validator.extract_from_human_replication_study import extract_from_human_replication_study


def extract_human_replication_info():
    parser = argparse.ArgumentParser(description="Validator: Extract expected replication info from SCORE reports")
    parser.add_argument('--preregistration', type=str, required=True, help='Path to PDF or DOCX pre-registration document')
    parser.add_argument('--score_report', type=str, required=True, help='Path to PDF or DOCX SCORE report')
    parser.add_argument('--output_path', type=str, required=True, help='Path to save the expected replication_info.json')
    args = parser.parse_args()

    extract_from_human_replication_study(args.preregistration, args.score_report, args.output_path)


if __name__ == "__main__":
    extract_human_replication_info()
