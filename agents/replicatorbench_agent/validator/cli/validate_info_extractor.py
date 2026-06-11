"""
LLM_Benchmarking__
|
validate_info_extractor.py
Created on Wed Jun 18 00:32:19 2025
@author: Rochana Obadage
"""

import os
import argparse
import json

from validator.compare_outputs import validate_study


def validate_info_extractor():
    parser = argparse.ArgumentParser(description="Validator: Validate extracted vs expected from info_extractor")
    parser.add_argument('--study_dir', type=str, required=True, help='Path to folder with both replication_info.json and replication_info_expected.json')
    parser.add_argument('--results_file', type=str, required=True, help='File name to store validation results [ex:info_exractor_validation_results.json]')
    parser.add_argument('--show_mismatches', action='store_true', help='If set, will print mismatches to console')
    args = parser.parse_args()
    
    extracted_path = f"{args.study_dir}/replication_info.json"
    expected_path = f"{args.study_dir}/replication_info_expected.json"
    result = validate_study(extracted_path, expected_path)

    results_full_path = os.path.join(args.study_dir, args.results_file)

    with open(results_full_path, 'w') as f:
        json.dump(result, f, indent=2)
        print(f"\nValidation results saved to: {results_full_path}")

    if args.show_mismatches:
        print("=== MATCHES ===")
        for match in result['matches']:
            print(match)
        
        print("\n=== MISMATCHES ===")
        for mismatch in result['mismatches']:
            print(json.dumps(mismatch, indent=2))


if __name__ == "__main__":
    validate_info_extractor()
