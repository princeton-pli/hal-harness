"""
LLM_Benchmarking__
|
validator--|compare_outputs.py
Created on Wed Jun 18 00:32:19 2025
@author: Rochana Obadage
"""

import json
import difflib
import os

def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)

def compare_values(val1, val2, tolerance=0.75):
    if isinstance(val1, str) and isinstance(val2, str):
        ratio = difflib.SequenceMatcher(None, val1, val2).ratio()
        return ratio >= tolerance
    return val1 == val2

def validate_study(extracted_path, expected_path):
    result = {'matches': [], 'mismatches': []}
    extracted = load_json(extracted_path)
    expected = load_json(expected_path)

    def compare_recursive(prefix, val1, val2):
        if isinstance(val2, dict):
            if not isinstance(val1, dict):
                result['mismatches'].append({
                    'key': prefix,
                    'extracted': val1,
                    'expected': val2
                })
                return
            for subkey in val2:
                sub_prefix = f"{prefix}.{subkey}" if prefix else subkey
                compare_recursive(sub_prefix, val1.get(subkey), val2[subkey])
        else:
            if compare_values(val1, val2):
                result['matches'].append({
                    'key': prefix,
                    'extracted': val1,
                    'expected': val2
                })
            else:
                result['mismatches'].append({
                    'key': prefix,
                    'extracted': val1,
                    'expected': val2
                })

    for key in expected:
        compare_recursive(key, extracted.get(key), expected[key])

    return result
