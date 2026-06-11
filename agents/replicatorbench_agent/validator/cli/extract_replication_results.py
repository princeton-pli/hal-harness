from info_extractor.interpret.agent import run_extraction
import argparse
import logging
import sys # Needed for sys.stdout


def main():
    parser = argparse.ArgumentParser(description="LLM-based Replication Results Extractor")
    parser.add_argument('--study_path', required=True, help="Path to case study folder")
    parser.add_argument("--show-prompt", action="store_true", help="Print the generated prompt and exit")
    args = parser.parse_args()
   
    run_extraction(study_path=args.study_path, show_prompt=args.show_prompt)


if __name__ == "__main__":
    main()