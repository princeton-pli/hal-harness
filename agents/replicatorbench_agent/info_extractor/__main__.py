# info_extractor/__main__.py
import argparse
import os
import re
import sys
from core.utils import get_logger
from .extractor import run_extraction

def main():
    parser = argparse.ArgumentParser("extractor")
    parser.add_argument("--stage", choices=["stage_1", "web_search"], required=True)
    parser.add_argument("--difficulty", choices=["easy", "medium", "hard"], required=True)
    parser.add_argument("--study-path", required=True)
    parser.add_argument("--show-prompt", action="store_true", default=False)
    parser.add_argument("--model-name", help="Please specify the OpenAI model to use.")
    args = parser.parse_args()

    case_name = os.path.basename(os.path.normpath(args.study_path))
    if "case_study" not in case_name:
        match = re.search(r"case_study_\d+", args.study_path)
        if match:
            case_name = match.group()

    log_file_name = f"{case_name}_{args.stage}_extractor.log"
    os.environ["LOG_FILE"] = log_file_name

    logger, formatter = get_logger()
    logger.info(
        f"Running extraction for {args.study_path} at {args.difficulty} difficulty, stage={args.stage}"
    )

    try:
        run_extraction(
            study_path=args.study_path,
            difficulty=args.difficulty,
            stage=args.stage,
            model_name=args.model_name,
            show_prompt=args.show_prompt
        )
    except Exception as e:
        logger.exception(f"Fatal error during run_extraction: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

