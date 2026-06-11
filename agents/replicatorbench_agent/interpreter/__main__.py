# unified CLI
import argparse, os, sys

def main():
    p = argparse.ArgumentParser("interpret")
    p.add_argument("--tier", choices=["easy", "medium", "hard"], default="easy")
    p.add_argument("--study-path", required=True)
    p.add_argument("--templates-dir", default="./templates")
    p.add_argument("--show-prompt", action="store_true", default=False)
    p.add_argument("--model-name", help="Please specify the OpenAI model to use.")
    args = p.parse_args()

    if args.tier == "easy":
        from interpreter.agent import run_interpret
        # run the agent that generates the plan / prereg JSON
        run_interpret(args.study_path, show_prompt=args.show_prompt, tier=args.tier, model=args.model_name)
    else:
        sys.exit(f"Stage/tier not implemented yet: {args.stage}/{args.tier}")

if __name__ == "__main__":
    main()
