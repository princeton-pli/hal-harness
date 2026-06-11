# unified CLI
import argparse, os, sys
from core.constants import DEFAULT_CODE_MODE, CODE_MODE_CHOICES

def main():
    p = argparse.ArgumentParser("generate")
    p.add_argument("--stage", choices=["design", "execute"], required=True)
    p.add_argument("--tier", choices=["easy", "medium", "hard"], default="easy")
    p.add_argument("--study-path", required=True)
    p.add_argument("--templates-dir", default="./templates")
    p.add_argument("--show-prompt", action="store_true", default=False)
    p.add_argument("--code-mode",choices=CODE_MODE_CHOICES,default=DEFAULT_CODE_MODE,help="Code execution mode: 'native' (run original language) or 'python' (translate all to Python and run Python).",)
    p.add_argument("--model-name", help="Please specify the OpenAI model to be used.")
    args = p.parse_args()

    if args.stage == "design":
        from generator.design_agent import run_design
        # run the agent that generates the plan / prereg JSON
        run_design(args.study_path,
        	show_prompt=args.show_prompt,
        	tier=args.tier,
        	code_mode=args.code_mode,
        	model_name=args.model_name
        )

    elif args.stage == "execute":
        # Agent-driven, step-by-step with human confirmation before executing
        from generator.execute_agent import run_execute
        run_execute(
            study_path=args.study_path,
            show_prompt=args.show_prompt,
            templates_dir=args.templates_dir,
            tier=args.tier,
            code_mode=args.code_mode,
            model_name=args.model_name
        )

    else:
        sys.exit(f"Stage/tier not implemented yet: {args.stage}/{args.tier}")

if __name__ == "__main__":
    main()
