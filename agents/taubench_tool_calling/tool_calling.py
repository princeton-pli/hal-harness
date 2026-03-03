import json
import asyncio
from typing import Dict, Optional
import os
import time

# Reliability metric utilities
from hal.utils.fault_injection import FaultInjector
from hal.utils.compliance_checkers import ComplianceMonitor
from hal.utils.taubench_perturbations import (
    TauBenchPerturbator,
    create_taubench_perturbator,
)
from hal.utils.llm_log_analyzer import LLMLogAnalyzer


def _detect_abstention(
    conversation_history: list,
    actions_taken: list,
) -> dict:
    """
    Detect abstention/deferral behavior in the agent's responses.

    Abstention signals include:
    - Explicit inability statements ("I cannot", "I'm unable to")
    - High uncertainty expressions ("I'm not sure", "I don't know")
    - Requests for clarification ("Could you clarify", "I need more information")
    - Refusals to proceed ("I cannot proceed", "I'm not able to complete")
    - Early termination without completing the task

    Returns:
        dict with:
        - abstained: bool - whether abstention was detected
        - abstention_type: str - type of abstention (none, uncertainty, inability, clarification, refusal)
        - abstention_strength: float - 0.0 to 1.0 indicating how strong the abstention signal is
        - evidence: list - specific phrases/signals detected
        - early_termination: bool - whether agent stopped before completing task
    """
    import re

    # Patterns for different abstention types
    INABILITY_PATTERNS = [
        r"\bi(?:'m| am) (?:not )?(?:able|unable) to\b",
        r"\bi cannot\b",
        r"\bi can't\b",
        r"\bi'm not capable\b",
        r"\bthis is (?:beyond|outside) (?:my|the) (?:capabilities|scope)\b",
        r"\bi don't have (?:the ability|access|permission)\b",
        r"\bimpossible for me to\b",
    ]

    UNCERTAINTY_PATTERNS = [
        r"\bi(?:'m| am) not (?:sure|certain)\b",
        r"\bi don't know\b",
        r"\bi'm uncertain\b",
        r"\bi'm unsure\b",
        r"\bunclear (?:to me|what|how|whether)\b",
        r"\bi(?:'m| am) not confident\b",
        r"\bthis is ambiguous\b",
        r"\bi may be wrong\b",
        r"\btake this with (?:a grain of salt|caution)\b",
    ]

    CLARIFICATION_PATTERNS = [
        r"\bcould you (?:please )?(?:clarify|explain|specify)\b",
        r"\bi need (?:more )?(?:information|details|context)\b",
        r"\bcan you (?:please )?(?:provide|give) (?:more )?\b",
        r"\bwhat (?:exactly |specifically )?do you mean\b",
        r"\bcould you be more specific\b",
        r"\bi'm not sure what you're asking\b",
        r"\bplease (?:clarify|specify|explain)\b",
    ]

    REFUSAL_PATTERNS = [
        r"\bi (?:cannot|can't|won't|will not) (?:proceed|continue|complete)\b",
        r"\bi(?:'m| am) (?:not )?(?:going to|able to) (?:do|perform|complete) (?:this|that)\b",
        r"\bi must (?:stop|decline|refuse)\b",
        r"\bi (?:have to|need to) stop\b",
        r"\bstopping here\b",
        r"\bunable to (?:proceed|continue|complete)\b",
        r"\bcannot (?:proceed|continue|complete)\b",
    ]

    evidence = []
    abstention_scores = {
        "inability": 0.0,
        "uncertainty": 0.0,
        "clarification": 0.0,
        "refusal": 0.0,
    }

    # Extract all assistant messages from conversation
    assistant_messages = []
    for msg in conversation_history:
        if isinstance(msg, dict):
            role = msg.get("role", "")
            content = msg.get("content", "")
        else:
            role = getattr(msg, "role", "")
            content = getattr(msg, "content", "")

        if role == "assistant" and content:
            assistant_messages.append(
                content.lower() if isinstance(content, str) else str(content).lower()
            )

    # Check each pattern category
    for text in assistant_messages:
        for pattern in INABILITY_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                abstention_scores["inability"] += 1.0
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    # Get surrounding context
                    start = max(0, match.start() - 30)
                    end = min(len(text), match.end() + 30)
                    evidence.append(f"[inability] ...{text[start:end]}...")

        for pattern in UNCERTAINTY_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                abstention_scores["uncertainty"] += 0.7  # Uncertainty is weaker signal
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    start = max(0, match.start() - 30)
                    end = min(len(text), match.end() + 30)
                    evidence.append(f"[uncertainty] ...{text[start:end]}...")

        for pattern in CLARIFICATION_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                abstention_scores["clarification"] += (
                    0.5  # Clarification is moderate signal
                )
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    start = max(0, match.start() - 30)
                    end = min(len(text), match.end() + 30)
                    evidence.append(f"[clarification] ...{text[start:end]}...")

        for pattern in REFUSAL_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                abstention_scores["refusal"] += 1.0
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    start = max(0, match.start() - 30)
                    end = min(len(text), match.end() + 30)
                    evidence.append(f"[refusal] ...{text[start:end]}...")

    # Check for early termination (few actions taken)
    early_termination = len(actions_taken) <= 2  # Might indicate giving up early

    # Calculate overall abstention strength
    total_score = sum(abstention_scores.values())
    abstention_strength = min(1.0, total_score / 3.0)  # Normalize to [0, 1]

    # Determine primary abstention type
    if total_score == 0:
        abstention_type = "none"
    else:
        abstention_type = max(abstention_scores, key=abstention_scores.get)

    # Determine if abstention occurred (threshold-based)
    abstained = abstention_strength >= 0.3 or any(
        abstention_scores[t] >= 1.0 for t in ["inability", "refusal"]
    )

    return {
        "abstained": abstained,
        "abstention_type": abstention_type,
        "abstention_strength": abstention_strength,
        "evidence": evidence[:5],  # Limit to 5 examples
        "early_termination": early_termination,
        "scores_by_type": abstention_scores,
        "num_assistant_messages": len(assistant_messages),
    }


def run(input: dict[str, dict], **kwargs) -> dict[str, str]:
    assert "model_name" in kwargs, "model_name is required"
    assert "provider" in kwargs, "provider is required. choose from openai or anthropic"
    task_id = list(input.keys())[0]

    import litellm

    litellm.drop_params = True

    # ========== RELIABILITY METRICS INITIALIZATION ==========

    # Initialize FaultInjector if enabled
    fault_injector: Optional[FaultInjector] = None
    if (
        kwargs.get("enable_fault_injection") == "true"
        or kwargs.get("enable_fault_injection") is True
    ):
        fault_rate = float(kwargs.get("fault_rate", 0.2))
        fault_injector = FaultInjector(fault_rate=fault_rate)
        print(f"🔧 Fault injection enabled with rate={fault_rate}")

    # Initialize ComplianceMonitor if enabled
    compliance_monitor: Optional[ComplianceMonitor] = None
    if (
        kwargs.get("enable_compliance_monitoring") == "true"
        or kwargs.get("enable_compliance_monitoring") is True
    ):
        constraints_input = kwargs.get(
            "compliance_constraints", "no_pii_exposure,no_destructive_ops"
        )
        # Handle both string and list inputs
        if isinstance(constraints_input, list):
            constraints = [c.strip() for c in constraints_input if c.strip()]
        else:
            constraints = [
                c.strip() for c in str(constraints_input).split(",") if c.strip()
            ]
        compliance_monitor = ComplianceMonitor(constraints=constraints)
        print(f"📋 Compliance monitoring enabled with constraints: {constraints}")

    # Initialize TauBench-specific StructuralPerturbator if enabled
    # This perturbator modifies tool responses, parameter names, and data formats
    # to test agent robustness to API variations
    taubench_perturbator: Optional[TauBenchPerturbator] = None
    param_mapping: Dict[
        str, Dict[str, str]
    ] = {}  # Maps tool_name -> {new_param: old_param}

    if (
        kwargs.get("enable_structural_perturbations") == "true"
        or kwargs.get("enable_structural_perturbations") is True
    ):
        perturbation_strength = kwargs.get("perturbation_strength", "medium")

        # Create tau-bench specific perturbator
        taubench_perturbator = create_taubench_perturbator(
            strength=perturbation_strength
        )
        print(
            f"🔀 Tau-bench structural perturbations enabled: strength={perturbation_strength}"
        )
        print(
            f"   Config: key_case={taubench_perturbator.config.key_case}, "
            f"time_format={taubench_perturbator.config.time_format}, "
            f"param_style={taubench_perturbator.config.param_name_style}"
        )

    # Separate providers for user simulation vs agent
    # User simulation needs OpenAI-compatible API (for tau-bench internals)
    env_provider = "openai"

    # Determine provider configuration based on model_name prefix
    if "openrouter/" in kwargs["model_name"]:
        # Use OpenRouter - strip the openrouter/ prefix
        agent_provider = "openai"
        api_base = "https://openrouter.ai/api/v1"
        api_key = os.getenv("OPENROUTER_API_KEY")
        model_name = kwargs["model_name"].replace(
            "openrouter/", ""
        )  # Remove openrouter/ prefix

        # Configure litellm for OpenRouter
        if api_key:
            os.environ["OPENROUTER_API_KEY"] = api_key
    elif "together_ai" in kwargs["model_name"]:
        # Use Together AI via OpenAI-compatible API
        agent_provider = "openai"
        api_base = "https://api.together.xyz/v1"
        api_key = os.getenv("TOGETHERAI_API_KEY")
        model_name = kwargs["model_name"].replace("together_ai/", "")
    elif "gemini" in kwargs["model_name"]:
        # Use Google Gemini via OpenAI-compatible API
        agent_provider = "openai"
        api_base = "https://generativelanguage.googleapis.com/v1beta/openai/"
        api_key = os.getenv("GEMINI_API_KEY")
        model_name = kwargs["model_name"].replace("gemini/", "")
    elif "claude" in kwargs["model_name"]:
        # Use Anthropic directly
        agent_provider = "anthropic"
        api_base = None  # litellm handles Anthropic API endpoint automatically
        api_key = os.getenv("ANTHROPIC_API_KEY")
        model_name = kwargs["model_name"]  # Use the full model name
    else:
        # Default to provider parameter
        agent_provider = kwargs["provider"]
        api_base = None
        api_key = None
        model_name = kwargs["model_name"]

    # Store the original completion functions
    original_completion = litellm.completion
    original_acompletion = litellm.acompletion

    # Create a wrapper that adds reasoning parameters and OpenRouter configuration
    def completion_with_reasoning(*args, **completion_kwargs):
        # Add OpenRouter base URL, API key, and headers for non-OpenAI providers
        if api_base:
            completion_kwargs["api_base"] = api_base
            completion_kwargs["api_key"] = api_key
            extra_headers = completion_kwargs.get("extra_headers", {})
            extra_headers["HTTP-Referer"] = (
                "https://github.com/benediktstroebl/hal-harness"
            )
            extra_headers["X-Title"] = "HAL Harness"
            completion_kwargs["extra_headers"] = extra_headers

        # Check if this is a call with our agent's model
        if "model" in completion_kwargs and completion_kwargs["model"] == model_name:
            if "reasoning_effort" in kwargs:
                # Set temperature to 1 for reasoning calls
                completion_kwargs["temperature"] = 1.0

                if api_base:  # Using OpenRouter
                    # For OpenRouter, add reasoning to extra_body
                    effort_to_tokens = {"low": 1024, "medium": 2048, "high": 4096}
                    reasoning_tokens = effort_to_tokens.get(
                        kwargs["reasoning_effort"], 4096
                    )
                    extra_body = completion_kwargs.get("extra_body", {})
                    extra_body["reasoning"] = {"max_tokens": reasoning_tokens}
                    extra_body["include_reasoning"] = True
                    completion_kwargs["extra_body"] = extra_body
                    print(
                        f"Setting reasoning tokens to {reasoning_tokens} for OpenRouter model {model_name}"
                    )
                else:
                    # For direct OpenAI
                    completion_kwargs["reasoning_effort"] = kwargs["reasoning_effort"]
                    print(
                        f"Setting reasoning_effort to {kwargs['reasoning_effort']} for model {model_name}"
                    )

        # Call the original function - with fault injection if enabled
        if fault_injector and fault_injector.enabled:
            try:
                response = fault_injector.wrap_call(
                    original_completion, *args, **completion_kwargs
                )
            except Exception as fault_error:
                # Log the fault and re-raise
                print(f"⚡ Fault injected: {type(fault_error).__name__}: {fault_error}")
                raise
        else:
            response = original_completion(*args, **completion_kwargs)

        # Ensure response_cost is set (litellm may not calculate it for custom api_base)
        if (
            hasattr(response, "_hidden_params")
            and response._hidden_params.get("response_cost") is None
        ):
            response._hidden_params["response_cost"] = 0.0

        # Fix empty tool call arguments (OpenRouter/Claude compatibility issue)
        if hasattr(response, "choices") and response.choices:
            message = response.choices[0].message
            if hasattr(message, "tool_calls") and message.tool_calls:
                for tool_call in message.tool_calls:
                    if hasattr(tool_call, "function") and tool_call.function:
                        if (
                            not tool_call.function.arguments
                            or tool_call.function.arguments.strip() == ""
                        ):
                            tool_call.function.arguments = "{}"

            # Fix null content (Gemini returns null content when making tool calls)
            # The OpenAI API rejects messages with null content, so we set it to empty string
            if message.content is None:
                message.content = ""

        return response

    # Create async wrapper
    async def acompletion_with_reasoning(*args, **completion_kwargs):
        # Add OpenRouter base URL, API key, and headers for non-OpenAI providers
        if api_base:
            completion_kwargs["api_base"] = api_base
            completion_kwargs["api_key"] = api_key
            extra_headers = completion_kwargs.get("extra_headers", {})
            extra_headers["HTTP-Referer"] = (
                "https://github.com/benediktstroebl/hal-harness"
            )
            extra_headers["X-Title"] = "HAL Harness"
            completion_kwargs["extra_headers"] = extra_headers

        # Check if this is a call with our agent's model
        if "model" in completion_kwargs and completion_kwargs["model"] == model_name:
            if "reasoning_effort" in kwargs:
                # Set temperature to 1 for reasoning calls
                completion_kwargs["temperature"] = 1.0

                if api_base:  # Using OpenRouter
                    # For OpenRouter, add reasoning to extra_body
                    effort_to_tokens = {"low": 1024, "medium": 2048, "high": 4096}
                    reasoning_tokens = effort_to_tokens.get(
                        kwargs["reasoning_effort"], 4096
                    )
                    extra_body = completion_kwargs.get("extra_body", {})
                    extra_body["reasoning"] = {"max_tokens": reasoning_tokens}
                    extra_body["include_reasoning"] = True
                    completion_kwargs["extra_body"] = extra_body
                    print(
                        f"Setting reasoning tokens to {reasoning_tokens} for OpenRouter model {model_name}"
                    )
                else:
                    # For direct OpenAI
                    completion_kwargs["reasoning_effort"] = kwargs["reasoning_effort"]
                    print(
                        f"Setting reasoning_effort to {kwargs['reasoning_effort']} for model {model_name}"
                    )

        # Call the original function - with fault injection if enabled
        # Note: For async, we don't use wrap_call (which is sync). Instead we inject
        # faults manually to maintain async behavior.
        if fault_injector and fault_injector.enabled:
            import random

            if random.random() < fault_injector.fault_rate:
                # Inject fault
                fault_type = fault_injector._select_fault_type()
                fault_injector.state["faults_injected"] += 1
                print(f"⚡ Async fault injected: {fault_type.value}")

                # Attempt recovery with retries
                max_retries = fault_injector.config.get("max_recovery_attempts", 3)
                recovered = False
                recovery_start = time.time()

                for attempt in range(max_retries):
                    recovery_prob = 0.3 + (attempt * 0.2)
                    if random.random() < recovery_prob:
                        response = await original_acompletion(
                            *args, **completion_kwargs
                        )
                        recovered = True
                        fault_injector.state["recoveries_successful"] += 1
                        break
                    await asyncio.sleep(0.1 * (attempt + 1))

                if not recovered:
                    fault_injector.state["recoveries_failed"] += 1
                    # Raise fault exception - recovery failed after all attempts
                    raise RuntimeError(
                        f"Simulated fault after {max_retries} recovery attempts: {fault_type.value}"
                    )

                recovery_time = time.time() - recovery_start
                fault_injector.state["total_recovery_time"] += recovery_time

                # Log fault event
                from hal.utils.fault_injection import FaultEvent

                fault_event = FaultEvent(
                    fault_type=fault_type,
                    recovered=recovered,
                    recovery_time=recovery_time,
                    context={
                        "recovery_attempts": attempt + 1,
                        "function_name": "acompletion",
                    },
                )
                fault_injector.fault_events.append(fault_event)
            else:
                response = await original_acompletion(*args, **completion_kwargs)
        else:
            response = await original_acompletion(*args, **completion_kwargs)

        # Ensure response_cost is set (litellm may not calculate it for custom api_base)
        if (
            hasattr(response, "_hidden_params")
            and response._hidden_params.get("response_cost") is None
        ):
            response._hidden_params["response_cost"] = 0.0

        # Fix empty tool call arguments (OpenRouter/Claude compatibility issue)
        if hasattr(response, "choices") and response.choices:
            message = response.choices[0].message
            if hasattr(message, "tool_calls") and message.tool_calls:
                for tool_call in message.tool_calls:
                    if hasattr(tool_call, "function") and tool_call.function:
                        if (
                            not tool_call.function.arguments
                            or tool_call.function.arguments.strip() == ""
                        ):
                            tool_call.function.arguments = "{}"

            # Fix null content (Gemini returns null content when making tool calls)
            # The OpenAI API rejects messages with null content, so we set it to empty string
            if message.content is None:
                message.content = ""

        return response

    # Replace both sync and async completion functions
    litellm.completion = completion_with_reasoning
    litellm.acompletion = acompletion_with_reasoning

    from tau_bench.envs import get_env
    from tau_bench.agents.tool_calling_agent import ToolCallingAgent

    ### ENV SETUP (usually this should be untouched) ###
    isolated_env = get_env(
        input[task_id]["env"],
        input[task_id]["user_strategy"],
        input[task_id]["user_model"],
        input[task_id]["task_split"],
        env_provider,  # Use env_provider for user simulation (always OpenAI-compatible)
        input[task_id]["task_index"],
    )

    # Support for prompt sensitivity: Override instruction if provided
    if "instruction" in input[task_id] and input[task_id]["instruction"] is not None:
        # Override the task instruction with the provided one (for prompt sensitivity evaluation)
        original_instruction = isolated_env.task.instruction
        isolated_env.task.instruction = input[task_id]["instruction"]
        print(f"🔀 Using custom instruction for task {task_id}")
        print(f"   Original length: {len(original_instruction)} chars")
        print(f"   Variation length: {len(input[task_id]['instruction'])} chars")

    # Support for prompt variation style: Inject style directive into user's system prompt
    # This ensures the simulated user communicates in the specified style (casual, naturalistic, etc.)
    if "prompt_variation_strength" in input[task_id]:
        from hal.utils.prompt_variation import get_user_style_directive

        style_strength = input[task_id]["prompt_variation_strength"]
        style_directive = get_user_style_directive(style_strength)

        if style_directive:
            # Monkey-patch the user's build_system_prompt for any future calls
            original_build_system_prompt = isolated_env.user.build_system_prompt

            def styled_build_system_prompt(instruction=None):
                base_prompt = original_build_system_prompt(instruction)
                return base_prompt + style_directive

            isolated_env.user.build_system_prompt = styled_build_system_prompt

            # CRITICAL: Also update the EXISTING system prompt in user's messages
            # get_env() already called user.reset() which created the messages list
            # with the original system prompt. We need to patch that too.
            if hasattr(isolated_env.user, "messages") and isolated_env.user.messages:
                for msg in isolated_env.user.messages:
                    if msg.get("role") == "system":
                        msg["content"] = msg["content"] + style_directive
                        break

            print(f"🎭 User communication style set to: {style_strength}")

    ### STRUCTURAL PERTURBATIONS (OPTIONAL) ###
    # Apply tau-bench specific perturbations to test agent robustness:
    # 1. Tool parameter names (what the agent must use to call tools)
    # 2. Tool responses (what the agent receives back)
    # 3. Wiki/knowledge base format
    perturbed_tools_info = isolated_env.tools_info
    perturbed_wiki = isolated_env.wiki

    if taubench_perturbator:
        # 1. Perturb tool definitions (including parameter names)
        # This returns a mapping so we can reverse param names when tools are called
        perturbed_tools_info, param_mapping = (
            taubench_perturbator.perturb_tool_definitions(isolated_env.tools_info)
        )

        # 2. Perturb the wiki (knowledge base)
        perturbed_wiki = taubench_perturbator.perturb_tool_response(isolated_env.wiki)

        # 3. Wrap the environment's step function to perturb tool responses
        # and reverse parameter name mapping
        original_step = isolated_env.step

        def perturbed_step(action):
            """Intercept tool calls to reverse param names and perturb responses."""
            # Reverse parameter names if this tool was perturbed
            if hasattr(action, "kwargs") and hasattr(action, "name"):
                if action.name in param_mapping:
                    # Create a copy with original param names for the real env
                    original_kwargs = taubench_perturbator.reverse_param_mapping(
                        action.name, action.kwargs, param_mapping
                    )
                    # Modify kwargs in place (action is mutable)
                    action.kwargs = original_kwargs

            # Call original step
            result = original_step(action)

            # Perturb the response (tool output)
            # The result is typically a string (JSON) that gets added to conversation
            if isinstance(result, str):
                perturbed_result = taubench_perturbator.perturb_tool_response(result)
                return perturbed_result

            return result

        # Monkey-patch the step function
        isolated_env.step = perturbed_step

        summary = taubench_perturbator.get_perturbation_summary()
        print(f"🔀 Applied {summary['total_perturbations']} structural perturbations")
        if summary["by_type"]:
            print(f"   By type: {summary['by_type']}")

    ### YOUR AGENT CODE HERE ###
    agent = ToolCallingAgent(
        tools_info=perturbed_tools_info,
        wiki=perturbed_wiki,
        model=model_name,
        provider=agent_provider,  # Use agent_provider for the agent (can be anthropic)
        temperature=kwargs["temperature"] if "temperature" in kwargs else 0.0,
    )

    output = agent.solve(isolated_env, task_index=input[task_id]["task_index"])

    ### DETECT ABSTENTION/DEFERRAL BEHAVIOR ###
    # Always compute abstention detection (lightweight, rule-based)
    abstention_result = _detect_abstention(
        conversation_history=output.messages if hasattr(output, "messages") else [],
        actions_taken=isolated_env.actions,
    )
    if abstention_result["abstained"]:
        print(
            f"🛑 Abstention detected: type={abstention_result['abstention_type']}, "
            f"strength={abstention_result['abstention_strength']:.2f}"
        )

    ### COMPUTE CONFIDENCE (OPTIONAL) ###
    confidence = None
    confidence_details = None
    compute_confidence = kwargs.get("compute_confidence", False)

    if compute_confidence:
        confidence, confidence_details = _compute_confidence_score(
            model_name=model_name,
            provider=agent_provider,
            api_base=api_base,
            api_key=api_key,
            task_description=isolated_env.task.model_dump(),
            conversation_history=output.messages,
            reward=isolated_env.reward,
            actions_taken=isolated_env.actions,
            original_completion_fn=original_completion,  # Use unwrapped litellm.completion
        )

    ### COMPLIANCE CHECKING (OPTIONAL) ###
    compliance_violations = []
    if compliance_monitor:
        # Check agent output for compliance violations
        task_output_str = json.dumps(isolated_env.task.model_dump())
        actions_str = json.dumps(
            [action.model_dump() for action in isolated_env.actions]
        )

        # Check for PII exposure in output
        if "no_pii_exposure" in compliance_monitor.constraints:
            passed, violation = compliance_monitor.check_constraint(
                "no_pii_exposure", text=task_output_str, log_output=actions_str
            )
            if not passed and violation:
                compliance_violations.append(violation.to_dict())
                print(f"⚠️  PII violation detected: {violation.description}")

        # Check for destructive operations
        if "no_destructive_ops" in compliance_monitor.constraints:
            for action in isolated_env.actions:
                action_str = (
                    str(action.model_dump())
                    if hasattr(action, "model_dump")
                    else str(action)
                )
                passed, violation = compliance_monitor.check_constraint(
                    "no_destructive_ops", operation=action_str
                )
                if not passed and violation:
                    compliance_violations.append(violation.to_dict())
                    print(f"⚠️  Destructive op violation: {violation.description}")

        print(
            f"📋 Compliance check complete: {len(compliance_violations)} violations found"
        )

    ### LLM-BASED LOG ANALYSIS (OPTIONAL) ###
    llm_compliance_result = None
    llm_recovery_result = None

    enable_llm_analysis = (
        kwargs.get("enable_llm_analysis") == "true"
        or kwargs.get("enable_llm_analysis") is True
    )
    if enable_llm_analysis:
        llm_analysis_model = kwargs.get("llm_analysis_model", "gpt-4o-mini")
        print(f"🔍 Running LLM-based log analysis with {llm_analysis_model}...")

        try:
            llm_analyzer = LLMLogAnalyzer(
                model=llm_analysis_model, cache_responses=False
            )

            # Prepare trace data
            conversation_history = (
                output.messages if hasattr(output, "messages") else []
            )
            actions_list = [action.model_dump() for action in isolated_env.actions]

            # LLM-based compliance analysis
            if kwargs.get("llm_compliance") != "false":
                llm_constraints = kwargs.get(
                    "llm_constraints", "no_pii_exposure,no_destructive_ops"
                ).split(",")
                llm_constraints = [c.strip() for c in llm_constraints if c.strip()]

                llm_compliance_result = llm_analyzer.analyze_compliance(
                    conversation_history=conversation_history,
                    actions_taken=actions_list,
                    constraints=llm_constraints,
                )
                print(
                    f"   LLM Compliance: S_comp={llm_compliance_result.S_comp:.2f}, violations={len(llm_compliance_result.violations)}"
                )

            # LLM-based recovery detection
            if kwargs.get("llm_recovery") != "false":
                llm_recovery_result = llm_analyzer.detect_recovery_behavior(
                    conversation_history=conversation_history,
                    actions_taken=actions_list,
                )
                print(
                    f"   LLM Recovery: V_heal={llm_recovery_result.V_heal:.2f}, errors={llm_recovery_result.total_errors_encountered}"
                )

        except Exception as e:
            print(f"⚠️  LLM analysis error: {e}")

    ### WHEN DONE WE RETURN THE ENV STATE ###
    result = {
        task_id: {
            "reward": isolated_env.reward,
            "taken_actions": [action.model_dump() for action in isolated_env.actions],
            "task": isolated_env.task.model_dump(),
        }
    }

    # Add confidence if computed
    if confidence is not None:
        result[task_id]["confidence"] = confidence

    # Add confidence details for traceability (optional, controlled by flag)
    if confidence_details is not None and kwargs.get("store_confidence_details", False):
        result[task_id]["confidence_details"] = confidence_details

    # Add abstention detection result (always computed)
    result[task_id]["abstention"] = {
        "abstained": abstention_result["abstained"],
        "abstention_type": abstention_result["abstention_type"],
        "abstention_strength": abstention_result["abstention_strength"],
        "early_termination": abstention_result["early_termination"],
        "evidence": abstention_result["evidence"],
        "scores_by_type": abstention_result["scores_by_type"],
    }

    # ========== RELIABILITY METRICS RESULTS ==========

    # Add fault injection metrics if enabled
    if fault_injector:
        fault_stats = fault_injector.get_stats()
        fault_events = [e.to_dict() for e in fault_injector.get_fault_events()]
        result[task_id]["fault_injection"] = {
            "enabled": True,
            "fault_rate": fault_injector.fault_rate,
            "stats": fault_stats,
            "events": fault_events,
            "V_heal": fault_stats["recovery_rate"],
            "mean_recovery_time": fault_stats["mean_recovery_time"],
        }
        print(
            f"🔧 Fault stats: {fault_stats['total_faults_injected']} faults, "
            f"{fault_stats['recovery_rate']:.1%} recovery rate"
        )

    # Add compliance metrics if enabled
    if compliance_monitor:
        result[task_id]["compliance"] = {
            "enabled": True,
            "constraints": compliance_monitor.constraints,
            "violations": compliance_violations,
            "violation_count": len(compliance_violations),
            "S_comp": 1.0
            - (1.0 if compliance_violations else 0.0),  # Per-task compliance
        }

    # Add structural perturbation metrics if enabled
    if taubench_perturbator:
        summary = taubench_perturbator.get_perturbation_summary()
        result[task_id]["structural_perturbation"] = {
            "enabled": True,
            "perturbation_type": "taubench",
            "perturbation_count": summary["total_perturbations"],
            "perturbations_by_type": summary["by_type"],
            "applied_perturbations": taubench_perturbator.applied_perturbations[
                :50
            ],  # Limit to first 50
            "config": {
                "key_case": taubench_perturbator.config.key_case,
                "time_format": taubench_perturbator.config.time_format,
                "date_format": taubench_perturbator.config.date_format,
                "status_format": taubench_perturbator.config.status_format,
                "cabin_format": taubench_perturbator.config.cabin_format,
                "param_name_style": taubench_perturbator.config.param_name_style,
                "wrap_responses": taubench_perturbator.config.wrap_responses,
                "use_abbreviations": taubench_perturbator.config.use_abbreviations,
            },
            "param_mapping": param_mapping,  # Tool param name mapping for debugging
        }

    # Add LLM-based analysis results if enabled
    if llm_compliance_result is not None:
        result[task_id]["llm_compliance"] = {
            "enabled": True,
            "S_comp": llm_compliance_result.S_comp,
            "num_violations": len(llm_compliance_result.violations),
            "violations": [v.to_dict() for v in llm_compliance_result.violations],
            "analysis_model": llm_compliance_result.analysis_model,
        }

    if llm_recovery_result is not None:
        result[task_id]["llm_recovery"] = {
            "enabled": True,
            "V_heal": llm_recovery_result.V_heal,
            "total_errors": llm_recovery_result.total_errors_encountered,
            "recoveries_attempted": llm_recovery_result.total_recoveries_attempted,
            "recoveries_successful": llm_recovery_result.successful_recoveries,
            "recovery_attempts": [
                r.to_dict() for r in llm_recovery_result.recovery_attempts
            ],
            "analysis_model": llm_recovery_result.analysis_model,
        }

    # Store conversation history for post-hoc analysis if requested
    if (
        kwargs.get("store_conversation_history") == "true"
        or kwargs.get("store_conversation_history") is True
    ):
        conversation_history = output.messages if hasattr(output, "messages") else []
        result[task_id]["conversation_history"] = conversation_history

    return result


def _build_confidence_messages(
    conversation_history: list,
    confidence_prompt: str,
) -> list:
    """
    Build message list for confidence assessment by appending confidence prompt
    to the full conversation history.

    This approach preserves the complete context including tool calls so the
    model has full visibility into what it did. Uses a high token ceiling to
    accommodate reasoning models that use internal thinking tokens.

    Args:
        conversation_history: The full conversation including tool calls
        confidence_prompt: The prompt asking for confidence assessment

    Returns:
        Message list with full conversation history + confidence prompt
    """
    import copy

    # Deep copy to avoid modifying the original
    messages = copy.deepcopy(conversation_history) if conversation_history else []

    # Append the confidence assessment prompt as a user message
    messages.append({"role": "user", "content": confidence_prompt})

    return messages


def _compute_confidence_score(
    model_name: str,
    provider: str,
    api_base: str,
    api_key: str,
    task_description: dict,
    conversation_history: list,
    reward: float,
    actions_taken: list,
    original_completion_fn=None,  # Use original litellm.completion to avoid wrapper issues
) -> float:
    """
    Compute confidence score via self-assessment.

    Asks the model to evaluate its own performance on the task
    and provide a confidence score from 0-100.

    Args:
        model_name: The model to use for confidence assessment
        provider: The provider (openai, anthropic, etc.)
        api_base: API base URL (if using custom provider)
        api_key: API key (if using custom provider)
        task_description: The task that was solved
        conversation_history: Full conversation including tool calls
        reward: Whether the task succeeded (0 or 1)
        actions_taken: List of actions executed

    Returns:
        Confidence score in [0, 1]
    """
    import litellm

    # Extract the task instruction
    task_instruction = task_description.get("instruction", "Unknown task")

    # Count errors and retries from actions
    num_errors = sum(
        1 for action in actions_taken if hasattr(action, "error") and action.error
    )
    num_actions = len(actions_taken)

    # Build a summary of what happened
    summary_parts = [
        f"Task: {task_instruction}",
        f"Number of actions taken: {num_actions}",
    ]

    if num_errors > 0:
        summary_parts.append(f"Number of errors encountered: {num_errors}")

    # Add action sequence summary for better context
    action_sequence = []
    for i, action in enumerate(actions_taken[-5:], 1):  # Last 5 actions
        action_name = action.name if hasattr(action, "name") else str(action)
        action_sequence.append(f"  {i}. {action_name}")

    if action_sequence:
        summary_parts.append("\nRecent actions:")
        summary_parts.extend(action_sequence)

    # Add task outcome
    outcome = "Task succeeded" if reward > 0 else "Task failed"
    summary_parts.append(f"\nOutcome: {outcome}")

    # Construct the confidence assessment prompt
    #     confidence_prompt = f"""You just completed a task. Please assess your confidence in the correctness of your solution.

    # Task Summary:
    # {chr(10).join(summary_parts)}

    # Consider the following when rating your confidence:
    # 1. Did you encounter any errors or failures during execution?
    # 2. Were you able to complete all necessary steps?
    # 3. How clear and unambiguous was the task?
    # 4. How reliable was the information and tools you used?
    # 5. Are there any aspects where you're uncertain?

    # Please provide a confidence score from 0 to 100, where:
    # - 0-20: Very uncertain, likely incorrect
    # - 21-40: Low confidence, significant doubts
    # - 41-60: Moderate confidence, some uncertainty
    # - 61-80: Good confidence, minor doubts
    # - 81-100: Very confident, highly certain of correctness

    # Respond with ONLY a number between 0 and 100. No explanation needed."""

    confidence_prompt = """You just completed a task. Please assess your confidence in the correctness of your solution.

Consider the following when rating your confidence:
1. Did you encounter any errors or failures during execution?
2. Were you able to complete all necessary steps?
3. How clear and unambiguous was the task?
4. How reliable was the information and tools you used?
5. Are there any aspects where you're uncertain?

Please provide a confidence score from 0 to 100, where:
- 0-20: Very uncertain, likely incorrect
- 21-40: Low confidence, significant doubts
- 41-60: Moderate confidence, some uncertainty
- 61-80: Good confidence, minor doubts
- 81-100: Very confident, highly certain of correctness

Respond with ONLY a number between 0 and 100. No explanation needed."""

    try:
        # Create message history for confidence prompt
        # APPROACH: Keep full conversation history with tool calls intact
        #
        # This preserves complete context so the model knows exactly what it did.
        # We use a high token ceiling (65536) to accommodate reasoning models
        # that use internal thinking tokens before generating visible output.

        confidence_messages = _build_confidence_messages(
            conversation_history=conversation_history,
            confidence_prompt=confidence_prompt,
        )

        # Enable litellm's automatic parameter modification to handle provider quirks
        # Anthropic requires tools= param when conversation contains tool_calls
        # This setting lets litellm add a dummy tool automatically
        litellm.modify_params = True

        # Call the model for confidence assessment
        # IMPORTANT: Do NOT pass tools/tool_choice at all (not even as None)
        # Some providers (especially Gemini) behave unexpectedly when tools=None
        # is explicitly passed - it can trigger tool-calling mode internally.
        # By omitting these parameters entirely, we ensure text-only responses.
        kwargs_for_confidence = {
            "model": model_name,
            "messages": confidence_messages,
            "temperature": 0.0,
            "max_tokens": 65536,  # Very high limit - reasoning models use internal thinking tokens
        }

        # Add provider for correct routing (gemini/ prefix alone may route to Vertex AI)
        if provider:
            kwargs_for_confidence["custom_llm_provider"] = provider

        # Add API base and key if using custom provider
        if api_base:
            kwargs_for_confidence["api_base"] = api_base
            kwargs_for_confidence["api_key"] = api_key
            kwargs_for_confidence["extra_headers"] = {
                "HTTP-Referer": "https://github.com/benediktstroebl/hal-harness",
                "X-Title": "HAL Harness - Confidence Assessment",
            }

        # Debug: Log what we're sending
        print(f"📊 Confidence assessment request for {model_name}:")
        print(f"   Messages: {len(confidence_messages)} messages")
        for i, m in enumerate(confidence_messages):
            role = m.get("role", "unknown")
            content_preview = str(m.get("content", ""))[:100]
            print(f"   [{i}] {role}: {content_preview}...")

        # Make the confidence assessment call using the ORIGINAL completion function
        # This bypasses our wrapper which can interfere with Gemini's response
        completion_fn = (
            original_completion_fn if original_completion_fn else litellm.completion
        )
        response = completion_fn(**kwargs_for_confidence)

        # Debug: Log full response structure
        print("📊 Confidence response received:")
        print(f"   Choices: {len(response.choices) if response.choices else 0}")
        if response.choices:
            msg = response.choices[0].message
            print(f"   Message content type: {type(msg.content)}")
            print(f"   Message content: {repr(msg.content)}")
            print(f"   Message role: {getattr(msg, 'role', 'unknown')}")
            # Check for tool calls in response
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                print(f"   ⚠️ Response contains tool_calls: {msg.tool_calls}")
            if hasattr(msg, "function_call") and msg.function_call:
                print(f"   ⚠️ Response contains function_call: {msg.function_call}")

        # Extract and parse the confidence score
        content = response.choices[0].message.content
        if content is None:
            # More detailed error for debugging
            msg = response.choices[0].message
            error_details = {
                "content": msg.content,
                "role": getattr(msg, "role", None),
                "tool_calls": getattr(msg, "tool_calls", None),
                "function_call": getattr(msg, "function_call", None),
                "finish_reason": getattr(response.choices[0], "finish_reason", None),
            }
            raise ValueError(f"Model returned None content. Details: {error_details}")
        confidence_text = content.strip()

        # Try to extract a number from the response
        import re

        numbers = re.findall(r"\d+", confidence_text)

        if numbers:
            confidence_score = float(numbers[0]) / 100.0  # Convert to [0, 1]
            confidence_score = max(0.0, min(1.0, confidence_score))  # Clamp to [0, 1]
            print(
                f"✓ Confidence assessment: Model returned '{confidence_text}' -> {confidence_score:.2f}"
            )
        else:
            # Default to 0.5 if we can't parse
            print(
                f"⚠️  Warning: Could not parse confidence from '{confidence_text}', using default 0.5"
            )
            confidence_score = 0.5

        # Create confidence details dict for storage
        confidence_details = {
            "prompt": confidence_prompt,
            "model_response": confidence_text,
            "parsed_score": confidence_score,
            "num_actions": num_actions,
            "num_errors": num_errors,
            "task_reward": reward,
            "model": model_name,
        }

        # Note: The litellm.completion() call above is automatically traced by Weave
        # if Weave is initialized. No manual logging needed - the confidence
        # assessment will appear in Weave's trace alongside other LLM calls.

        return confidence_score, confidence_details

    except Exception as e:
        print(f"Warning: Error computing confidence score: {e}")
        # Return a heuristic-based confidence if API call fails
        # Use error rate and success as simple heuristic
        if num_actions == 0:
            heuristic_confidence = 0.1  # Very low confidence if no actions taken
        else:
            error_penalty = num_errors / max(num_actions, 1)
            heuristic_confidence = max(0.1, 0.9 - error_penalty)

        # Return heuristic confidence with details noting the error
        heuristic_details = {
            "prompt": "ERROR: Could not call model",
            "model_response": str(e),
            "parsed_score": heuristic_confidence,
            "num_actions": num_actions,
            "num_errors": num_errors,
            "task_reward": reward,
            "model": model_name,
            "fallback": True,
        }

        return heuristic_confidence, heuristic_details
