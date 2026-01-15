import json
import asyncio
from functools import partial
from typing import Dict, Any, Optional, List
import os
import time

# Reliability metric utilities
from hal.utils.fault_injection import FaultInjector, FaultEvent
from hal.utils.compliance_checkers import ComplianceMonitor, ComplianceViolation
from hal.utils.structural_perturbations import (
    StructuralPerturbator,
    PerturbationType,
    PerturbationStrength,
    PerturbationConfig
)
from hal.utils.llm_log_analyzer import LLMLogAnalyzer


def run(input: dict[str, dict], **kwargs) -> dict[str, str]:

    assert 'model_name' in kwargs, 'model_name is required'
    assert 'provider' in kwargs, 'provider is required. choose from openai or anthropic'
    task_id = list(input.keys())[0]

    import litellm
    litellm.drop_params = True

    # ========== RELIABILITY METRICS INITIALIZATION ==========

    # Initialize FaultInjector if enabled
    fault_injector: Optional[FaultInjector] = None
    if kwargs.get('enable_fault_injection') == 'true' or kwargs.get('enable_fault_injection') is True:
        fault_rate = float(kwargs.get('fault_rate', 0.2))
        fault_injector = FaultInjector(fault_rate=fault_rate)
        print(f"🔧 Fault injection enabled with rate={fault_rate}")

    # Initialize ComplianceMonitor if enabled
    compliance_monitor: Optional[ComplianceMonitor] = None
    if kwargs.get('enable_compliance_monitoring') == 'true' or kwargs.get('enable_compliance_monitoring') is True:
        constraints_input = kwargs.get('compliance_constraints', 'no_pii_exposure,no_destructive_ops')
        # Handle both string and list inputs
        if isinstance(constraints_input, list):
            constraints = [c.strip() for c in constraints_input if c.strip()]
        else:
            constraints = [c.strip() for c in str(constraints_input).split(',') if c.strip()]
        compliance_monitor = ComplianceMonitor(constraints=constraints)
        print(f"📋 Compliance monitoring enabled with constraints: {constraints}")

    # Initialize StructuralPerturbator if enabled
    perturbator: Optional[StructuralPerturbator] = None
    if kwargs.get('enable_structural_perturbations') == 'true' or kwargs.get('enable_structural_perturbations') is True:
        perturbation_type = kwargs.get('perturbation_type', 'all')
        perturbation_strength = kwargs.get('perturbation_strength', 'medium')

        # Get preset config based on strength
        try:
            strength_enum = PerturbationStrength(perturbation_strength)
            config = PerturbationConfig.get_preset(strength_enum)
        except ValueError:
            config = PerturbationConfig()

        perturbator = StructuralPerturbator(
            perturbation_type=perturbation_type,
            config=config
        )
        print(f"🔀 Structural perturbations enabled: type={perturbation_type}, strength={perturbation_strength}")

    # Separate providers for user simulation vs agent
    # User simulation needs OpenAI-compatible API (for tau-bench internals)
    env_provider = 'openai'

    # Determine provider configuration based on model_name prefix
    if 'openrouter/' in kwargs['model_name']:
        # Use OpenRouter - strip the openrouter/ prefix
        agent_provider = 'openai'
        api_base = "https://openrouter.ai/api/v1"
        api_key = os.getenv('OPENROUTER_API_KEY')
        model_name = kwargs['model_name'].replace('openrouter/', '')  # Remove openrouter/ prefix

        # Configure litellm for OpenRouter
        if api_key:
            os.environ['OPENROUTER_API_KEY'] = api_key
    elif 'together_ai' in kwargs['model_name']:
        # Use Together AI via OpenAI-compatible API
        agent_provider = 'openai'
        api_base = "https://api.together.xyz/v1"
        api_key = os.getenv('TOGETHERAI_API_KEY')
        model_name = kwargs['model_name'].replace('together_ai/', '')
    elif 'gemini' in kwargs['model_name']:
        # Use Google Gemini via OpenAI-compatible API
        agent_provider = 'openai'
        api_base = "https://generativelanguage.googleapis.com/v1beta/openai/"
        api_key = os.getenv('GEMINI_API_KEY')
        model_name = kwargs['model_name'].replace('gemini/', '')
    elif 'claude' in kwargs['model_name']:
        # Use Anthropic directly
        agent_provider = 'anthropic'
        api_base = None  # litellm handles Anthropic API endpoint automatically
        api_key = os.getenv('ANTHROPIC_API_KEY')
        model_name = kwargs['model_name']  # Use the full model name
    else:
        # Default to provider parameter
        agent_provider = kwargs['provider']
        api_base = None
        api_key = None
        model_name = kwargs['model_name']

    # Store the original completion functions
    original_completion = litellm.completion
    original_acompletion = litellm.acompletion

    # Create a wrapper that adds reasoning parameters and OpenRouter configuration
    def completion_with_reasoning(*args, **completion_kwargs):
        # Check if this is a call with our agent's model
        if 'model' in completion_kwargs and completion_kwargs['model'] == model_name:
            # Add custom API base URL, API key, and headers for our agent's model
            if api_base:
                completion_kwargs['api_base'] = api_base
                completion_kwargs['api_key'] = api_key
                extra_headers = completion_kwargs.get('extra_headers', {})
                extra_headers['HTTP-Referer'] = 'https://github.com/benediktstroebl/hal-harness'
                extra_headers['X-Title'] = 'HAL Harness'
                completion_kwargs['extra_headers'] = extra_headers
            if 'reasoning_effort' in kwargs:
                # Set temperature to 1 for reasoning calls
                completion_kwargs['temperature'] = 1.0

                if api_base:  # Using OpenRouter
                    # For OpenRouter, add reasoning to extra_body
                    effort_to_tokens = {
                        'low': 1024,
                        'medium': 2048,
                        'high': 4096
                    }
                    reasoning_tokens = effort_to_tokens.get(kwargs['reasoning_effort'], 4096)
                    extra_body = completion_kwargs.get('extra_body', {})
                    extra_body['reasoning'] = {"max_tokens": reasoning_tokens}
                    extra_body['include_reasoning'] = True
                    completion_kwargs['extra_body'] = extra_body
                    print(f"Setting reasoning tokens to {reasoning_tokens} for OpenRouter model {model_name}")
                else:
                    # For direct OpenAI
                    completion_kwargs['reasoning_effort'] = kwargs['reasoning_effort']
                    print(f"Setting reasoning_effort to {kwargs['reasoning_effort']} for model {model_name}")

        # Call the original function - with fault injection if enabled
        if fault_injector and fault_injector.enabled:
            try:
                response = fault_injector.wrap_call(original_completion, *args, **completion_kwargs)
            except Exception as fault_error:
                # Log the fault and re-raise
                print(f"⚡ Fault injected: {type(fault_error).__name__}: {fault_error}")
                raise
        else:
            response = original_completion(*args, **completion_kwargs)

        # Ensure response_cost is set (litellm may not calculate it for custom api_base)
        if hasattr(response, '_hidden_params') and response._hidden_params.get('response_cost') is None:
            response._hidden_params['response_cost'] = 0.0

        # Fix empty tool call arguments (OpenRouter/Claude compatibility issue)
        if hasattr(response, 'choices') and response.choices:
            message = response.choices[0].message
            if hasattr(message, 'tool_calls') and message.tool_calls:
                for tool_call in message.tool_calls:
                    if hasattr(tool_call, 'function') and tool_call.function:
                        if not tool_call.function.arguments or tool_call.function.arguments.strip() == "":
                            tool_call.function.arguments = "{}"

        return response

    # Create async wrapper
    async def acompletion_with_reasoning(*args, **completion_kwargs):
        # Check if this is a call with our agent's model
        if 'model' in completion_kwargs and completion_kwargs['model'] == model_name:
            # Add custom API base URL, API key, and headers for our agent's model
            if api_base:
                completion_kwargs['api_base'] = api_base
                completion_kwargs['api_key'] = api_key
                extra_headers = completion_kwargs.get('extra_headers', {})
                extra_headers['HTTP-Referer'] = 'https://github.com/benediktstroebl/hal-harness'
                extra_headers['X-Title'] = 'HAL Harness'
                completion_kwargs['extra_headers'] = extra_headers
            if 'reasoning_effort' in kwargs:
                # Set temperature to 1 for reasoning calls
                completion_kwargs['temperature'] = 1.0

                if api_base:  # Using OpenRouter
                    # For OpenRouter, add reasoning to extra_body
                    effort_to_tokens = {'low': 1024, 'medium': 2048, 'high': 4096}
                    reasoning_tokens = effort_to_tokens.get(kwargs['reasoning_effort'], 4096)
                    extra_body = completion_kwargs.get('extra_body', {})
                    extra_body['reasoning'] = {"max_tokens": reasoning_tokens}
                    extra_body['include_reasoning'] = True
                    completion_kwargs['extra_body'] = extra_body
                    print(f"Setting reasoning tokens to {reasoning_tokens} for OpenRouter model {model_name}")
                else:
                    # For direct OpenAI
                    completion_kwargs['reasoning_effort'] = kwargs['reasoning_effort']
                    print(f"Setting reasoning_effort to {kwargs['reasoning_effort']} for model {model_name}")

        # Call the original function - with fault injection if enabled
        # Note: For async, we don't use wrap_call (which is sync). Instead we inject
        # faults manually to maintain async behavior.
        if fault_injector and fault_injector.enabled:
            import random
            if random.random() < fault_injector.fault_rate:
                # Inject fault
                fault_type = fault_injector._select_fault_type()
                fault_injector.state['faults_injected'] += 1
                print(f"⚡ Async fault injected: {fault_type.value}")

                # Attempt recovery with retries
                max_retries = fault_injector.config.get('max_recovery_attempts', 3)
                recovered = False
                recovery_start = time.time()

                for attempt in range(max_retries):
                    recovery_prob = 0.3 + (attempt * 0.2)
                    if random.random() < recovery_prob:
                        response = await original_acompletion(*args, **completion_kwargs)
                        recovered = True
                        fault_injector.state['recoveries_successful'] += 1
                        break
                    await asyncio.sleep(0.1 * (attempt + 1))

                if not recovered:
                    fault_injector.state['recoveries_failed'] += 1
                    # Raise fault exception - recovery failed after all attempts
                    raise RuntimeError(f"Simulated fault after {max_retries} recovery attempts: {fault_type.value}")

                recovery_time = time.time() - recovery_start
                fault_injector.state['total_recovery_time'] += recovery_time

                # Log fault event
                from hal.utils.fault_injection import FaultEvent
                fault_event = FaultEvent(
                    fault_type=fault_type,
                    recovered=recovered,
                    recovery_time=recovery_time,
                    context={'recovery_attempts': attempt + 1, 'function_name': 'acompletion'}
                )
                fault_injector.fault_events.append(fault_event)
            else:
                response = await original_acompletion(*args, **completion_kwargs)
        else:
            response = await original_acompletion(*args, **completion_kwargs)

        # Ensure response_cost is set (litellm may not calculate it for custom api_base)
        if hasattr(response, '_hidden_params') and response._hidden_params.get('response_cost') is None:
            response._hidden_params['response_cost'] = 0.0

        # Fix empty tool call arguments (OpenRouter/Claude compatibility issue)
        if hasattr(response, 'choices') and response.choices:
            message = response.choices[0].message
            if hasattr(message, 'tool_calls') and message.tool_calls:
                for tool_call in message.tool_calls:
                    if hasattr(tool_call, 'function') and tool_call.function:
                        if not tool_call.function.arguments or tool_call.function.arguments.strip() == "":
                            tool_call.function.arguments = "{}"

        return response

    # Replace both sync and async completion functions
    litellm.completion = completion_with_reasoning
    litellm.acompletion = acompletion_with_reasoning

    from tau_bench.envs import get_env
    from tau_bench.agents.tool_calling_agent import ToolCallingAgent

    ### ENV SETUP (usually this should be untouched) ###
    isolated_env = get_env(
        input[task_id]['env'],
        input[task_id]['user_strategy'],
        input[task_id]['user_model'],
        input[task_id]['task_split'],
        env_provider,  # Use env_provider for user simulation (always OpenAI-compatible)
        input[task_id]['task_index']
    )

    # Support for prompt sensitivity: Override instruction if provided
    if 'instruction' in input[task_id] and input[task_id]['instruction'] is not None:
        # Override the task instruction with the provided one (for prompt sensitivity evaluation)
        original_instruction = isolated_env.task.instruction
        isolated_env.task.instruction = input[task_id]['instruction']
        print(f"🔀 Using custom instruction for task {task_id}")
        print(f"   Original length: {len(original_instruction)} chars")
        print(f"   Variation length: {len(input[task_id]['instruction'])} chars")

    # Support for prompt variation style: Inject style directive into user's system prompt
    # This ensures the simulated user communicates in the specified style (casual, naturalistic, etc.)
    if 'prompt_variation_strength' in input[task_id]:
        from hal.utils.prompt_variation import get_user_style_directive
        style_strength = input[task_id]['prompt_variation_strength']
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
            if hasattr(isolated_env.user, 'messages') and isolated_env.user.messages:
                for msg in isolated_env.user.messages:
                    if msg.get('role') == 'system':
                        msg['content'] = msg['content'] + style_directive
                        break

            print(f"🎭 User communication style set to: {style_strength}")

    ### STRUCTURAL PERTURBATIONS (OPTIONAL) ###
    # Apply perturbations to the wiki/tools info to test agent robustness
    perturbed_tools_info = isolated_env.tools_info
    perturbed_wiki = isolated_env.wiki

    if perturbator:
        # Perturb the wiki (knowledge base) - apply data format changes
        if isinstance(perturbed_wiki, str):
            # Try parsing as JSON and perturbing
            try:
                wiki_data = json.loads(perturbed_wiki)
                if isinstance(wiki_data, dict):
                    wiki_data = perturbator.perturb_api_response(wiki_data)
                    perturbed_wiki = json.dumps(wiki_data)
            except (json.JSONDecodeError, TypeError):
                pass  # Keep original if not JSON

        # Perturb tools_info if it's a list of tool definitions
        # NOTE: We only perturb the 'description' field within tools, NOT the structural
        # schema keys (type, function, name, parameters) which are required by the API
        if isinstance(perturbed_tools_info, list):
            perturbed_tools = []
            for tool in perturbed_tools_info:
                if isinstance(tool, dict):
                    # Deep copy to avoid modifying original
                    perturbed_tool = json.loads(json.dumps(tool))

                    # Only perturb safe fields - descriptions and parameter descriptions
                    # Do NOT perturb: type, function, name, parameters.type, parameters.required
                    if 'function' in perturbed_tool and isinstance(perturbed_tool['function'], dict):
                        func = perturbed_tool['function']

                        # Perturb function description (safe)
                        if 'description' in func and isinstance(func['description'], str):
                            # Apply data format perturbations to description text
                            func['description'] = perturbator.perturb_data(func['description'])

                        # Perturb parameter descriptions only (not names or structure)
                        if 'parameters' in func and isinstance(func['parameters'], dict):
                            params = func['parameters']
                            if 'properties' in params and isinstance(params['properties'], dict):
                                for prop_name, prop_def in params['properties'].items():
                                    if isinstance(prop_def, dict) and 'description' in prop_def:
                                        prop_def['description'] = perturbator.perturb_data(prop_def['description'])

                    perturbed_tools.append(perturbed_tool)
                else:
                    perturbed_tools.append(tool)
            perturbed_tools_info = perturbed_tools

        print(f"🔀 Applied {len(perturbator.applied_perturbations)} structural perturbations")

    ### YOUR AGENT CODE HERE ###
    agent = ToolCallingAgent(
        tools_info=perturbed_tools_info,
        wiki=perturbed_wiki,
        model=model_name,
        provider=agent_provider,  # Use agent_provider for the agent (can be anthropic)
        temperature=kwargs['temperature'] if 'temperature' in kwargs else 0.0
    )

    output = agent.solve(isolated_env, task_index=input[task_id]['task_index'])

    ### COMPUTE CONFIDENCE (OPTIONAL) ###
    confidence = None
    confidence_details = None
    compute_confidence = kwargs.get('compute_confidence', False)

    if compute_confidence:
        confidence, confidence_details = _compute_confidence_score(
            model_name=model_name,
            provider=agent_provider,
            api_base=api_base,
            api_key=api_key,
            task_description=isolated_env.task.model_dump(),
            conversation_history=output.messages,
            reward=isolated_env.reward,
            actions_taken=isolated_env.actions
        )

    ### COMPLIANCE CHECKING (OPTIONAL) ###
    compliance_violations = []
    if compliance_monitor:
        # Check agent output for compliance violations
        task_output_str = json.dumps(isolated_env.task.model_dump())
        actions_str = json.dumps([action.model_dump() for action in isolated_env.actions])

        # Check for PII exposure in output
        if 'no_pii_exposure' in compliance_monitor.constraints:
            passed, violation = compliance_monitor.check_constraint(
                'no_pii_exposure',
                text=task_output_str,
                log_output=actions_str
            )
            if not passed and violation:
                compliance_violations.append(violation.to_dict())
                print(f"⚠️  PII violation detected: {violation.description}")

        # Check for destructive operations
        if 'no_destructive_ops' in compliance_monitor.constraints:
            for action in isolated_env.actions:
                action_str = str(action.model_dump()) if hasattr(action, 'model_dump') else str(action)
                passed, violation = compliance_monitor.check_constraint(
                    'no_destructive_ops',
                    operation=action_str
                )
                if not passed and violation:
                    compliance_violations.append(violation.to_dict())
                    print(f"⚠️  Destructive op violation: {violation.description}")

        print(f"📋 Compliance check complete: {len(compliance_violations)} violations found")

    ### LLM-BASED LOG ANALYSIS (OPTIONAL) ###
    llm_compliance_result = None
    llm_recovery_result = None

    enable_llm_analysis = kwargs.get('enable_llm_analysis') == 'true' or kwargs.get('enable_llm_analysis') is True
    if enable_llm_analysis:
        llm_analysis_model = kwargs.get('llm_analysis_model', 'gpt-4o-mini')
        print(f"🔍 Running LLM-based log analysis with {llm_analysis_model}...")

        try:
            llm_analyzer = LLMLogAnalyzer(model=llm_analysis_model, cache_responses=False)

            # Prepare trace data
            conversation_history = output.messages if hasattr(output, 'messages') else []
            actions_list = [action.model_dump() for action in isolated_env.actions]

            # LLM-based compliance analysis
            if kwargs.get('llm_compliance') != 'false':
                llm_constraints = kwargs.get('llm_constraints', 'no_pii_exposure,no_destructive_ops').split(',')
                llm_constraints = [c.strip() for c in llm_constraints if c.strip()]

                llm_compliance_result = llm_analyzer.analyze_compliance(
                    conversation_history=conversation_history,
                    actions_taken=actions_list,
                    constraints=llm_constraints
                )
                print(f"   LLM Compliance: S_comp={llm_compliance_result.S_comp:.2f}, violations={len(llm_compliance_result.violations)}")

            # LLM-based recovery detection
            if kwargs.get('llm_recovery') != 'false':
                llm_recovery_result = llm_analyzer.detect_recovery_behavior(
                    conversation_history=conversation_history,
                    actions_taken=actions_list
                )
                print(f"   LLM Recovery: V_heal={llm_recovery_result.V_heal:.2f}, errors={llm_recovery_result.total_errors_encountered}")

        except Exception as e:
            print(f"⚠️  LLM analysis error: {e}")

    ### WHEN DONE WE RETURN THE ENV STATE ###
    result = {
        task_id: {
            "reward": isolated_env.reward,
            "taken_actions": [action.model_dump() for action in isolated_env.actions],
            "task": isolated_env.task.model_dump()
        }
    }

    # Add confidence if computed
    if confidence is not None:
        result[task_id]["confidence"] = confidence

    # Add confidence details for traceability (optional, controlled by flag)
    if confidence_details is not None and kwargs.get('store_confidence_details', False):
        result[task_id]["confidence_details"] = confidence_details

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
            "V_heal": fault_stats['recovery_rate'],
            "mean_recovery_time": fault_stats['mean_recovery_time']
        }
        print(f"🔧 Fault stats: {fault_stats['total_faults_injected']} faults, "
              f"{fault_stats['recovery_rate']:.1%} recovery rate")

    # Add compliance metrics if enabled
    if compliance_monitor:
        result[task_id]["compliance"] = {
            "enabled": True,
            "constraints": compliance_monitor.constraints,
            "violations": compliance_violations,
            "violation_count": len(compliance_violations),
            "S_comp": 1.0 - (1.0 if compliance_violations else 0.0)  # Per-task compliance
        }

    # Add structural perturbation metrics if enabled
    if perturbator:
        result[task_id]["structural_perturbation"] = {
            "enabled": True,
            "perturbation_type": perturbator.perturbation_type.value,
            "perturbation_count": len(perturbator.applied_perturbations),
            "applied_perturbations": perturbator.applied_perturbations,
            "config": {
                "api_parameter_case": perturbator.config.api_parameter_case,
                "api_endpoint_style": perturbator.config.api_endpoint_style,
                "api_response_wrapper": perturbator.config.api_response_wrapper,
                "db_column_naming": perturbator.config.db_column_naming,
                "date_format": perturbator.config.date_format,
            }
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
            "recovery_attempts": [r.to_dict() for r in llm_recovery_result.recovery_attempts],
            "analysis_model": llm_recovery_result.analysis_model,
        }

    # Store conversation history for post-hoc analysis if requested
    if kwargs.get('store_conversation_history') == 'true' or kwargs.get('store_conversation_history') is True:
        conversation_history = output.messages if hasattr(output, 'messages') else []
        result[task_id]["conversation_history"] = conversation_history

    return result


def _compute_confidence_score(
    model_name: str,
    provider: str,
    api_base: str,
    api_key: str,
    task_description: dict,
    conversation_history: list,
    reward: float,
    actions_taken: list
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
    task_instruction = task_description.get('instruction', 'Unknown task')

    # Count errors and retries from actions
    num_errors = sum(1 for action in actions_taken if hasattr(action, 'error') and action.error)
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
        action_name = action.name if hasattr(action, 'name') else str(action)
        action_sequence.append(f"  {i}. {action_name}")

    if action_sequence:
        summary_parts.append(f"\nRecent actions:")
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
    
    confidence_prompt = f"""You just completed a task. Please assess your confidence in the correctness of your solution.

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
        # APPROACH: Continue the existing conversation (best context, reasonable cost)
        #
        # Instead of starting fresh with just a summary, we append to the conversation
        # that was just used for the task. This gives the model full context about
        # what actually happened - it literally just executed this task!
        #
        # Cost is reasonable because:
        # - The conversation context is already "warm" (just used it)
        # - We're only adding one more short turn
        # - The model can give informed confidence based on actual execution

        if conversation_history and len(conversation_history) > 0:
            # Continue the existing conversation - the model just executed this task!
            confidence_messages = list(conversation_history)  # Copy the conversation

            # Append the confidence assessment request as a natural follow-up
            confidence_messages.append({
                "role": "user",
                "content": confidence_prompt
            })
        else:
            # Fallback: If somehow we don't have conversation history, use summary
            # (This shouldn't normally happen, but better safe than sorry)
            confidence_messages = [
                {
                    "role": "system",
                    "content": "You are a helpful assistant that provides accurate self-assessments of your task performance."
                },
                {
                    "role": "user",
                    "content": confidence_prompt
                }
            ]

        # Call the model for confidence assessment
        # IMPORTANT: Explicitly disable tools to force text response
        # Without this, models (especially Gemini) may try to call tools
        # or return None content when the conversation history contains tool calls
        kwargs_for_confidence = {
            'model': model_name,
            'messages': confidence_messages,
            'temperature': 0.0,
            'max_tokens': 10,
            'custom_llm_provider': provider,
            'tools': None,  # Disable tools to prevent tool calling
            'tool_choice': None  # Ensure no tool selection
        }

        # Add API base and key if using custom provider
        if api_base:
            kwargs_for_confidence['api_base'] = api_base
            kwargs_for_confidence['api_key'] = api_key
            kwargs_for_confidence['extra_headers'] = {
                'HTTP-Referer': 'https://github.com/benediktstroebl/hal-harness',
                'X-Title': 'HAL Harness - Confidence Assessment'
            }

        # Make the confidence assessment call
        # This will be automatically traced by Weave if it's active
        response = litellm.completion(**kwargs_for_confidence)

        # Extract and parse the confidence score
        content = response.choices[0].message.content
        if content is None:
            raise ValueError(f"Model returned None content. Response: {response}")
        confidence_text = content.strip()

        # Try to extract a number from the response
        import re
        numbers = re.findall(r'\d+', confidence_text)

        if numbers:
            confidence_score = float(numbers[0]) / 100.0  # Convert to [0, 1]
            confidence_score = max(0.0, min(1.0, confidence_score))  # Clamp to [0, 1]
            print(f"✓ Confidence assessment: Model returned '{confidence_text}' -> {confidence_score:.2f}")
        else:
            # Default to 0.5 if we can't parse
            print(f"⚠️  Warning: Could not parse confidence from '{confidence_text}', using default 0.5")
            confidence_score = 0.5

        # Create confidence details dict for storage
        confidence_details = {
            "prompt": confidence_prompt,
            "model_response": confidence_text,
            "parsed_score": confidence_score,
            "num_actions": num_actions,
            "num_errors": num_errors,
            "task_reward": reward,
            "model": model_name
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
            "fallback": True
        }

        return heuristic_confidence, heuristic_details
