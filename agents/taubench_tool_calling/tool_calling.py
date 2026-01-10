import json
from functools import partial
from typing import Dict, Any
import os

def run(input: dict[str, dict], **kwargs) -> dict[str, str]:

    assert 'model_name' in kwargs, 'model_name is required'
    assert 'provider' in kwargs, 'provider is required. choose from openai or anthropic'
    task_id = list(input.keys())[0]

    import litellm
    litellm.drop_params = True

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

        # Call the original function
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

        # Call the original function
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

    ### YOUR AGENT CODE HERE ###
    agent = ToolCallingAgent(
        tools_info=isolated_env.tools_info,
        wiki=isolated_env.wiki,
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
