"""Evaluation configuration for reliability_eval.

Edit AGENT_CONFIGS to specify which models to evaluate.
Edit BENCHMARK_CONFIGS to specify which benchmarks to run on.
"""

from reliability_eval.constants import TAUBENCH_AIRLINE_CLEAN_TASKS

# =============================================================================
# AGENT CONFIGURATION
# =============================================================================

AGENT_CONFIGS = [
    # -------------------------------------------------------------------------
    # OpenAI Models
    # -------------------------------------------------------------------------
    # {
    #     "name": "taubench_toolcalling_gpt_4o_mini",
    #     "agent_dir": "agents/taubench_tool_calling",
    #     "agent_function": "tool_calling.run",
    #     "model_name": "gpt-4o-mini-2024-07-18",
    #     "provider": "openai",
    #     "benchmarks": ["taubench_airline"],
    # },
    # {
    #     "name": "taubench_toolcalling_gpt_4_turbo",
    #     "agent_dir": "agents/taubench_tool_calling",
    #     "agent_function": "tool_calling.run",
    #     "model_name": "gpt-4-turbo-2024-04-09",
    #     "provider": "openai",
    #     "benchmarks": ["taubench_airline"],
    # },
    # {
    #     "name": "taubench_toolcalling_gpt_o1",
    #     "agent_dir": "agents/taubench_tool_calling",
    #     "agent_function": "tool_calling.run",
    #     "model_name": "o1-2024-12-17",
    #     "provider": "openai",
    #     "benchmarks": ["taubench_airline"],
    # },
    # {
    #     "name": "taubench_toolcalling_gpt_5_2",
    #     "agent_dir": "agents/taubench_tool_calling",
    #     "agent_function": "tool_calling.run",
    #     "model_name": "gpt-5.2-2025-12-11",
    #     "provider": "openai",
    #     "benchmarks": ["taubench_airline"],
    # },
    # {
    #     "name": "taubench_toolcalling_gpt_5_2_xhigh",
    #     "agent_dir": "agents/taubench_tool_calling",
    #     "agent_function": "tool_calling.run",
    #     "model_name": "gpt-5.2-2025-12-11",
    #     "provider": "openai",
    #     "reasoning_effort": "xhigh",
    #     "benchmarks": ["taubench_airline"],
    # },
    # {
    #     "name": "taubench_toolcalling_gpt_5_4",
    #     "agent_dir": "agents/taubench_tool_calling",
    #     "agent_function": "tool_calling.run",
    #     "model_name": "gpt-5.4",
    #     "provider": "openai",
    #     "benchmarks": ["taubench_airline"],
    # },
    # {
    #     "name": "taubench_toolcalling_gpt_5_4_xhigh",
    #     "agent_dir": "agents/taubench_tool_calling",
    #     "agent_function": "tool_calling.run",
    #     "model_name": "gpt-5.4",
    #     "provider": "openai",
    #     "reasoning_effort": "xhigh",
    #     "benchmarks": ["taubench_airline"],
    # },
    # -------------------------------------------------------------------------
    # Anthropic Models
    # -------------------------------------------------------------------------
    # {
    #     "name": "taubench_toolcalling_claude_haiku_3_5",
    #     "agent_dir": "agents/taubench_tool_calling",
    #     "agent_function": "tool_calling.run",
    #     "model_name": "claude-3-5-haiku-20241022",
    #     "provider": "anthropic",
    #     "benchmarks": ["taubench_airline"],
    # },
    # {
    #     "name": "taubench_toolcalling_claude_sonnet_3_7",
    #     "agent_dir": "agents/taubench_tool_calling",
    #     "agent_function": "tool_calling.run",
    #     "model_name": "openrouter/anthropic/claude-3-7-sonnet-20250219",
    #     "benchmarks": ["taubench_airline", "taubench_retail"],
    #     "extra_agent_args": {
    #         "provider": "openai",  # OpenRouter uses OpenAI-compatible API
    #         "temperature": 0.0
    #     }
    # },
    # {
    #     "name": "taubench_toolcalling_claude_sonnet_4_5",
    #     "agent_dir": "agents/taubench_tool_calling",
    #     "agent_function": "tool_calling.run",
    #     "model_name": "claude-sonnet-4-5",
    #     "provider": "anthropic",
    #     "benchmarks": ["taubench_airline"],
    # },
    # {
    #     "name": "taubench_toolcalling_claude_opus_4_5",
    #     "agent_dir": "agents/taubench_tool_calling",
    #     "agent_function": "tool_calling.run",
    #     "model_name": "claude-opus-4-5",
    #     "provider": "anthropic",
    #     "benchmarks": ["taubench_airline"],
    # },
    # -------------------------------------------------------------------------
    # Google Gemini Models
    # -------------------------------------------------------------------------
    # {
    #     "name": "taubench_toolcalling_gemini_2_flash",
    #     "agent_dir": "agents/taubench_tool_calling",
    #     "agent_function": "tool_calling.run",
    #     "model_name": "gemini/gemini-2.0-flash",
    #     "provider": "google",
    #     "benchmarks": ["taubench_airline"],
    # },
    {
        "name": "taubench_toolcalling_gemini_2_5_flash",
        "agent_dir": "agents/taubench_tool_calling",
        "agent_function": "tool_calling.run",
        "model_name": "gemini/gemini-2.5-flash",
        "provider": "google",
        "benchmarks": ["taubench_airline"],
    },
    # {
    #     "name": "taubench_toolcalling_gemini_2_5_pro",
    #     "agent_dir": "agents/taubench_tool_calling",
    #     "agent_function": "tool_calling.run",
    #     "model_name": "gemini/gemini-2.5-pro",
    #     "provider": "google",
    #     "benchmarks": ["taubench_airline"],
    # },
    # {
    #     "name": "taubench_toolcalling_gemini_3_pro",
    #     "agent_dir": "agents/taubench_tool_calling",
    #     "agent_function": "tool_calling.run",
    #     "model_name": "gemini/gemini-3-pro-preview",
    #     "provider": "google",
    #     "benchmarks": ["taubench_airline"],
    # },
    # =========================================================================
    # Agentic Scaffold Agents (Claude Code, OpenAI Codex)
    # These use full agentic CLI tools rather than raw model APIs.
    # For taubench: tools are exposed via a local HTTP bridge.
    # For GAIA: the CLI agent uses its built-in tools (web search, bash, etc.)
    # Requirements: respective CLI tools must be installed.
    # =========================================================================
    # -------------------------------------------------------------------------
    # Claude Code (Anthropic CLI agent)
    # Install: npm install -g @anthropic-ai/claude-code
    # -------------------------------------------------------------------------
    # {
    #     "name": "taubench_claude_code_sonnet_4_5",
    #     "agent_dir": "agents/claude_code_agent",
    #     "agent_function": "main.run",
    #     "model_name": "claude-sonnet-4-5",
    #     "provider": "anthropic",
    #     "benchmarks": ["taubench_airline"],
    # },
    # {
    #     "name": "taubench_claude_code_opus_4_5",
    #     "agent_dir": "agents/claude_code_agent",
    #     "agent_function": "main.run",
    #     "model_name": "claude-opus-4-5",
    #     "provider": "anthropic",
    #     "benchmarks": ["taubench_airline"],
    # },
    # {
    #     "name": "gaia_claude_code_sonnet_4_5",
    #     "agent_dir": "agents/claude_code_agent",
    #     "agent_function": "main.run",
    #     "model_name": "claude-sonnet-4-5",
    #     "provider": "anthropic",
    #     "benchmarks": ["gaia"],
    # },
    # -------------------------------------------------------------------------
    # OpenAI Codex (OpenAI CLI agent)
    # Install: npm install -g @openai/codex
    # -------------------------------------------------------------------------
    # {
    #     "name": "taubench_codex_o4_mini",
    #     "agent_dir": "agents/openai_codex_agent",
    #     "agent_function": "main.run",
    #     "model_name": "o4-mini",
    #     "provider": "openai",
    #     "benchmarks": ["taubench_airline"],
    # },
    # {
    #     "name": "taubench_codex_gpt_4_1",
    #     "agent_dir": "agents/openai_codex_agent",
    #     "agent_function": "main.run",
    #     "model_name": "gpt-4.1",
    #     "provider": "openai",
    #     "benchmarks": ["taubench_airline"],
    # },
    # {
    #     "name": "gaia_codex_o4_mini",
    #     "agent_dir": "agents/openai_codex_agent",
    #     "agent_function": "main.run",
    #     "model_name": "o4-mini",
    #     "provider": "openai",
    #     "benchmarks": ["gaia"],
    # },
    # {
    #     "name": "taubench_codex_gpt_5_2",
    #     "agent_dir": "agents/openai_codex_agent",
    #     "agent_function": "main.run",
    #     "model_name": "gpt-5.2-2025-12-11",
    #     "provider": "openai",
    #     "task_timeout": 1800,  # 30 min — xhigh reasoning + multi-turn CLI needs more time
    #     "benchmarks": ["taubench_airline"],
    # },
    # {
    #     "name": "taubench_codex_gpt_5_2_medium",
    #     "agent_dir": "agents/openai_codex_agent",
    #     "agent_function": "main.run",
    #     "model_name": "gpt-5.2-2025-12-11",
    #     "provider": "openai",
    #     "reasoning_effort": "medium",
    #     "task_timeout": 1800,  # 30 min — xhigh reasoning + multi-turn CLI needs more time
    #     "benchmarks": ["taubench_airline"],
    # },
    # {
    #     "name": "taubench_codex_gpt_5_2_codex_medium",
    #     "agent_dir": "agents/openai_codex_agent",
    #     "agent_function": "main.run",
    #     "model_name": "gpt-5.2-codex",
    #     "provider": "openai",
    #     "reasoning_effort": "medium",
    #     "task_timeout": 1800,
    #     "benchmarks": ["taubench_airline"],
    # },
    # {
    #     "name": "taubench_codex_gpt_5_4",
    #     "agent_dir": "agents/openai_codex_agent",
    #     "agent_function": "main.run",
    #     "model_name": "gpt-5.4",
    #     "provider": "openai",
    #     "task_timeout": 1800,
    #     "benchmarks": ["taubench_airline"],
    # },
    # {
    #     "name": "taubench_codex_gpt_5_4_medium",
    #     "agent_dir": "agents/openai_codex_agent",
    #     "agent_function": "main.run",
    #     "model_name": "gpt-5.4",
    #     "provider": "openai",
    #     "reasoning_effort": "medium",
    #     "task_timeout": 1800,
    #     "benchmarks": ["taubench_airline"],
    # },
    # =========================================================================
    # GAIA Benchmark Agents (using hal_generalist_agent)
    # =========================================================================
    # -------------------------------------------------------------------------
    # OpenAI Models (GAIA)
    # -------------------------------------------------------------------------
    # {
    #     "name": "gaia_generalist_gpt_4o_mini",
    #     "agent_dir": "agents/hal_generalist_agent",
    #     "agent_function": "main.run",
    #     "model_name": "gpt-4o-mini-2024-07-18",
    #     "benchmarks": ["gaia"],
    #     "extra_agent_args": {"temperature": 0.0},
    # },
    # {
    #     "name": "gaia_generalist_gpt_4_turbo",
    #     "agent_dir": "agents/hal_generalist_agent",
    #     "agent_function": "main.run",
    #     "model_name": "gpt-4-turbo-2024-04-09",
    #     "benchmarks": ["gaia"],
    #     "extra_agent_args": {
    #         "temperature": 0.0
    #     }
    # },
    # {
    #     "name": "gaia_generalist_gpt_o1",
    #     "agent_dir": "agents/hal_generalist_agent",
    #     "agent_function": "main.run",
    #     "model_name": "o1-2024-12-17",
    #     "benchmarks": ["gaia"],
    #     "extra_agent_args": {
    #         "temperature": 0.0
    #     }
    # },
    # {
    #     "name": "gaia_generalist_gpt_5_2",
    #     "agent_dir": "agents/hal_generalist_agent",
    #     "agent_function": "main.run",
    #     "model_name": "gpt-5.2-2025-12-11",
    #     "benchmarks": ["gaia"],
    #     "extra_agent_args": {
    #         "temperature": 0.0
    #     }
    # },
    # {
    #     "name": "gaia_generalist_gpt_5_2_medium",
    #     "agent_dir": "agents/hal_generalist_agent",
    #     "agent_function": "main.run",
    #     "model_name": "gpt-5.2-2025-12-11",
    #     "reasoning_effort": "medium",
    #     "benchmarks": ["gaia"],
    #     "extra_agent_args": {
    #         "temperature": 0.0
    #     }
    # },
    # {
    #     "name": "gaia_generalist_gpt_5_4",
    #     "agent_dir": "agents/hal_generalist_agent",
    #     "agent_function": "main.run",
    #     "model_name": "gpt-5.4",
    #     "benchmarks": ["gaia"],
    #     "extra_agent_args": {
    #         "temperature": 0.0
    #     }
    # },
    # {
    #     "name": "gaia_generalist_gpt_5_4_medium",
    #     "agent_dir": "agents/hal_generalist_agent",
    #     "agent_function": "main.run",
    #     "model_name": "gpt-5.4",
    #     "reasoning_effort": "medium",
    #     "benchmarks": ["gaia"],
    #     "extra_agent_args": {
    #         "temperature": 0.0
    #     }
    # },
    # -------------------------------------------------------------------------
    # Anthropic Models (GAIA) - via OpenRouter
    # -------------------------------------------------------------------------
    # {
    #     "name": "gaia_generalist_claude_haiku_3_5",
    #     "agent_dir": "agents/hal_generalist_agent",
    #     "agent_function": "main.run",
    #     "model_name": "openrouter/anthropic/claude-3-5-haiku",
    #     "benchmarks": ["gaia"],
    #     "extra_agent_args": {
    #         "provider": "openai",  # OpenRouter uses OpenAI-compatible API
    #         "temperature": 0.0
    #     }
    # },
    # {
    #     "name": "gaia_generalist_claude_sonnet_3_7",
    #     "agent_dir": "agents/hal_generalist_agent",
    #     "agent_function": "main.run",
    #     "model_name": "openrouter/anthropic/claude-3.7-sonnet",
    #     "benchmarks": ["gaia"],
    #     "extra_agent_args": {
    #         "provider": "openai",  # OpenRouter uses OpenAI-compatible API
    #         "temperature": 0.0
    #     }
    # },
    # {
    #     "name": "gaia_generalist_claude_sonnet_4_5",
    #     "agent_dir": "agents/hal_generalist_agent",
    #     "agent_function": "main.run",
    #     "model_name": "openrouter/anthropic/claude-sonnet-4.5",
    #     "benchmarks": ["gaia"],
    #     "extra_agent_args": {
    #         "provider": "openai",  # OpenRouter uses OpenAI-compatible API
    #         "temperature": 0.0
    #     }
    # },
    # {
    #     "name": "gaia_generalist_claude_opus_4_5",
    #     "agent_dir": "agents/hal_generalist_agent",
    #     "agent_function": "main.run",
    #     "model_name": "openrouter/anthropic/claude-opus-4.5",
    #     "benchmarks": ["gaia"],
    #     "extra_agent_args": {
    #         "provider": "openai",  # OpenRouter uses OpenAI-compatible API
    #         "temperature": 0.0
    #     }
    # },
    # -------------------------------------------------------------------------
    # Google Gemini Models (GAIA)
    # -------------------------------------------------------------------------
    # {
    #     "name": "gaia_generalist_gemini_2_flash",
    #     "agent_dir": "agents/hal_generalist_agent",
    #     "agent_function": "main.run",
    #     "model_name": "gemini/gemini-2.0-flash",
    #     "benchmarks": ["gaia"],
    #     "extra_agent_args": {
    #         "provider": "google",
    #         "temperature": 0.0
    #     }
    # },
    # {
    #     "name": "gaia_generalist_gemini_2_5_flash",
    #     "agent_dir": "agents/hal_generalist_agent",
    #     "agent_function": "main.run",
    #     "model_name": "gemini/gemini-2.5-flash",
    #     "benchmarks": ["gaia"],
    #     "extra_agent_args": {
    #         "provider": "google",
    #         "temperature": 0.0
    #     }
    # },
    # {
    #     "name": "gaia_generalist_gemini_2_5_pro",
    #     "agent_dir": "agents/hal_generalist_agent",
    #     "agent_function": "main.run",
    #     "model_name": "gemini/gemini-2.5-pro",
    #     "benchmarks": ["gaia"],
    #     "extra_agent_args": {
    #         "provider": "google",
    #         "temperature": 0.0
    #     }
    # },
    # {
    #     "name": "gaia_generalist_gemini_3_pro",
    #     "agent_dir": "agents/hal_generalist_agent",
    #     "agent_function": "main.run",
    #     "model_name": "gemini/gemini-3-pro-preview",
    #     "benchmarks": ["gaia"],
    #     "extra_agent_args": {
    #         "provider": "google",
    #         "temperature": 0.0
    #     }
    # },
]

# =============================================================================
# BENCHMARK CONFIGURATION
# =============================================================================

BENCHMARK_CONFIGS = {
    "taubench_airline": {
        "benchmark_name": "taubench_airline",
        "requires_docker": False,
        "requires_vm": False,
        "max_concurrent": 5,  # Can run in parallel
        "task_ids": TAUBENCH_AIRLINE_CLEAN_TASKS,
        "compliance_constraints": [
            "pii_handling_customer_service",  # Customer service context - legitimate PII exchange allowed
            "destructive_ops_customer_service",  # Cancellations/mods with user confirmation are OK
            "data_minimization_customer_service",  # Accessing customer data to help them is OK
            "policy_circumvention_customer_service",  # Detect agents manipulated to bypass policies
            "financial_accuracy_customer_service",  # Detect wrong refund/charge amounts
            "authentication_bypass_customer_service",  # Detect improper identity verification
            "commitment_overreach_customer_service",  # Detect promises beyond agent authority
        ],
    },
    # "taubench_retail": {
    #     "benchmark_name": "taubench_retail",
    #     "requires_docker": False,
    #     "requires_vm": False,
    #     "max_concurrent": 5,  # Can run in parallel
    #     "compliance_constraints": [
    #         "pii_handling_customer_service",  # Customer service context
    #         "destructive_ops_customer_service",  # Cancellations/mods with user confirmation are OK
    #         "data_minimization_customer_service",  # Accessing customer data to help them is OK
    #         "policy_circumvention_customer_service",  # Detect agents manipulated to bypass policies
    #         "financial_accuracy_customer_service",  # Detect wrong refund/charge amounts
    #         "authentication_bypass_customer_service",  # Detect improper identity verification
    #         "commitment_overreach_customer_service",  # Detect promises beyond agent authority
    #     ],
    # },
    # "gaia": {
    #     "benchmark_name": "gaia",
    #     "requires_docker": False,
    #     "requires_vm": False,
    #     "max_concurrent": 5,  # GAIA tasks are independent, can run in parallel
    #     "compliance_constraints": [
    #         "pii_handling_gaia",  # Q&A context - providing requested info is expected
    #         "no_destructive_ops",
    #         "safe_code_execution",  # Check for harmful code patterns
    #         "no_unauthorized_access",  # Check for unauthorized resource access
    #     ],
    # },
}
