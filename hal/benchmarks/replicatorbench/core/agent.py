import os
import re
import sys
import time
import json
import logging
import tiktoken
from openai import OpenAI
from core.utils import get_logger
from core.constants import API_KEY, GENERATE_REACT_CONSTANTS

logger, formatter = get_logger()
client = OpenAI(api_key=API_KEY) 

ACTION_PATTERNS = [
    # 1) Code-fenced arguments (preferred; captures multiline safely)
    re.compile(r'(?mis)^\s*Action:\s*([a-z0-9_]+)\s*:\s*```(?:json)?\s*(.*?)\s*```'),

    # 2) Multiline JSON object right after Action: (non-fenced)
    re.compile(r'(?mis)^\s*Action:\s*([a-z0-9_]+)\s*:\s*(\{.*?\})\s*(?:\n|$)'),

    # 3) Bolded "Action:" some models produce
    re.compile(r'(?mis)^\s*\*\*Action:\*\*\s*([a-z0-9_]+)\s*:\s*(.+?)\s*(?:\n|$)'),

    # 4) Single-line Action (last!)
    re.compile(r'(?mi)^\s*Action:\s*([a-z0-9_]+)\s*:\s*(.+)$'),
]

DICT_ONLY_ACTIONS = {
    "write_file",
    "edit_file",
    "read_file",   # this is kwargs-based in our system
    "read_json",
    # add others that require kwargs
}

REASONING_MODELS = ("o1", "o3", "gpt-5", "gpt-5-mini")

def is_reasoning_model(model: str) -> bool:
    return model.startswith(REASONING_MODELS)

def messages_to_responses_input(messages):
    """
    Convert chat-style messages to Responses API input.
    - system/user/developer -> input_text
    - assistant -> output_text
    """
    output = []
    for m in messages:
        role = m.get("role", "user")
        content = m.get("content", "")

        if not isinstance(content, str):
            content = str(content)

        part_type = "output_text" if role == "assistant" else "input_text"

        output.append({
            "role": role,
            "content": [
                {
                    "type": part_type,
                    "text": content
                }
            ]
        })
    return output

def _extract_action(text: str):
    for pat in ACTION_PATTERNS:
        m = pat.search(text)
        if m:
            return m.group(1), m.group(2).strip()
    return None

def update_metadata(study_path: str, stage: str, data: dict):
    """
    Updates metadata.json in the study_path with metrics for a specific stage.
    """
    meta_path = os.path.join(study_path, "metadata.json")
    
    # Load existing or create new
    if os.path.exists(meta_path):
        try:
            with open(meta_path, "r") as f:
                metadata = json.load(f)
        except json.JSONDecodeError:
            metadata = {}
    else:
        metadata = {}

    # Update the specific stage
    metadata[stage] = data

    # Write back
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"Updated metadata for {stage} in {meta_path}")

class Agent:
    def __init__(self, system="", session_state=None, model="gpt-4o", tools=None):
        self.system = system
        self.messages = []
        if self.system:
            self.messages.append({"role": "system", "content": system})
        self.session_state = session_state or {}
        self.model = model
        self.tools = tools
        self._tpm_window_start = time.time()
        self._tpm_tokens = 0  # tokens used since last reset

    def __call__(self, message, tool_outputs=None):
        content, usage = self.execute(message, tool_outputs)
        return content, usage
    
    def execute(self, message=None, tool_outputs=None):
        # 1. Add user message
        if message:
            self.messages.append({"role": "user", "content": message})
        
        # 2. Add tool outputs (results from previous turn)
        if tool_outputs:
            self.messages.extend(tool_outputs)

        # Rate Limiting (TPM)
        now = time.time()
        if now - self._tpm_window_start >= 60:
            self._tpm_window_start = now
            self._tpm_tokens = 0

        # Simple estimation (you can keep your tiktoken logic if preferred)
        estimated_tokens = 1000 
        if self._tpm_tokens + estimated_tokens > 30000:
            logger.info("Rate limit approach, going to sleep...zZZ\n")
            time.sleep(25)
            self._tpm_window_start = time.time()
            self._tpm_tokens = 0

        # Call API
        params = {
            "model": self.model,
            "messages": self.messages,
        }

        # Handle Tools
        if self.tools:
            params["tools"] = self.tools
            params["tool_choice"] = "auto"

        # Handle Model Specifics (o1/o3 vs GPT-4o)
        is_reasoning = self.model in REASONING_MODELS
        
        if is_reasoning:
            # o1/o3 support 'max_completion_tokens' instead of 'max_tokens'
            # and may support 'reasoning_effort'
            params["reasoning_effort"] = "medium" # or "high" / "low"
            # params["max_completion_tokens"] = 4000
        else:
            params["temperature"] = 0
            # params["max_tokens"] = 4000

        try:
            # Unified call for both GPT-4o and o3
            completion = client.chat.completions.create(**params)
            
            response_message = completion.choices[0].message
            usage = completion.usage

            # Update usage stats
            usage_stats = {
                "prompt_tokens": usage.prompt_tokens,
                "completion_tokens": usage.completion_tokens,
                "total_tokens": usage.total_tokens
            }
            self._tpm_tokens += usage_stats["total_tokens"]

            # Append the full message object
            self.messages.append(response_message)

            return response_message, usage_stats

        except Exception as e:
            logger.error(f"Error in OpenAI call: {e}")
            raise e

    def _execute_tool_call(self, known_actions, action, action_input_str):
        tool_func = known_actions[action]

        try:
            parsed_args = json.loads(action_input_str.strip())
            print(f"DEBUG: Parsed args for '{action}': {parsed_args}, Type: {type(parsed_args)}")

            if isinstance(parsed_args, dict):
                if "dataset" in action:
                    return tool_func(self.session_state, **parsed_args)
                return tool_func(**parsed_args)

            else:
                if "dataset" in action:
                    return tool_func(self.session_state, parsed_args)
                return tool_func(parsed_args)

        except json.JSONDecodeError as e:
            if action in DICT_ONLY_ACTIONS:
                return (
                f"Error: Tool '{action}' requires JSON object args.\n"
                f"JSON parse error: {e}\n"
                f"Got: {action_input_str.strip()[:500]}"
                )

            # Raw-string fallback only for single-string tools
            raw = action_input_str.strip()
            if (raw.startswith('"') and raw.endswith('"')) or (raw.startswith("'") and raw.endswith("'")):
                raw = raw[1:-1]

            try:
                if "dataset" in action:
                    return tool_func(self.session_state, raw)
                return tool_func(raw)
            except Exception as e2:
                return f"Error while executing tool '{action}' with raw string arg: {e2}"
        
        except Exception as e3:
            return f"Error while executing tool '{action}' with raw string arg: {e3}"

    def _get_encoding(self):
        if tiktoken is None:
            return None
        try:
            return tiktoken.encoding_for_model(self.model)
        except Exception:
            try:
                return tiktoken.get_encoding("cl100k_base")
            except Exception:
                return None

    def count_current_tokens(self) -> int:
        """
        Rough count of tokens of self.messages (input side only).
        Uses tiktoken if available; otherwise ~4 chars/token heuristic.
        """
        enc = self._get_encoding()

        def cnt(s: str) -> int:
            if enc:
                return len(enc.encode(s))
            return max(1, len(s) // 4)

        total = 0
        for m in self.messages:
            role = m.get("role", "")
            content = m.get("content", "")
            if isinstance(content, list):
                # Only count text parts for multimodal messages
                content = "\n".join(
                    part.get("text", "")
                    for part in content
                    if isinstance(part, dict) and part.get("type") == "text"
                )
            elif not isinstance(content, str):
                content = str(content)
            total += cnt(role + ": " + content) + 6  # small overhead
        total += 2
        return total

def run_react_loop(system_prompt: str, known_actions: dict, tool_definitions: list, question: str, *,
                   max_turns: int = 50, session_state=None, on_final=None, log_turns: bool=True,
                   study_path: str = None, stage_name: str = None, checkpoint_map: dict = None, model_name: str=None):
    
    thought_instruction = "\nIMPORTANT: Before calling any tool, you must output a short 'Thought' explaining your reasoning."
    bot = Agent(system_prompt + thought_instruction, model=model_name, session_state=session_state or {}, tools=tool_definitions)    
    next_prompt = question
    tool_outputs = []
    
    start_time = time.time()
    total_tokens_used = 0
    total_prompt_tokens_used = 0
    total_completion_tokens_used = 0
    turn_metrics = []
    
    current_checkpoint = "0. Initialization"
    checkpoint_stats = {} 

    for i in range(max_turns):
        turn_start = time.time()
        
        if log_turns:
            logger.info(f"\n--- Turn {i+1} ---")
            # Only log input if it's a user message or observation (skip empty inputs)
            if next_prompt: 
                 # Truncate very long observations in logs to avoid clutter, if desired
                 display_prompt = next_prompt[:5000] + "..." if len(next_prompt) > 5000 else next_prompt
                 logger.info(f"***Agent input:\n{display_prompt}")

        # Call the Agent
        response_msg, usage = bot(message=next_prompt, tool_outputs=tool_outputs)
        
        # Reset inputs
        next_prompt = None 
        tool_outputs = []

        # Track usage
        turn_prompt = usage.get("prompt_tokens", 0)
        turn_completion = usage.get("completion_tokens", 0)
        turn_total = usage.get("total_tokens", 0)
        total_prompt_tokens_used += turn_prompt
        total_completion_tokens_used += turn_completion
        total_tokens_used += turn_total

        # log the "Thought"
        # if tools are called, message.content might contain the reasoning.
        content_text = response_msg.content or ""
        if log_turns and content_text.strip():
            logger.info(f"***Agent output (Thought):\n{content_text}")

        action = None
        is_final = False
        
        # Branch 1: Tools Present
        if response_msg.tool_calls:
            # Checkpoint logic based on first tool
            primary_tool = response_msg.tool_calls[0]
            action = primary_tool.function.name
            
            if checkpoint_map and action in checkpoint_map:
                current_checkpoint = checkpoint_map[action]
            elif not checkpoint_map:
                current_checkpoint = "Running Action"
            
            # Log the Action clearly
            logger.info(f" -- Running Action(s): {len(response_msg.tool_calls)} tools requested. Primary: {action} [Checkpoint: {current_checkpoint}]")

            for tool_call in response_msg.tool_calls:
                func_name = tool_call.function.name
                args_str = tool_call.function.arguments
                call_id = tool_call.id
                
                tool_result_content = ""
                
                # Execute Tool
                if func_name in known_actions:
                    try:
                        func_args = json.loads(args_str)
                        func = known_actions[func_name]
                        
                        # Handle session_state injection
                        if func_name.startswith("get_dataset") or func_name == "load_dataset":
                             observation = func(bot.session_state, **func_args)
                        # inject study_path to restrict access to outside folders
                        elif func_name == "list_files_in_folder":
                            func_args["study_path"] = study_path 
                            observation = func(**func_args)
                        else:
                             observation = func(**func_args)
                        
                        tool_result_content = str(observation)
                             
                    except Exception as e:
                        error_msg = f"Error executing {func_name}: {str(e)}"
                        logger.error(error_msg)
                        tool_result_content = error_msg
                        if "Unknown action" in str(e):
                             update_metadata(study_path, stage_name, {
                                "error": f"Unknown action: {func_name}",
                                "partial_turns": turn_metrics
                            })
                else:
                    tool_result_content = f"Error: Tool {func_name} not found."
                    update_metadata(study_path, stage_name, {"error": f"Unknown action: {func_name}", "partial_turns": turn_metrics})

                # Log the Observation which is the result of the tool
                if log_turns:
                    # truncate very long file reads in logs so they don't fill the terminal
                    log_content = tool_result_content[:2000] + "\n... (truncated)" if len(tool_result_content) > 2000 else tool_result_content
                    logger.info(f"***Observation ({func_name}):\n{log_content}")

                tool_outputs.append({
                    "tool_call_id": call_id,
                    "role": "tool",
                    "name": func_name,
                    "content": tool_result_content
                })

        # Branch 2: No Tools (Text Response / Final Answer)
        else:
            # Check for JSON Answer (Answer: {...} or ```json ... ``` or raw json)
            json_answer_str = None
            
            # 1. Answer: { ... }
            answer_match = re.search(r'Answer:\s*(\{.*?\})\s*$', content_text, re.DOTALL)
            if answer_match: json_answer_str = answer_match.group(1).strip()
            
            # 2. Markdown JSON
            elif "```json" in content_text:
                 json_match = re.search(r'```json\n(.*?)\n```', content_text, re.DOTALL)
                 if json_match: json_answer_str = json_match.group(1).strip()
            
            # 3. Raw JSON
            elif content_text.strip().startswith("{") and content_text.strip().endswith("}"):
                 json_answer_str = content_text.strip()

            if json_answer_str:
                is_final = True
                current_checkpoint = "8. Final Output & Parsing"
            else:
                if i == 0:
                     next_prompt = "Reminder: Please use the available tools or provide the final Answer JSON."
                     continue
                
        if is_final and json_answer_str:
            turn_duration = time.time() - turn_start
            stats = checkpoint_stats.get(current_checkpoint, {"time": 0.0,"tokens": 0,"prompt_tokens": 0,"completion_tokens": 0,"turns": 0})
            stats["time"] += turn_duration
            stats["tokens"] += turn_total
            stats["prompt_tokens"] += turn_prompt
            stats["completion_tokens"] += turn_completion
            stats["turns"] += 1
            checkpoint_stats[current_checkpoint] = stats
            
            try:
                json_start = json_answer_str.find('{')
                json_end = json_answer_str.rfind('}')
                if json_start != -1 and json_end != -1:
                    json_answer_str = json_answer_str[json_start : json_end + 1]

                final_answer = json.loads(json_answer_str)
                logger.info("\n--- Final Answer Found ---")
                
                if on_final: on_final(final_answer)

                if study_path and stage_name:
                    total_time = time.time() - start_time
                    metric_data = {
                        "status": "Success",
                        "total_time_seconds": round(total_time, 2),
                        "total_tokens": total_tokens_used,
                        "prompt_tokens": total_prompt_tokens_used,
                        "completion_tokens": total_completion_tokens_used,
                        "total_turns": i + 1,
                        "checkpoint_stats": checkpoint_stats,
                        "turn_history": turn_metrics
                    }
                    update_metadata(study_path, stage_name, metric_data)

                return final_answer
            except Exception as e:
                logger.error(f"Error parsing final answer: {e}")
                return {"error": str(e)}

        turn_duration = time.time() - turn_start
        if current_checkpoint not in checkpoint_stats:
            checkpoint_stats[current_checkpoint] = {"time": 0.0,"tokens": 0,"prompt_tokens": 0,"completion_tokens": 0,"turns": 0,}        
        checkpoint_stats[current_checkpoint]["time"] += turn_duration
        checkpoint_stats[current_checkpoint]["tokens"] += turn_total
        checkpoint_stats[current_checkpoint]["prompt_tokens"] += turn_prompt
        checkpoint_stats[current_checkpoint]["completion_tokens"] += turn_completion
        checkpoint_stats[current_checkpoint]["turns"] += 1

        turn_metrics.append({
            "turn": i + 1,
            "action": action if action else "None",
            "checkpoint": current_checkpoint,
            "duration_seconds": round(turn_duration, 2),
            "prompt_tokens": turn_prompt,
            "completion_tokens": turn_completion,
            "total_tokens": turn_total,
        })

    logger.warning("Max turns reached.")
    if study_path and stage_name:
        update_metadata(study_path, stage_name, {
            "status": "Failed - Max Turns Reached",
            "total_time_seconds": round(time.time() - start_time, 2),
            "total_tokens": total_tokens_used,
            "prompt_tokens": total_prompt_tokens_used,
            "completion_tokens": total_completion_tokens_used,
            "total_turns": max_turns,
            "checkpoint_stats": checkpoint_stats,
            "turn_history": turn_metrics
        })
        
    return {"error": "Max turns reached without a final answer."}

def save_output(extracted_json, study_path, filename: str = "replication_info.json", stage_name: str = "design"):
    os.makedirs(study_path, exist_ok=True)
    out_path = os.path.join(study_path, filename)
    with open(out_path, "w") as f:
        json.dump(extracted_json, f, indent=2)
    logger.info(f"{stage_name.capitalize()} stage output saved to {out_path}")
    return out_path