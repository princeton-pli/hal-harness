"""
Utility for generating prompt variations for prompt sensitivity evaluation.
"""

import os
from typing import Dict, List, Any
from openai import OpenAI


class PromptVariationGenerator:
    """Generates semantic-preserving variations of prompts using LLM."""

    def __init__(self, model_name: str = "gpt-4o-mini-2024-07-18", num_variations: int = 3):
        """
        Initialize the prompt variation generator.

        Args:
            model_name: OpenAI model to use for generating variations
            num_variations: Number of variations to generate per prompt
        """
        self.model_name = model_name
        self.num_variations = num_variations
        self.client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    def generate_variations(self, prompt: str, task_id: str = None) -> List[str]:
        """
        Generate semantic-preserving variations of a prompt.

        Args:
            prompt: The original prompt text to vary
            task_id: Optional task identifier for logging

        Returns:
            List of prompt variations (includes original + variations)
        """
        system_prompt = """You are an expert at paraphrasing text while preserving exact semantic meaning.

Your task is to generate variations of the given prompt that:
1. Preserve the exact meaning and intent of the original
2. Use different wording, phrasing, and sentence structure
3. Maintain the same level of specificity and detail
4. Keep all critical information and constraints
5. Sound natural and fluent

Generate variations that differ in style (e.g., formal vs casual, verbose vs concise) while keeping the core meaning identical.

Output ONLY the variations, one per line, without numbering or additional text."""

        user_prompt = f"""Generate {self.num_variations} semantic-preserving variations of this prompt:

{prompt}

Remember: The variations must preserve the exact meaning, just use different wording."""

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.7,
                max_tokens=2000
            )

            # Parse variations from response
            variations_text = response.choices[0].message.content.strip()
            variations = [v.strip() for v in variations_text.split('\n') if v.strip()]

            # Filter out any that are just numbers or very short
            variations = [v for v in variations if len(v) > 10]

            # Ensure we have the right number of variations
            variations = variations[:self.num_variations]

            # Include original prompt as first variation
            all_variations = [prompt] + variations

            return all_variations

        except Exception as e:
            print(f"Error generating variations for task {task_id}: {e}")
            # Fallback: return just the original prompt
            return [prompt]

    def apply_variations_to_dataset(self, dataset: Dict[str, Any], prompt_field: str = "Question") -> Dict[str, List[Dict[str, Any]]]:
        """
        Apply prompt variations to an entire dataset.

        Args:
            dataset: Dictionary mapping task_ids to task data
            prompt_field: Name of the field containing the prompt to vary

        Returns:
            Dictionary mapping task_ids to lists of varied task data
            Each task_id maps to a list where each element is the task data with a different prompt variation
        """
        varied_dataset = {}

        for task_id, task_data in dataset.items():
            # Check if the prompt field exists
            if prompt_field not in task_data:
                print(f"Warning: Prompt field '{prompt_field}' not found in task {task_id}. Skipping variations.")
                varied_dataset[task_id] = [task_data]
                continue

            # Get original prompt
            original_prompt = task_data[prompt_field]

            # Generate variations
            variations = self.generate_variations(original_prompt, task_id)

            # Create task data for each variation
            varied_tasks = []
            for i, varied_prompt in enumerate(variations):
                varied_task = task_data.copy()
                varied_task[prompt_field] = varied_prompt
                varied_task['prompt_variation_id'] = i  # Track which variation this is
                varied_tasks.append(varied_task)

            varied_dataset[task_id] = varied_tasks

        return varied_dataset


def get_prompt_field_for_benchmark(benchmark_name: str) -> str:
    """
    Get the appropriate prompt field name for a given benchmark.

    Args:
        benchmark_name: Name of the benchmark

    Returns:
        Field name containing the prompt text, or None if the benchmark
        doesn't support prompt variations (e.g., TauBench where prompts
        come from environment objects, not input data)
    """
    # Map benchmark names to their prompt fields
    prompt_field_map = {
        'gaia': 'Question',
        'usaco': 'problem_statement',
        'swebench_verified': 'problem_statement',
        'swebench_verified_mini': 'problem_statement',
        'appworld_test_normal': 'instruction',
        'appworld_test_challenge': 'instruction',
        'assistantbench': 'task',
        'scicode': 'problem_statement',
        'scicode_easy': 'problem_statement',
        'scicode_hard': 'problem_statement',
    }

    # TauBench now supported via instruction field
    if benchmark_name in ['taubench_retail', 'taubench_airline']:
        return 'instruction'

    # Handle inspect benchmarks
    if benchmark_name.startswith('inspect_evals/'):
        # Most inspect benchmarks use 'input' or 'question'
        return 'input'

    return prompt_field_map.get(benchmark_name, 'Question')  # Default to 'Question'
