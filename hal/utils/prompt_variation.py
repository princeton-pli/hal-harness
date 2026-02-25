"""
Utility for generating prompt variations for prompt sensitivity evaluation.

This module provides tools for generating semantic-preserving variations of prompts
at different strength levels to test agent robustness to input phrasing.

Variation Strength Levels:
- mild: Surface-level changes (synonyms, formality)
- medium: Structural changes (reordering, sentence restructuring)
- strong: Significant rewrites (conversational style, implicit information)
- naturalistic: Realistic user behavior (typos, abbreviations, informal speech)
"""

import os
from typing import Dict, List, Any
from enum import Enum
from openai import OpenAI


class VariationStrength(Enum):
    """Strength levels for prompt variations."""

    MILD = "mild"
    MEDIUM = "medium"
    STRONG = "strong"
    NATURALISTIC = "naturalistic"


# Style directives for the simulated user's SYSTEM PROMPT (not instruction content)
# These are injected into the user simulator's system prompt to control HOW it communicates
USER_STYLE_DIRECTIVES = {
    VariationStrength.MILD: "",  # No directive - use default formal style
    VariationStrength.MEDIUM: (
        "\n\nCommunication Style Guidelines:\n"
        "- Speak in a moderately casual tone\n"
        "- Use complete sentences but don't be overly formal\n"
        "- You may use contractions (don't, can't, I'll) and simple language\n"
        "- Be polite but not stiff"
    ),
    VariationStrength.STRONG: (
        "\n\nCommunication Style Guidelines:\n"
        "- Speak very casually and conversationally, like chatting with a helpful friend\n"
        "- Use contractions freely (don't, won't, I'm, it's)\n"
        "- Include natural filler words occasionally (um, so, like, basically, you know)\n"
        "- Use informal expressions (sounds good, works for me, got it)\n"
        "- Be friendly and relaxed in tone\n"
        "- Still convey all necessary information clearly"
    ),
    VariationStrength.NATURALISTIC: (
        "\n\nCommunication Style Guidelines:\n"
        "- Type like a real person texting or chatting online\n"
        "- Use common abbreviations: pls, thx, info, w/, b/c, rn, idk, tbh\n"
        "- Use lowercase freely, inconsistent capitalization is fine\n"
        "- Include occasional typos or quick corrections (e.g., 'teh' -> 'the', 'actualy' -> 'actually')\n"
        "- Use incomplete sentences and fragments\n"
        "- Skip unnecessary words (e.g., 'need flight seattle' instead of 'I need a flight to Seattle')\n"
        "- Use casual punctuation: multiple periods..., exclamation marks!\n"
        "- Include parenthetical asides (not a morning person lol)\n"
        "- Example: 'yeah so i need to change my flight... the one to seattle? can we move it to next week pls'"
    ),
}


# System prompts for each variation strength level
VARIATION_SYSTEM_PROMPTS = {
    VariationStrength.MILD: """You are an expert at paraphrasing text while preserving exact semantic meaning.

Your task is to generate variations of the given prompt that:
1. Preserve the exact meaning and intent of the original
2. Use different wording and phrasing
3. Maintain the same level of specificity and detail
4. Keep all critical information and constraints
5. Sound natural and fluent

Techniques to use:
- Synonym substitution (e.g., "book" → "reserve", "flight" → "trip")
- Formality changes (formal ↔ casual)
- Voice changes (active ↔ passive)
- Minor sentence restructuring

Output ONLY the variations, one per line, without numbering or additional text.""",
    VariationStrength.MEDIUM: """You are an expert at paraphrasing text while preserving exact semantic meaning.

Your task is to generate variations that significantly restructure the prompt while keeping ALL information intact.

Techniques to use:
1. REORDER information - put constraints and requirements in a completely different order
2. RESTRUCTURE sentences - combine multiple short sentences or split long ones
3. CHANGE perspective - switch between "I want X" and "X is what I need" and "Please do X"
4. VARY specificity slightly - "3 bags" could become "three pieces of luggage"
5. MIX communication styles - imperative ("Book X") vs declarative ("I need X booked")

CRITICAL: All original information MUST be preserved. Do not add or remove any constraints.

Output ONLY the variations, one per line, without numbering or additional text.""",
    VariationStrength.STRONG: """You are an expert at generating realistic rewrites of instructions while preserving exact semantic meaning.

Your task is to generate variations that sound like different people wrote them, with very different styles and structures.

Techniques to use:
1. CONVERSATIONAL REWRITES - Write as if someone is casually explaining what they need
   - Add natural filler phrases: "So basically...", "Oh, and...", "I think..."
   - Use contractions and informal language

2. IMPLICIT INFORMATION - Make some explicit statements more implicit (but still inferable)
   - "3 bags" → "I'll have a few bags with me - three total"
   - "no insurance" → "skip any add-ons like insurance"

3. COMPLETE RESTRUCTURING - Organize information in a completely different logical flow
   - Lead with preferences instead of requirements
   - Group related items differently

4. DIFFERENT PERSONAS - Write as if from different user types
   - Busy professional (concise, direct)
   - Chatty user (more verbose, conversational)
   - Detail-oriented user (very explicit about everything)

CRITICAL: ALL original information and constraints MUST be preserved and inferable. The agent should be able to extract the exact same requirements.

Output ONLY the variations, one per line, without numbering or additional text.""",
    VariationStrength.NATURALISTIC: """You are an expert at generating realistic user input variations that mimic how real people actually type and communicate.

Your task is to generate variations that reflect authentic user behavior while preserving ALL semantic meaning.

Techniques to use:
1. INFORMAL TYPING PATTERNS
   - Lowercase text or inconsistent capitalization
   - Common abbreviations: "pls", "thx", "w/", "b4", "info", "appt"
   - Arrows and symbols: "NYC → Seattle", "flight @ 11am"
   - Casual punctuation: multiple periods..., exclamation marks!

2. REALISTIC USER BEHAVIORS
   - Self-corrections: "I mean...", "actually...", "wait, let me clarify..."
   - Hedging language: "I think", "probably", "if possible"
   - Run-on thoughts: connecting multiple ideas without clear breaks
   - Parenthetical asides: (not a morning person), (if that matters)

3. VARIED FORMALITY
   - Mix of formal and casual in same message
   - Colloquialisms: "works for me", "sounds good", "no biggie"

4. NATURAL IMPERFECTIONS
   - Occasional minor typos that don't change meaning
   - Missing articles: "need flight to Seattle" instead of "need a flight"
   - Sentence fragments: "Economy class. 3 bags. No insurance."

CRITICAL:
- ALL original information MUST be preserved and clearly extractable
- Variations should be challenging but NOT ambiguous about the actual requirements
- A competent agent should still be able to understand and fulfill the request

Output ONLY the variations, one per line, without numbering or additional text.""",
}


# User prompt templates for each strength level
VARIATION_USER_PROMPTS = {
    VariationStrength.MILD: """Generate {num_variations} semantic-preserving variations of this prompt:

{prompt}

Remember: The variations must preserve the exact meaning, just use different wording.""",
    VariationStrength.MEDIUM: """Generate {num_variations} RESTRUCTURED variations of this prompt.
Each variation should organize the information in a DIFFERENT ORDER and use DIFFERENT sentence structures.

Original prompt:
{prompt}

Remember: ALL information must be preserved, but the structure should be notably different.""",
    VariationStrength.STRONG: """Generate {num_variations} SIGNIFICANTLY DIFFERENT variations of this prompt.
Each variation should sound like it was written by a different person with a different communication style.

Original prompt:
{prompt}

Make each variation feel genuinely different - one could be very conversational, another very concise, another more formal but restructured. ALL original information and constraints must be preserved.""",
    VariationStrength.NATURALISTIC: """Generate {num_variations} REALISTIC variations of this prompt that mimic how actual users type.
Include natural imperfections, abbreviations, and informal patterns while keeping ALL information intact.

Original prompt:
{prompt}

Examples of naturalistic style:
- "nyc to seattle, may 20, 1-way. economy pls. nothing before 11am"
- "So I need to get from New York to Seattle... May 20th works. Just one way. Economy's fine, and I'd prefer not to leave super early - like after 11?"
- "booking: NYC→SEA, 5/20, economy, one-way. prefer late morning departure (11am+)"

Generate diverse naturalistic variations. ALL requirements must be preserved and clearly extractable.""",
}


class PromptVariationGenerator:
    """Generates semantic-preserving variations of prompts using LLM."""

    def __init__(
        self,
        model_name: str = "gpt-4o-mini-2024-07-18",
        num_variations: int = 3,
        strength: str = "mild",
    ):
        """
        Initialize the prompt variation generator.

        Args:
            model_name: OpenAI model to use for generating variations
            num_variations: Number of variations to generate per prompt
            strength: Variation strength level - one of:
                - "mild": Surface-level changes (synonyms, formality) [default]
                - "medium": Structural changes (reordering, restructuring)
                - "strong": Significant rewrites (conversational, implicit info)
                - "naturalistic": Realistic user behavior (typos, abbreviations)
        """
        self.model_name = model_name
        self.num_variations = num_variations

        # Check for API key
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY environment variable is required for prompt variation generation. "
                "Set it with: export OPENAI_API_KEY=your-key"
            )
        self.client = OpenAI(api_key=api_key)

        # Parse strength level
        try:
            self.strength = VariationStrength(strength.lower())
        except ValueError:
            print(f"Warning: Unknown variation strength '{strength}'. Using 'mild'.")
            self.strength = VariationStrength.MILD

        # Get prompts for this strength level
        self.system_prompt = VARIATION_SYSTEM_PROMPTS[self.strength]
        self.user_prompt_template = VARIATION_USER_PROMPTS[self.strength]

    def generate_variations(self, prompt: str, task_id: str = None) -> List[str]:
        """
        Generate semantic-preserving variations of a prompt.

        Args:
            prompt: The original prompt text to vary
            task_id: Optional task identifier for logging

        Returns:
            List of prompt variations (includes original + variations)
            Each variation includes a style directive prefix (for non-mild strengths)
            that instructs the simulated user HOW to communicate.
        """
        import re

        user_prompt = self.user_prompt_template.format(
            num_variations=self.num_variations, prompt=prompt
        )

        try:
            # Use higher temperature for stronger variations
            temperature = {
                VariationStrength.MILD: 0.7,
                VariationStrength.MEDIUM: 0.8,
                VariationStrength.STRONG: 0.9,
                VariationStrength.NATURALISTIC: 0.9,
            }.get(self.strength, 0.7)

            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=temperature,
                max_tokens=3000,  # Increased for longer variations
            )

            # Parse variations from response
            variations_text = response.choices[0].message.content.strip()
            variations = [v.strip() for v in variations_text.split("\n") if v.strip()]

            # Filter out any that are just numbers or very short
            variations = [v for v in variations if len(v) > 10]

            # Remove any leading numbers/bullets that the model might have added
            cleaned_variations = []
            for v in variations:
                # Remove common prefixes like "1.", "1)", "- ", "• ", etc.
                cleaned = re.sub(r"^[\d]+[\.\)]\s*", "", v)
                cleaned = re.sub(r"^[-•]\s*", "", cleaned)
                cleaned_variations.append(cleaned.strip())

            variations = [v for v in cleaned_variations if len(v) > 10]

            # Ensure we have the right number of variations
            variations = variations[: self.num_variations]

            # Include original prompt as first variation
            # Note: Style directive is NOT prepended here - it should be injected
            # into the user simulator's system prompt by the agent code
            all_variations = [prompt] + variations

            return all_variations

        except Exception as e:
            print(f"Error generating variations for task {task_id}: {e}")
            # Fallback: return just the original prompt
            return [prompt]

    def generate_single_variation_for_dataset(
        self, dataset: Dict[str, Any], prompt_field: str, variation_index: int
    ) -> Dict[str, Dict[str, Any]]:
        """
        Generate a single specific variation for all tasks in a dataset.

        This is more efficient than generating all variations when only one is needed.

        Args:
            dataset: Dictionary mapping task_ids to task data
            prompt_field: Name of the field containing the prompt to vary
            variation_index: Which variation to generate (0=original, 1..N=variations)

        Returns:
            Dictionary mapping task_ids to single varied task data (not a list)
        """
        varied_dataset = {}
        total_tasks = len(dataset)

        if variation_index == 0:
            # Index 0 = original prompt, no generation needed
            print(f"Using original prompts (variation 0) for {total_tasks} tasks...")
            for task_id, task_data in dataset.items():
                varied_task = task_data.copy()
                varied_task["prompt_variation_id"] = 0
                varied_task["prompt_variation_strength"] = self.strength.value
                varied_dataset[task_id] = varied_task
            return varied_dataset

        print(
            f"Generating {self.strength.value} variation {variation_index} for {total_tasks} tasks..."
        )

        for idx, (task_id, task_data) in enumerate(dataset.items()):
            print(
                f"  [{idx + 1}/{total_tasks}] Generating variation {variation_index} for task {task_id}...",
                end=" ",
                flush=True,
            )

            # Check if the prompt field exists
            if prompt_field not in task_data:
                print(f"skipped (no '{prompt_field}' field)")
                varied_task = task_data.copy()
                varied_task["prompt_variation_id"] = variation_index
                varied_task["prompt_variation_strength"] = self.strength.value
                varied_dataset[task_id] = varied_task
                continue

            # Get original prompt
            original_prompt = task_data[prompt_field]

            # Generate variations (returns [original, var1, var2, ...])
            variations = self.generate_variations(original_prompt, task_id)

            # Pick the specific variation we need
            if variation_index < len(variations):
                varied_prompt = variations[variation_index]
            else:
                # Fallback to original if requested index doesn't exist
                print(
                    f"warning: variation {variation_index} not available, using original...",
                    end=" ",
                )
                varied_prompt = original_prompt

            # Create task data for this variation
            varied_task = task_data.copy()
            varied_task[prompt_field] = varied_prompt
            varied_task["prompt_variation_id"] = variation_index
            varied_task["prompt_variation_strength"] = self.strength.value
            varied_dataset[task_id] = varied_task
            print("done")

        print(
            f"Completed generating variation {variation_index} for all {total_tasks} tasks"
        )
        return varied_dataset

    def apply_variations_to_dataset(
        self, dataset: Dict[str, Any], prompt_field: str = "Question"
    ) -> Dict[str, List[Dict[str, Any]]]:
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
        total_tasks = len(dataset)

        print(f"Generating {self.strength.value} variations for {total_tasks} tasks...")

        for idx, (task_id, task_data) in enumerate(dataset.items()):
            print(
                f"  [{idx + 1}/{total_tasks}] Generating variations for task {task_id}...",
                end=" ",
                flush=True,
            )
            # Check if the prompt field exists
            if prompt_field not in task_data:
                print(f"skipped (no '{prompt_field}' field)")
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
                varied_task["prompt_variation_id"] = i  # Track which variation this is
                varied_task["prompt_variation_strength"] = (
                    self.strength.value
                )  # Track strength level
                varied_tasks.append(varied_task)

            varied_dataset[task_id] = varied_tasks
            print(f"done ({len(variations)} variations)")

        print(f"Completed generating variations for all {total_tasks} tasks")
        return varied_dataset


def get_user_style_directive(strength: str) -> str:
    """
    Get the style directive for the simulated user's system prompt.

    This should be appended to the user simulator's system prompt to control
    HOW it communicates (not WHAT it says).

    Args:
        strength: Variation strength level (mild, medium, strong, naturalistic)

    Returns:
        Style directive string to append to user's system prompt
    """
    try:
        strength_enum = VariationStrength(strength.lower())
    except ValueError:
        return ""
    return USER_STYLE_DIRECTIVES.get(strength_enum, "")


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
        "gaia": "Question",
        "usaco": "problem_statement",
        "swebench_verified": "problem_statement",
        "swebench_verified_mini": "problem_statement",
        "appworld_test_normal": "instruction",
        "appworld_test_challenge": "instruction",
        "assistantbench": "task",
        "scicode": "problem_statement",
        "scicode_easy": "problem_statement",
        "scicode_hard": "problem_statement",
    }

    # TauBench now supported via instruction field
    if benchmark_name in ["taubench_retail", "taubench_airline"]:
        return "instruction"

    # Handle inspect benchmarks
    if benchmark_name.startswith("inspect_evals/"):
        # Most inspect benchmarks use 'input' or 'question'
        return "input"

    return prompt_field_map.get(benchmark_name, "Question")  # Default to 'Question'
