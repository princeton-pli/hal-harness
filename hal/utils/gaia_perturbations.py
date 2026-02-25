"""
GAIA-Specific Structural Perturbations for R_struct Metric

This module provides semantic-preserving structural perturbations for the GAIA benchmark
by modifying:
1. Question text format (case, whitespace, punctuation)
2. Instruction format and phrasing
3. Data formats within questions (numbers, dates)
4. Tool output formats (search results, webpage content)

These perturbations test whether agents can handle variations in how questions
and information are presented while maintaining answer correctness.
"""

import re
import random
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum


class GaiaPerturbationStrength(Enum):
    """Strength levels for GAIA perturbations."""

    MILD = "mild"  # Only formatting changes (case, whitespace)
    MEDIUM = "medium"  # Formatting + instruction rephrasing + data format changes
    SEVERE = "severe"  # All changes + noise injection + tool output perturbations


@dataclass
class GaiaPerturbationConfig:
    """Configuration for GAIA-specific perturbations."""

    # Question text perturbations
    question_case: str = "original"  # original, lowercase, uppercase, mixed
    normalize_whitespace: bool = False  # Normalize multiple spaces to single
    add_noise_words: bool = False  # Add filler words like "please", "kindly"

    # Instruction format perturbations
    instruction_style: str = "original"  # original, formal, casual, terse
    reorder_instructions: bool = False  # Reorder bullet points in instructions

    # Data format perturbations (in questions)
    number_format: str = "original"  # original, words, with_commas
    date_format: str = "original"  # original, verbose, compact

    # Tool output perturbations
    perturb_search_results: bool = False  # Add noise to search result formatting
    perturb_webpage_content: bool = False  # Add noise to webpage content
    wrap_tool_responses: bool = False  # Wrap responses in metadata structure

    # Noise injection
    add_irrelevant_context: bool = False  # Add irrelevant sentences to context

    @staticmethod
    def get_preset(strength: GaiaPerturbationStrength) -> "GaiaPerturbationConfig":
        """Get preset configuration for given strength."""
        if strength == GaiaPerturbationStrength.MILD:
            return GaiaPerturbationConfig(
                question_case="lowercase",
                normalize_whitespace=True,
            )
        elif strength == GaiaPerturbationStrength.MEDIUM:
            return GaiaPerturbationConfig(
                question_case="lowercase",
                normalize_whitespace=True,
                add_noise_words=True,
                instruction_style="formal",
                reorder_instructions=True,
                number_format="with_commas",
                date_format="verbose",
            )
        elif strength == GaiaPerturbationStrength.SEVERE:
            return GaiaPerturbationConfig(
                question_case="mixed",
                normalize_whitespace=True,
                add_noise_words=True,
                instruction_style="terse",
                reorder_instructions=True,
                number_format="words",
                date_format="verbose",
                perturb_search_results=True,
                perturb_webpage_content=True,
                wrap_tool_responses=True,
                add_irrelevant_context=True,
            )
        else:
            return GaiaPerturbationConfig()


# ============================================================================
# Word/Phrase Mappings
# ============================================================================

# Noise words to inject
NOISE_PREFIXES = [
    "Please",
    "Kindly",
    "Could you",
    "I need you to",
    "Help me to",
]

NOISE_SUFFIXES = [
    "Thank you.",
    "Thanks in advance.",
    "I appreciate your help.",
]

# Instruction style variants
INSTRUCTION_STYLES = {
    "original": {
        "return_only": "Return only your answer",
        "short_phrase": "a short phrase with as few words as possible",
        "if_number": "If the answer is a number, return only the number without any units",
        "if_string": "If the answer is a string, don't include articles",
        "comma_list": "If the answer is a comma separated list",
    },
    "formal": {
        "return_only": "Please provide exclusively your final answer",
        "short_phrase": "a concise phrase using minimal words",
        "if_number": "In the case of numerical answers, provide only the numeric value excluding units",
        "if_string": "For string-based answers, omit articles",
        "comma_list": "For answers consisting of comma-delimited items",
    },
    "casual": {
        "return_only": "Just give me your answer",
        "short_phrase": "keep it short and simple",
        "if_number": "For numbers, just the number, no units",
        "if_string": "For text, skip the articles",
        "comma_list": "If it's a list",
    },
    "terse": {
        "return_only": "Answer only",
        "short_phrase": "brief phrase",
        "if_number": "Numbers: no units",
        "if_string": "Strings: no articles",
        "comma_list": "Lists",
    },
}

# Number to word mappings for small numbers
NUMBER_WORDS = {
    0: "zero",
    1: "one",
    2: "two",
    3: "three",
    4: "four",
    5: "five",
    6: "six",
    7: "seven",
    8: "eight",
    9: "nine",
    10: "ten",
    11: "eleven",
    12: "twelve",
}

# Irrelevant context sentences for noise injection
IRRELEVANT_SENTENCES = [
    "The weather has been quite variable lately.",
    "Many people find this topic interesting.",
    "This information may be useful for your research.",
    "Several studies have explored related topics.",
    "Context is important for understanding complex questions.",
]


class GaiaPerturbator:
    """Apply structural perturbations to GAIA benchmark inputs."""

    def __init__(self, config: Optional[GaiaPerturbationConfig] = None):
        """
        Initialize perturbator.

        Args:
            config: Configuration for perturbations (None = default)
        """
        self.config = config or GaiaPerturbationConfig()
        self.applied_perturbations: List[Dict[str, Any]] = []
        self._rng = random.Random(42)  # Reproducible randomness

    def set_seed(self, seed: int):
        """Set random seed for reproducibility."""
        self._rng = random.Random(seed)

    # ========== Question Text Perturbations ==========

    def perturb_question(self, question: str) -> str:
        """
        Apply question text perturbations.

        Args:
            question: Original question text

        Returns:
            Perturbed question text
        """
        original = question
        perturbed = question

        # Apply case transformation
        if self.config.question_case == "lowercase":
            perturbed = perturbed.lower()
        elif self.config.question_case == "uppercase":
            perturbed = perturbed.upper()
        elif self.config.question_case == "mixed":
            perturbed = self._apply_mixed_case(perturbed)

        # Normalize whitespace
        if self.config.normalize_whitespace:
            perturbed = self._normalize_whitespace(perturbed)

        # Add noise words
        if self.config.add_noise_words:
            perturbed = self._add_noise_words(perturbed)

        # Apply date format changes BEFORE number formatting to avoid corrupting dates
        if self.config.date_format != "original":
            perturbed = self._format_dates(perturbed)

        # Apply number format changes (after date formatting to preserve date structure)
        if self.config.number_format != "original":
            perturbed = self._format_numbers(perturbed)

        # Add irrelevant context
        if self.config.add_irrelevant_context:
            perturbed = self._add_irrelevant_context(perturbed)

        if perturbed != original:
            self.applied_perturbations.append(
                {
                    "type": "question_text",
                    "original": original[:100] + "..."
                    if len(original) > 100
                    else original,
                    "perturbed": perturbed[:100] + "..."
                    if len(perturbed) > 100
                    else perturbed,
                }
            )

        return perturbed

    def _apply_mixed_case(self, text: str) -> str:
        """Apply random mixed case to text."""
        result = []
        for i, char in enumerate(text):
            if char.isalpha():
                if self._rng.random() > 0.5:
                    result.append(char.upper())
                else:
                    result.append(char.lower())
            else:
                result.append(char)
        return "".join(result)

    def _normalize_whitespace(self, text: str) -> str:
        """Normalize multiple spaces to single space."""
        return " ".join(text.split())

    def _add_noise_words(self, text: str) -> str:
        """Add filler words to the question."""
        # Add prefix
        if self._rng.random() > 0.5:
            prefix = self._rng.choice(NOISE_PREFIXES)
            if not text[0].isupper():
                text = text[0].lower() + text[1:]
            text = f"{prefix} {text}"

        # Add suffix
        if self._rng.random() > 0.5:
            suffix = self._rng.choice(NOISE_SUFFIXES)
            if not text.endswith(".") and not text.endswith("?"):
                text = text + "."
            text = f"{text} {suffix}"

        return text

    def _format_numbers(self, text: str) -> str:
        """Transform number formats in text."""
        if self.config.number_format == "with_commas":
            # Add commas to large numbers (e.g., 10000 -> 10,000)
            # Exclude 4-digit years (1900-2100) to avoid corrupting dates
            def add_commas(match):
                num = int(match.group(0))
                # Skip if it looks like a year (1900-2100)
                if 1900 <= num <= 2100:
                    return match.group(0)
                return f"{num:,}"

            # Only apply to numbers with 5+ digits, or 4-digit non-years
            text = re.sub(r"\b\d{4,}\b", add_commas, text)

        elif self.config.number_format == "words":
            # Convert small numbers to words (only standalone numbers, not part of dates)
            def num_to_word(match):
                num = int(match.group(0))
                # Skip if preceded by hyphen (likely part of date like 01-15)
                start = match.start()
                if start > 0 and text[start - 1] == "-":
                    return match.group(0)
                if num in NUMBER_WORDS:
                    return NUMBER_WORDS[num]
                return match.group(0)

            text = re.sub(r"\b\d{1,2}\b", num_to_word, text)

        return text

    def _format_dates(self, text: str) -> str:
        """Transform date formats in text."""
        if self.config.date_format == "verbose":
            # Convert YYYY-MM-DD to Month Day, Year
            month_names = [
                "January",
                "February",
                "March",
                "April",
                "May",
                "June",
                "July",
                "August",
                "September",
                "October",
                "November",
                "December",
            ]

            def iso_to_verbose(match):
                year, month, day = match.groups()
                month_name = month_names[int(month) - 1]
                return f"{month_name} {int(day)}, {year}"

            text = re.sub(r"(\d{4})-(\d{2})-(\d{2})", iso_to_verbose, text)

        elif self.config.date_format == "compact":
            # Convert YYYY-MM-DD to YYYYMMDD
            text = re.sub(r"(\d{4})-(\d{2})-(\d{2})", r"\1\2\3", text)

        return text

    def _add_irrelevant_context(self, text: str) -> str:
        """Add irrelevant sentences to the question."""
        sentence = self._rng.choice(IRRELEVANT_SENTENCES)

        # Insert in the middle or at the end
        if self._rng.random() > 0.5:
            # Find a sentence boundary
            sentences = text.split(". ")
            if len(sentences) > 1:
                insert_pos = self._rng.randint(1, len(sentences) - 1)
                sentences.insert(insert_pos, sentence.rstrip("."))
                text = ". ".join(sentences)
            else:
                text = f"{text} {sentence}"
        else:
            text = f"{text} {sentence}"

        return text

    # ========== Instruction Perturbations ==========

    def perturb_instructions(self, instructions: str) -> str:
        """
        Apply instruction format perturbations.

        Args:
            instructions: Original instruction text

        Returns:
            Perturbed instruction text
        """
        original = instructions
        perturbed = instructions

        # Apply instruction style
        if self.config.instruction_style != "original":
            perturbed = self._apply_instruction_style(perturbed)

        # Reorder instruction bullet points
        if self.config.reorder_instructions:
            perturbed = self._reorder_bullets(perturbed)

        if perturbed != original:
            self.applied_perturbations.append(
                {
                    "type": "instructions",
                    "original": original[:100] + "..."
                    if len(original) > 100
                    else original,
                    "perturbed": perturbed[:100] + "..."
                    if len(perturbed) > 100
                    else perturbed,
                }
            )

        return perturbed

    def _apply_instruction_style(self, text: str) -> str:
        """Apply instruction style transformations."""
        style = INSTRUCTION_STYLES.get(self.config.instruction_style, {})
        original_style = INSTRUCTION_STYLES["original"]

        for key, original_phrase in original_style.items():
            if key in style and original_phrase in text:
                text = text.replace(original_phrase, style[key])

        return text

    def _reorder_bullets(self, text: str) -> str:
        """Reorder bullet points in instructions."""
        # Find bullet point sections (lines starting with -)
        lines = text.split("\n")
        bullet_indices = [
            i for i, line in enumerate(lines) if line.strip().startswith("-")
        ]

        if len(bullet_indices) > 1:
            # Extract bullet lines
            bullets = [lines[i] for i in bullet_indices]
            # Shuffle them
            self._rng.shuffle(bullets)
            # Put them back
            for i, idx in enumerate(bullet_indices):
                lines[idx] = bullets[i]
            text = "\n".join(lines)

        return text

    # ========== Tool Output Perturbations ==========

    def perturb_tool_output(self, output: Any, tool_name: str = "") -> Any:
        """
        Apply perturbations to tool outputs.

        Args:
            output: Original tool output
            tool_name: Name of the tool (for tool-specific perturbations)

        Returns:
            Perturbed tool output
        """
        if output is None:
            return output

        # Convert to string for manipulation
        if isinstance(output, str):
            perturbed = output

            # Apply search result perturbations
            if self.config.perturb_search_results and "search" in tool_name.lower():
                perturbed = self._perturb_search_results(perturbed)

            # Apply webpage content perturbations
            if self.config.perturb_webpage_content and "webpage" in tool_name.lower():
                perturbed = self._perturb_webpage_content(perturbed)

            # Wrap in metadata structure
            if self.config.wrap_tool_responses:
                perturbed = self._wrap_response(perturbed)

            if perturbed != output:
                self.applied_perturbations.append(
                    {
                        "type": f"tool_output_{tool_name}",
                        "original": str(output)[:50] + "..."
                        if len(str(output)) > 50
                        else str(output),
                        "perturbed": str(perturbed)[:50] + "..."
                        if len(str(perturbed)) > 50
                        else str(perturbed),
                    }
                )

            return perturbed

        return output

    def _perturb_search_results(self, text: str) -> str:
        """Add noise to search result formatting."""
        # Add extra whitespace between results
        text = re.sub(r"\n(\d+\.)", r"\n\n\1", text)

        # Add "[Result]" prefix markers
        text = re.sub(r"^(\d+\.)", r"[Result \1]", text, flags=re.MULTILINE)

        return text

    def _perturb_webpage_content(self, text: str) -> str:
        """Add noise to webpage content."""
        # Add navigation breadcrumbs noise
        noise_header = "[Navigation: Home > Category > Page]\n\n"
        text = noise_header + text

        # Add footer noise
        noise_footer = "\n\n[Footer: Copyright | Privacy Policy | Terms of Use]"
        text = text + noise_footer

        return text

    def _wrap_response(self, text: str) -> str:
        """Wrap response in metadata structure."""
        return f"[Response Status: OK]\n[Data Begin]\n{text}\n[Data End]"

    # ========== Full Prompt Perturbation ==========

    def perturb_gaia_prompt(self, prompt: str, question: str) -> Tuple[str, str]:
        """
        Apply perturbations to a full GAIA prompt.

        Args:
            prompt: Full prompt including instructions and question
            question: Just the question part

        Returns:
            Tuple of (perturbed_prompt, perturbed_question)
        """
        # Perturb the question
        perturbed_question = self.perturb_question(question)

        # Perturb the prompt by replacing original question with perturbed
        perturbed_prompt = prompt.replace(question, perturbed_question)

        # Apply instruction perturbations to the instruction part
        instruction_part = prompt.replace(question, "").strip()
        perturbed_instruction = self.perturb_instructions(instruction_part)
        perturbed_prompt = perturbed_instruction + "\n\n" + perturbed_question

        return perturbed_prompt, perturbed_question

    # ========== Statistics and Reporting ==========

    def get_perturbation_summary(self) -> Dict[str, Any]:
        """Get summary of applied perturbations."""
        by_type = {}
        for p in self.applied_perturbations:
            ptype = p.get("type", "unknown")
            by_type[ptype] = by_type.get(ptype, 0) + 1

        return {
            "total_perturbations": len(self.applied_perturbations),
            "by_type": by_type,
        }

    def get_config_dict(self) -> Dict[str, Any]:
        """Get configuration as dictionary."""
        return {
            "question_case": self.config.question_case,
            "normalize_whitespace": self.config.normalize_whitespace,
            "add_noise_words": self.config.add_noise_words,
            "instruction_style": self.config.instruction_style,
            "reorder_instructions": self.config.reorder_instructions,
            "number_format": self.config.number_format,
            "date_format": self.config.date_format,
            "perturb_search_results": self.config.perturb_search_results,
            "perturb_webpage_content": self.config.perturb_webpage_content,
            "wrap_tool_responses": self.config.wrap_tool_responses,
            "add_irrelevant_context": self.config.add_irrelevant_context,
        }

    def reset(self):
        """Reset applied perturbations tracking."""
        self.applied_perturbations = []


def create_gaia_perturbator(strength: str = "medium") -> GaiaPerturbator:
    """
    Factory function to create a GAIA perturbator with preset strength.

    Args:
        strength: Perturbation strength ("mild", "medium", "severe")

    Returns:
        Configured GaiaPerturbator
    """
    strength_enum = GaiaPerturbationStrength(strength.lower())
    config = GaiaPerturbationConfig.get_preset(strength_enum)
    return GaiaPerturbator(config=config)


# ============================================================================
# Tool Wrapper for Applying Perturbations to Tool Outputs
# ============================================================================


class PerturbedToolWrapper:
    """Wrapper that applies perturbations to tool outputs."""

    def __init__(self, original_tool: Any, perturbator: GaiaPerturbator):
        """
        Initialize wrapper.

        Args:
            original_tool: Original smolagents tool
            perturbator: GaiaPerturbator instance
        """
        self.original_tool = original_tool
        self.perturbator = perturbator

        # Copy attributes from original tool
        self.name = getattr(original_tool, "name", "unknown")
        self.description = getattr(original_tool, "description", "")
        self.inputs = getattr(original_tool, "inputs", {})
        self.output_type = getattr(original_tool, "output_type", "string")

    def forward(self, *args, **kwargs) -> Any:
        """Execute tool and perturb output."""
        result = self.original_tool.forward(*args, **kwargs)
        return self.perturbator.perturb_tool_output(result, self.name)

    def __call__(self, *args, **kwargs) -> Any:
        """Execute tool and perturb output."""
        if hasattr(self.original_tool, "__call__"):
            result = self.original_tool(*args, **kwargs)
        else:
            result = self.original_tool.forward(*args, **kwargs)
        return self.perturbator.perturb_tool_output(result, self.name)


def wrap_tools_with_perturbation(
    tools: List[Any], perturbator: GaiaPerturbator
) -> List[Any]:
    """
    Wrap a list of tools with perturbation wrappers.

    Args:
        tools: List of smolagents tools
        perturbator: GaiaPerturbator instance

    Returns:
        List of wrapped tools
    """
    return [PerturbedToolWrapper(tool, perturbator) for tool in tools]


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    # Example 1: Perturb a question
    perturbator = create_gaia_perturbator(strength="medium")

    question = "What is the population of Paris in 2024-01-15?"
    perturbed = perturbator.perturb_question(question)

    print("Original:", question)
    print("Perturbed:", perturbed)
    print()

    # Example 2: Full prompt perturbation
    prompt = """Please answer the question below. You should:

- Return only your answer, which should be a number, or a short phrase with as few words as possible.
- If the answer is a number, return only the number without any units.

Here is the question:

What is the GDP of France in 2023?"""

    question = "What is the GDP of France in 2023?"
    perturbed_prompt, perturbed_q = perturbator.perturb_gaia_prompt(prompt, question)

    print("Perturbed Prompt:")
    print(perturbed_prompt)
    print()

    # Show statistics
    print("Perturbation Summary:", perturbator.get_perturbation_summary())
