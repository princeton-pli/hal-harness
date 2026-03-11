"""
Tau-Bench Specific Structural Perturbations for R_struct Metric

This module provides meaningful structural perturbations for tau-bench benchmarks
by modifying:
1. Tool response formats (what the agent receives)
2. Tool definition parameter names (what the agent must use)
3. Data formats (dates, times, enums, nested structures)

These perturbations test whether agents can handle real-world API variations
while maintaining functional correctness.
"""

import re
import json
import copy
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum


class TauBenchPerturbationStrength(Enum):
    """Strength levels for tau-bench perturbations."""

    MILD = "mild"  # Only key naming convention changes
    MEDIUM = "medium"  # Naming + some structural changes + data format changes
    SEVERE = "severe"  # Complete restructuring + abbreviations + nested↔flat


@dataclass
class TauBenchPerturbationConfig:
    """Configuration for tau-bench specific perturbations."""

    # Key naming transformations
    key_case: str = "snake_case"  # snake_case, camelCase, PascalCase

    # Structural transformations
    flatten_nested: bool = (
        False  # Flatten nested objects (e.g., name.first_name → first_name)
    )
    nest_flat: bool = False  # Nest flat objects (e.g., first_name → name.first_name)
    wrap_responses: bool = False  # Wrap in {status: "success", data: ...}

    # Data format transformations
    time_format: str = "24h"  # 24h (14:00:00), 12h (2:00 PM), compact (1400)
    date_format: str = "iso"  # iso (2024-01-15), us (01/15/2024), compact (20240115)
    status_format: str = "lowercase"  # lowercase, uppercase, abbreviated, numeric
    cabin_format: str = (
        "full"  # full (basic_economy), abbreviated (Y/J/F), title (Basic Economy)
    )

    # Key abbreviations (for severe mode)
    use_abbreviations: bool = False

    # Parameter name mapping (for tool definitions)
    param_name_style: str = "original"  # original, camelCase, abbreviated

    @staticmethod
    def get_preset(
        strength: TauBenchPerturbationStrength,
    ) -> "TauBenchPerturbationConfig":
        """Get preset configuration for given strength."""
        if strength == TauBenchPerturbationStrength.MILD:
            return TauBenchPerturbationConfig(
                key_case="camelCase",
                param_name_style="camelCase",
            )
        elif strength == TauBenchPerturbationStrength.MEDIUM:
            return TauBenchPerturbationConfig(
                key_case="camelCase",
                wrap_responses=True,
                time_format="12h",
                date_format="us",
                status_format="uppercase",
                cabin_format="title",
                param_name_style="camelCase",
            )
        elif strength == TauBenchPerturbationStrength.SEVERE:
            return TauBenchPerturbationConfig(
                key_case="camelCase",
                nest_flat=True,
                wrap_responses=True,
                time_format="compact",
                date_format="compact",
                status_format="abbreviated",
                cabin_format="abbreviated",
                use_abbreviations=True,
                param_name_style="abbreviated",
            )
        else:
            return TauBenchPerturbationConfig()


# ============================================================================
# Key/Field Name Mappings
# ============================================================================

# Standard abbreviations for airline domain
AIRLINE_ABBREVIATIONS = {
    # Flight related
    "flight_number": "flt_no",
    "flight_type": "flt_type",
    "origin": "orig",
    "destination": "dest",
    "scheduled_departure_time_est": "dep_time",
    "scheduled_arrival_time_est": "arr_time",
    "available_seats": "avail_seats",
    # Passenger/User related
    "first_name": "fname",
    "last_name": "lname",
    "user_id": "uid",
    "email": "email",  # Keep as is
    "dob": "dob",  # Already abbreviated
    "address": "addr",
    "address1": "addr1",
    "address2": "addr2",
    "country": "ctry",
    "state": "st",
    "zip": "zip",  # Keep as is
    # Reservation related
    "reservation_id": "res_id",
    "created_at": "created",
    "payment_methods": "pay_methods",
    "payment_history": "pay_hist",
    "total_baggages": "bags",
    "nonfree_baggages": "paid_bags",
    "insurance": "ins",
    "passengers": "pax",
    "saved_passengers": "saved_pax",
    "membership": "member",
    # Cabin classes
    "basic_economy": "Y",
    "economy": "M",
    "business": "J",
    "first": "F",
    # Status values
    "available": "AVL",
    "confirmed": "CNF",
    "cancelled": "CXL",
    "pending": "PND",
    "completed": "CMP",
}

# Reverse mapping for tool parameter names
PARAM_ABBREVIATIONS = {
    "flight_number": "fltNo",
    "origin": "orig",
    "destination": "dest",
    "date": "dt",
    "user_id": "uid",
    "reservation_id": "resId",
    "passenger_id": "paxId",
    "cabin": "cls",
    "payment_id": "payId",
    "baggage_allowance": "bags",
}


class TauBenchPerturbator:
    """
    Applies structural perturbations to tau-bench tool responses and definitions.

    This class intercepts tool responses and modifies them according to the
    configuration, testing whether agents can handle API format variations.
    """

    def __init__(self, config: Optional[TauBenchPerturbationConfig] = None):
        self.config = config or TauBenchPerturbationConfig()
        self.applied_perturbations: List[Dict[str, Any]] = []

        # Build key mapping based on config
        self._key_transform_cache: Dict[str, str] = {}

    # ========================================================================
    # Main Entry Points
    # ========================================================================

    def perturb_tool_response(self, response: Any) -> Any:
        """
        Perturb a tool response (the data returned to the agent).

        Args:
            response: Raw tool response (can be dict, list, or string)

        Returns:
            Perturbed response
        """
        # Parse if string
        if isinstance(response, str):
            try:
                parsed = json.loads(response)
                perturbed = self._perturb_data(parsed)
                return json.dumps(perturbed)
            except json.JSONDecodeError:
                return response
        else:
            return self._perturb_data(response)

    def perturb_tool_definitions(
        self, tools_info: List[Dict]
    ) -> Tuple[List[Dict], Dict[str, Dict[str, str]]]:
        """
        Perturb tool definitions (parameter names the agent must use).

        Args:
            tools_info: List of OpenAI-style tool definitions

        Returns:
            Tuple of (perturbed_tools, param_mapping) where param_mapping
            maps tool_name -> {new_param_name: original_param_name}
        """
        if self.config.param_name_style == "original":
            return tools_info, {}

        perturbed_tools = []
        param_mapping = {}  # tool_name -> {new_name: old_name}

        for tool in tools_info:
            perturbed_tool = copy.deepcopy(tool)
            tool_name = tool.get("function", {}).get("name", "")

            if (
                "function" in perturbed_tool
                and "parameters" in perturbed_tool["function"]
            ):
                params = perturbed_tool["function"]["parameters"]
                if "properties" in params:
                    new_properties = {}
                    tool_param_map = {}
                    required = params.get("required", [])
                    new_required = []

                    for param_name, param_def in params["properties"].items():
                        new_name = self._transform_param_name(param_name)
                        new_properties[new_name] = param_def

                        if new_name != param_name:
                            tool_param_map[new_name] = param_name
                            self.applied_perturbations.append(
                                {
                                    "type": "tool_param_rename",
                                    "tool": tool_name,
                                    "original": param_name,
                                    "perturbed": new_name,
                                }
                            )

                        # Update required list
                        if param_name in required:
                            new_required.append(new_name)

                    params["properties"] = new_properties
                    if required:
                        params["required"] = new_required

                    if tool_param_map:
                        param_mapping[tool_name] = tool_param_map

            perturbed_tools.append(perturbed_tool)

        return perturbed_tools, param_mapping

    def reverse_param_mapping(
        self,
        tool_name: str,
        kwargs: Dict[str, Any],
        param_mapping: Dict[str, Dict[str, str]],
    ) -> Dict[str, Any]:
        """
        Reverse parameter name mapping for a tool call.

        When the agent calls a tool with perturbed param names, this converts
        them back to the original names that the environment expects.

        Args:
            tool_name: Name of the tool being called
            kwargs: Arguments the agent provided (with perturbed names)
            param_mapping: Mapping from perturb_tool_definitions

        Returns:
            kwargs with original parameter names
        """
        if tool_name not in param_mapping:
            return kwargs

        tool_map = param_mapping[tool_name]
        reversed_kwargs = {}

        for key, value in kwargs.items():
            # If this key was perturbed, use the original name
            original_key = tool_map.get(key, key)
            reversed_kwargs[original_key] = value

        return reversed_kwargs

    # ========================================================================
    # Internal Perturbation Methods
    # ========================================================================

    def _perturb_data(self, data: Any) -> Any:
        """Recursively perturb data structure."""
        if isinstance(data, dict):
            return self._perturb_dict(data)
        elif isinstance(data, list):
            return [self._perturb_data(item) for item in data]
        elif isinstance(data, str):
            return self._perturb_string_value(data)
        else:
            return data

    def _perturb_dict(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Perturb a dictionary (main workhorse)."""
        result = {}

        # First pass: transform all values and keys
        for key, value in data.items():
            # Transform the key name
            new_key = self._transform_key(key)

            # Transform the value
            new_value = self._perturb_value(key, value)

            result[new_key] = new_value

        # Apply structural transformations
        if self.config.nest_flat:
            result = self._apply_nesting(result)
        elif self.config.flatten_nested:
            result = self._apply_flattening(result)

        # Wrap response if configured
        if self.config.wrap_responses:
            result = self._wrap_response(result)

        return result

    def _perturb_value(self, key: str, value: Any) -> Any:
        """Perturb a value based on its key and type."""
        # Handle nested structures first
        if isinstance(value, dict):
            # Special handling for known nested structures
            if key in ["available_seats", "prices"]:
                return self._perturb_cabin_dict(value)
            elif key in ["name", "address"]:
                return self._perturb_data(value)
            else:
                return self._perturb_data(value)
        elif isinstance(value, list):
            return [self._perturb_data(item) for item in value]
        elif isinstance(value, str):
            # Context-aware string transformation
            return self._perturb_string_value(value, key)
        else:
            return value

    def _perturb_cabin_dict(self, cabin_data: Dict[str, Any]) -> Dict[str, Any]:
        """Perturb cabin class keys (basic_economy, economy, business)."""
        result = {}
        for cabin, value in cabin_data.items():
            new_cabin = self._transform_cabin_class(cabin)
            result[new_cabin] = value
        return result

    def _perturb_string_value(self, value: str, key: str = "") -> str:
        """Perturb string values based on context."""
        # Time format transformation
        if key.endswith("_time") or key.endswith("_time_est") or "time" in key.lower():
            return self._transform_time(value)

        # Date format transformation
        if key in ["dob", "date", "created_at"] or "date" in key.lower():
            return self._transform_date(value)

        # Status transformation
        if key == "status":
            return self._transform_status(value)

        # Cabin class in string context
        if key == "cabin" or value in ["basic_economy", "economy", "business", "first"]:
            return self._transform_cabin_class(value)

        return value

    # ========================================================================
    # Key Transformation
    # ========================================================================

    def _transform_key(self, key: str) -> str:
        """Transform a key name based on configuration."""
        if key in self._key_transform_cache:
            return self._key_transform_cache[key]

        original_key = key

        # Apply abbreviations first if enabled
        if self.config.use_abbreviations and key in AIRLINE_ABBREVIATIONS:
            key = AIRLINE_ABBREVIATIONS[key]

        # Then apply case transformation
        if self.config.key_case == "camelCase":
            key = self._to_camel_case(key)
        elif self.config.key_case == "PascalCase":
            key = self._to_pascal_case(key)

        if key != original_key:
            self._key_transform_cache[original_key] = key
            self.applied_perturbations.append(
                {"type": "key_rename", "original": original_key, "perturbed": key}
            )

        return key

    def _transform_param_name(self, param: str) -> str:
        """Transform a tool parameter name."""
        if self.config.param_name_style == "original":
            return param
        elif self.config.param_name_style == "camelCase":
            return self._to_camel_case(param)
        elif self.config.param_name_style == "abbreviated":
            if param in PARAM_ABBREVIATIONS:
                return PARAM_ABBREVIATIONS[param]
            return self._to_camel_case(param)
        return param

    def _to_camel_case(self, text: str) -> str:
        """Convert snake_case to camelCase."""
        components = text.split("_")
        return components[0].lower() + "".join(x.title() for x in components[1:])

    def _to_pascal_case(self, text: str) -> str:
        """Convert snake_case to PascalCase."""
        components = text.split("_")
        return "".join(x.title() for x in components)

    # ========================================================================
    # Data Format Transformations
    # ========================================================================

    def _transform_time(self, time_str: str) -> str:
        """Transform time format."""
        if self.config.time_format == "24h":
            return time_str  # Already 24h format typically

        # Parse HH:MM:SS format
        match = re.match(r"(\d{2}):(\d{2}):(\d{2})", time_str)
        if not match:
            return time_str

        hour, minute = int(match.group(1)), match.group(2)

        if self.config.time_format == "12h":
            period = "AM" if hour < 12 else "PM"
            hour_12 = hour % 12
            if hour_12 == 0:
                hour_12 = 12
            return f"{hour_12}:{minute} {period}"
        elif self.config.time_format == "compact":
            return f"{hour:02d}{minute}"

        return time_str

    def _transform_date(self, date_str: str) -> str:
        """Transform date format."""
        if self.config.date_format == "iso":
            return date_str  # Keep ISO format

        # Parse YYYY-MM-DD format
        match = re.match(r"(\d{4})-(\d{2})-(\d{2})", date_str)
        if not match:
            return date_str

        year, month, day = match.groups()

        if self.config.date_format == "us":
            return f"{month}/{day}/{year}"
        elif self.config.date_format == "compact":
            return f"{year}{month}{day}"

        return date_str

    def _transform_status(self, status: str) -> str:
        """Transform status values."""
        if self.config.status_format == "lowercase":
            return status.lower()
        elif self.config.status_format == "uppercase":
            return status.upper()
        elif self.config.status_format == "abbreviated":
            abbrev = AIRLINE_ABBREVIATIONS.get(status.lower())
            return abbrev if abbrev else status.upper()[:3]
        elif self.config.status_format == "numeric":
            status_map = {
                "available": "1",
                "confirmed": "2",
                "cancelled": "0",
                "pending": "3",
            }
            return status_map.get(status.lower(), status)

        return status

    def _transform_cabin_class(self, cabin: str) -> str:
        """Transform cabin class names."""
        if self.config.cabin_format == "full":
            return cabin
        elif self.config.cabin_format == "title":
            return cabin.replace("_", " ").title()
        elif self.config.cabin_format == "abbreviated":
            cabin_codes = {
                "basic_economy": "Y",
                "economy": "M",
                "business": "J",
                "first": "F",
            }
            return cabin_codes.get(cabin.lower(), cabin)

        return cabin

    # ========================================================================
    # Structural Transformations
    # ========================================================================

    def _apply_nesting(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Convert flat structure to nested (e.g., first_name → name.first)."""
        # Group by common prefixes
        groups = {}
        ungrouped = {}

        for key, value in data.items():
            # Check for groupable keys (snake_case with underscore)
            if "_" in key and not isinstance(value, dict):
                parts = key.split("_", 1)
                prefix, suffix = parts[0], parts[1]

                # Only group known patterns
                if prefix in ["first", "last", "scheduled"]:
                    # Remap to standard groups
                    if prefix in ["first", "last"] and suffix == "name":
                        group_name = "name"
                        new_key = prefix
                    elif prefix == "scheduled":
                        group_name = "schedule"
                        new_key = suffix.replace("_est", "")
                    else:
                        ungrouped[key] = value
                        continue

                    if group_name not in groups:
                        groups[group_name] = {}
                    groups[group_name][new_key] = value
                else:
                    ungrouped[key] = value
            else:
                ungrouped[key] = value

        # Merge groups back
        result = ungrouped.copy()
        result.update(groups)

        if groups:
            self.applied_perturbations.append(
                {"type": "structure_nested", "groups_created": list(groups.keys())}
            )

        return result

    def _apply_flattening(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Convert nested structure to flat (e.g., name.first → first_name)."""
        result = {}

        for key, value in data.items():
            if isinstance(value, dict) and key in ["name", "address", "schedule"]:
                # Flatten this nested structure
                for sub_key, sub_value in value.items():
                    flat_key = f"{key}_{sub_key}"
                    result[flat_key] = sub_value

                self.applied_perturbations.append(
                    {"type": "structure_flattened", "original_key": key}
                )
            else:
                result[key] = value

        return result

    def _wrap_response(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Wrap response in status/data structure."""
        # Don't double-wrap
        if "status" in data and "data" in data:
            return data

        self.applied_perturbations.append({"type": "response_wrapped"})

        return {"status": "success", "data": data}

    # ========================================================================
    # Utility Methods
    # ========================================================================

    def get_perturbation_summary(self) -> Dict[str, Any]:
        """Get summary of applied perturbations."""
        summary = {
            "total_perturbations": len(self.applied_perturbations),
            "by_type": {},
        }

        for p in self.applied_perturbations:
            ptype = p.get("type", "unknown")
            if ptype not in summary["by_type"]:
                summary["by_type"][ptype] = 0
            summary["by_type"][ptype] += 1

        return summary

    def reset_tracking(self):
        """Reset perturbation tracking."""
        self.applied_perturbations = []
        self._key_transform_cache = {}


# ============================================================================
# Environment Wrapper for Tool Call Interception
# ============================================================================


class PerturbedTauBenchEnv:
    """
    Wrapper for tau-bench environment that intercepts tool calls and responses.

    This wrapper:
    1. Perturbs tool definitions (parameter names)
    2. Intercepts tool calls to reverse parameter names
    3. Perturbs tool responses before returning to agent
    """

    def __init__(self, env, perturbator: TauBenchPerturbator):
        """
        Initialize wrapper.

        Args:
            env: Original tau-bench environment
            perturbator: TauBenchPerturbator instance
        """
        self._env = env
        self.perturbator = perturbator

        # Perturb tool definitions and store mapping
        original_tools = env.tools_info
        self._perturbed_tools, self._param_mapping = (
            perturbator.perturb_tool_definitions(original_tools)
        )

        # Track original wiki for comparison
        self._original_wiki = env.wiki

    @property
    def tools_info(self) -> List[Dict]:
        """Return perturbed tool definitions."""
        return self._perturbed_tools

    @property
    def wiki(self) -> str:
        """Return perturbed wiki (knowledge base)."""
        return self.perturbator.perturb_tool_response(self._original_wiki)

    def __getattr__(self, name):
        """Delegate all other attributes to wrapped environment."""
        return getattr(self._env, name)

    def step(self, action):
        """
        Intercept step to reverse param names and perturb response.

        This is the key method that makes perturbations transparent to the
        underlying environment while testing the agent's robustness.
        """
        # Reverse parameter names if needed
        if hasattr(action, "kwargs") and action.name in self._param_mapping:
            original_kwargs = self.perturbator.reverse_param_mapping(
                action.name, action.kwargs, self._param_mapping
            )
            # Create modified action with original param names
            action = self._create_action_with_kwargs(action, original_kwargs)

        # Execute original step
        result = self._env.step(action)

        # Perturb the response if it contains tool output
        if isinstance(result, tuple):
            # Assuming (observation, reward, done, info) format
            obs, *rest = result
            perturbed_obs = self.perturbator.perturb_tool_response(obs)
            return (perturbed_obs, *rest)
        else:
            return self.perturbator.perturb_tool_response(result)

    def _create_action_with_kwargs(self, action, new_kwargs):
        """Create a copy of action with modified kwargs."""
        # This depends on the action class structure in tau-bench
        # For now, try direct attribute modification
        import copy

        new_action = copy.copy(action)
        new_action.kwargs = new_kwargs
        return new_action


# ============================================================================
# Factory Functions
# ============================================================================


def create_taubench_perturbator(
    strength: str = "medium", custom_config: Optional[Dict] = None
) -> TauBenchPerturbator:
    """
    Factory function to create tau-bench perturbator.

    Args:
        strength: Perturbation strength (mild, medium, severe)
        custom_config: Optional custom configuration dict

    Returns:
        Configured TauBenchPerturbator
    """
    if custom_config:
        config = TauBenchPerturbationConfig(**custom_config)
    else:
        try:
            strength_enum = TauBenchPerturbationStrength(strength)
        except ValueError:
            strength_enum = TauBenchPerturbationStrength.MEDIUM
        config = TauBenchPerturbationConfig.get_preset(strength_enum)

    return TauBenchPerturbator(config)
