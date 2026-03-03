"""
Structural Perturbation Framework for R_struct Metric

Applies semantic-preserving structural perturbations to evaluate agent robustness
to environmental changes (API formats, database schemas, file paths, data formats).
"""

import re
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass
from enum import Enum


class PerturbationType(Enum):
    """Types of structural perturbations."""

    API = "api"
    DATABASE = "database"
    FILE = "file"
    DATA_FORMAT = "data_format"
    ALL = "all"


class PerturbationStrength(Enum):
    """Strength of perturbations."""

    MILD = "mild"  # Only naming conventions
    MEDIUM = "medium"  # Naming + structure
    SEVERE = "severe"  # Complete restructuring


@dataclass
class PerturbationConfig:
    """Configuration for structural perturbations."""

    # API perturbations
    api_endpoint_style: str = "original"  # original, versioned, shortened
    api_parameter_case: str = "snake_case"  # snake_case, camelCase, kebab-case
    api_response_wrapper: bool = False  # Wrap responses in {status, data} structure

    # Database perturbations
    db_column_naming: str = "original"  # original, abbreviated, camelCase
    db_table_naming: str = "original"  # original, prefixed, suffixed
    db_schema_style: str = "flat"  # flat, nested

    # File perturbations
    file_path_depth: int = 0  # +N adds directories, -N removes directories
    file_naming_case: str = "snake_case"  # snake_case, camelCase, PascalCase
    file_format: Optional[str] = None  # json, csv, xml, yaml (None = keep original)

    # Data format perturbations
    date_format: str = "iso"  # iso (2024-01-15), us (01/15/2024), eu (15/01/2024), unix
    number_format: str = "numeric"  # numeric, string, string_with_commas
    boolean_format: str = "bool"  # bool, string, numeric, yes_no

    @staticmethod
    def from_dict(config: Dict[str, Any]) -> "PerturbationConfig":
        """Create config from dictionary."""
        return PerturbationConfig(
            **{k: v for k, v in config.items() if hasattr(PerturbationConfig, k)}
        )

    @staticmethod
    def get_preset(strength: PerturbationStrength) -> "PerturbationConfig":
        """Get preset configuration for given strength."""
        if strength == PerturbationStrength.MILD:
            return PerturbationConfig(
                api_parameter_case="camelCase",
                db_column_naming="camelCase",
                file_naming_case="camelCase",
            )
        elif strength == PerturbationStrength.MEDIUM:
            return PerturbationConfig(
                api_endpoint_style="versioned",
                api_parameter_case="camelCase",
                api_response_wrapper=True,
                db_column_naming="abbreviated",
                db_table_naming="prefixed",
                file_path_depth=1,
                file_naming_case="camelCase",
                date_format="us",
                number_format="string_with_commas",
            )
        elif strength == PerturbationStrength.SEVERE:
            return PerturbationConfig(
                api_endpoint_style="shortened",
                api_parameter_case="kebab-case",
                api_response_wrapper=True,
                db_column_naming="abbreviated",
                db_table_naming="suffixed",
                db_schema_style="nested",
                file_path_depth=2,
                file_naming_case="PascalCase",
                file_format="csv",
                date_format="unix",
                number_format="string_with_commas",
                boolean_format="yes_no",
            )
        else:
            return PerturbationConfig()


class StructuralPerturbator:
    """Apply structural perturbations to benchmark environments."""

    def __init__(
        self,
        perturbation_type: Union[PerturbationType, str],
        config: Optional[PerturbationConfig] = None,
    ):
        """
        Initialize perturbator.

        Args:
            perturbation_type: Type of perturbations to apply
            config: Configuration for perturbations (None = default)
        """
        if isinstance(perturbation_type, str):
            perturbation_type = PerturbationType(perturbation_type)

        self.perturbation_type = perturbation_type
        self.config = config or PerturbationConfig()

        # Track applied perturbations for reporting
        self.applied_perturbations: List[Dict[str, Any]] = []

    # ========== API Perturbations ==========

    def perturb_api_endpoint(self, endpoint: str) -> str:
        """
        Apply endpoint structure perturbations.

        Args:
            endpoint: Original endpoint (e.g., "/api/v1/users/123")

        Returns:
            Perturbed endpoint
        """
        if self.perturbation_type not in [PerturbationType.API, PerturbationType.ALL]:
            return endpoint

        original = endpoint

        if self.config.api_endpoint_style == "versioned":
            # Change version or add version
            if "/v1/" in endpoint:
                endpoint = endpoint.replace("/v1/", "/v2/")
            elif "/api/" in endpoint and "/v" not in endpoint:
                endpoint = endpoint.replace("/api/", "/api/v1/")

        elif self.config.api_endpoint_style == "shortened":
            # Remove /api/ prefix or version
            endpoint = endpoint.replace("/api/v1/", "/")
            endpoint = endpoint.replace("/api/", "/")

        if endpoint != original:
            self.applied_perturbations.append(
                {"type": "api_endpoint", "original": original, "perturbed": endpoint}
            )

        return endpoint

    def perturb_api_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply parameter naming perturbations.

        Args:
            params: Original parameters

        Returns:
            Perturbed parameters
        """
        if self.perturbation_type not in [PerturbationType.API, PerturbationType.ALL]:
            return params

        perturbed = {}

        for key, value in params.items():
            new_key = self._convert_case(key, self.config.api_parameter_case)

            # Recursively perturb nested dictionaries
            if isinstance(value, dict):
                value = self.perturb_api_params(value)

            perturbed[new_key] = value

            if new_key != key:
                self.applied_perturbations.append(
                    {"type": "api_param", "original": key, "perturbed": new_key}
                )

        return perturbed

    def perturb_api_response(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply response structure perturbations.

        Args:
            response: Original response

        Returns:
            Perturbed response
        """
        if self.perturbation_type not in [PerturbationType.API, PerturbationType.ALL]:
            return response

        # First perturb keys recursively
        perturbed = {}
        for key, value in response.items():
            new_key = self._convert_case(key, self.config.api_parameter_case)

            if isinstance(value, dict):
                value = self.perturb_api_response(value)
            elif isinstance(value, list):
                value = [
                    self.perturb_api_response(item) if isinstance(item, dict) else item
                    for item in value
                ]

            perturbed[new_key] = value

        # Apply response wrapper if configured
        if self.config.api_response_wrapper:
            perturbed = {"status": "success", "data": perturbed}
            self.applied_perturbations.append(
                {"type": "api_response_wrapper", "wrapped": True}
            )

        return perturbed

    # ========== Database Perturbations ==========

    def perturb_database_column(self, column_name: str) -> str:
        """
        Apply column naming perturbations.

        Args:
            column_name: Original column name

        Returns:
            Perturbed column name
        """
        if self.perturbation_type not in [
            PerturbationType.DATABASE,
            PerturbationType.ALL,
        ]:
            return column_name

        original = column_name

        if self.config.db_column_naming == "abbreviated":
            # Common abbreviations
            abbreviations = {
                "user_id": "uid",
                "first_name": "fname",
                "last_name": "lname",
                "email_address": "email",
                "phone_number": "phone",
                "created_at": "created",
                "updated_at": "updated",
            }
            column_name = abbreviations.get(column_name, column_name)

        elif self.config.db_column_naming == "camelCase":
            column_name = self._convert_case(column_name, "camelCase")

        if column_name != original:
            self.applied_perturbations.append(
                {"type": "db_column", "original": original, "perturbed": column_name}
            )

        return column_name

    def perturb_database_table(self, table_name: str) -> str:
        """
        Apply table naming perturbations.

        Args:
            table_name: Original table name

        Returns:
            Perturbed table name
        """
        if self.perturbation_type not in [
            PerturbationType.DATABASE,
            PerturbationType.ALL,
        ]:
            return table_name

        original = table_name

        if self.config.db_table_naming == "prefixed":
            if not table_name.startswith("tbl_"):
                table_name = f"tbl_{table_name}"

        elif self.config.db_table_naming == "suffixed":
            if not table_name.endswith("_table"):
                table_name = f"{table_name}_records"

        if table_name != original:
            self.applied_perturbations.append(
                {"type": "db_table", "original": original, "perturbed": table_name}
            )

        return table_name

    def perturb_database_schema(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply schema structure perturbations (flat vs nested).

        Args:
            data: Original data structure

        Returns:
            Perturbed data structure
        """
        if self.perturbation_type not in [
            PerturbationType.DATABASE,
            PerturbationType.ALL,
        ]:
            return data

        if self.config.db_schema_style == "nested":
            # Convert flat structure to nested
            # Example: {user_id: 1, user_name: "John"} -> {user: {id: 1, name: "John"}}
            nested = {}
            for key, value in data.items():
                if "_" in key:
                    parts = key.split("_", 1)
                    prefix, suffix = parts[0], parts[1]

                    if prefix not in nested:
                        nested[prefix] = {}
                    nested[prefix][suffix] = value
                else:
                    nested[key] = value

            if nested != data:
                self.applied_perturbations.append(
                    {"type": "db_schema", "style": "nested"}
                )

            return nested

        return data

    # ========== File Perturbations ==========

    def perturb_file_path(self, path: str) -> str:
        """
        Apply file path perturbations.

        Args:
            path: Original file path

        Returns:
            Perturbed file path
        """
        if self.perturbation_type not in [PerturbationType.FILE, PerturbationType.ALL]:
            return path

        original = path
        parts = path.split("/")

        # Adjust path depth
        if self.config.file_path_depth > 0:
            # Add directories
            for i in range(self.config.file_path_depth):
                parts.insert(-1, f"data{i + 1}")
        elif self.config.file_path_depth < 0:
            # Remove directories (keep at least filename)
            remove_count = min(abs(self.config.file_path_depth), len(parts) - 1)
            parts = parts[remove_count:]

        # Change filename case
        if len(parts) > 0:
            filename = parts[-1]
            name, ext = filename.rsplit(".", 1) if "." in filename else (filename, "")
            name = self._convert_case(name, self.config.file_naming_case)
            parts[-1] = f"{name}.{ext}" if ext else name

        # Change file format
        if self.config.file_format and len(parts) > 0:
            filename = parts[-1]
            if "." in filename:
                name, _ = filename.rsplit(".", 1)
                parts[-1] = f"{name}.{self.config.file_format}"

        path = "/".join(parts)

        if path != original:
            self.applied_perturbations.append(
                {"type": "file_path", "original": original, "perturbed": path}
            )

        return path

    # ========== Data Format Perturbations ==========

    def perturb_date(self, date_str: str) -> Union[str, int]:
        """
        Apply date format perturbations.

        Args:
            date_str: Original date string (ISO format: 2024-01-15)

        Returns:
            Perturbed date
        """
        if self.perturbation_type not in [
            PerturbationType.DATA_FORMAT,
            PerturbationType.ALL,
        ]:
            return date_str

        # Parse ISO date
        match = re.match(r"(\d{4})-(\d{2})-(\d{2})", date_str)
        if not match:
            return date_str

        year, month, day = match.groups()

        if self.config.date_format == "us":
            return f"{month}/{day}/{year}"
        elif self.config.date_format == "eu":
            return f"{day}/{month}/{year}"
        elif self.config.date_format == "unix":
            # Simplified: just return a timestamp (not accurate conversion)
            return int(f"{year}{month}{day}")

        return date_str

    def perturb_number(self, number: Union[int, float]) -> Union[int, float, str]:
        """
        Apply number format perturbations.

        Args:
            number: Original number

        Returns:
            Perturbed number
        """
        if self.perturbation_type not in [
            PerturbationType.DATA_FORMAT,
            PerturbationType.ALL,
        ]:
            return number

        if self.config.number_format == "string":
            return str(number)
        elif self.config.number_format == "string_with_commas":
            if isinstance(number, float):
                return f"{number:,.2f}"
            else:
                return f"{number:,}"

        return number

    def perturb_boolean(self, value: bool) -> Union[bool, str, int]:
        """
        Apply boolean format perturbations.

        Args:
            value: Original boolean

        Returns:
            Perturbed boolean
        """
        if self.perturbation_type not in [
            PerturbationType.DATA_FORMAT,
            PerturbationType.ALL,
        ]:
            return value

        if self.config.boolean_format == "string":
            return "true" if value else "false"
        elif self.config.boolean_format == "numeric":
            return 1 if value else 0
        elif self.config.boolean_format == "yes_no":
            return "yes" if value else "no"

        return value

    def perturb_data(self, data: Any) -> Any:
        """
        Apply all data format perturbations recursively.

        Args:
            data: Original data (any type)

        Returns:
            Perturbed data
        """
        if isinstance(data, dict):
            return {k: self.perturb_data(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self.perturb_data(item) for item in data]
        elif isinstance(data, bool):
            return self.perturb_boolean(data)
        elif isinstance(data, (int, float)):
            return self.perturb_number(data)
        elif isinstance(data, str):
            # Try to detect and perturb dates
            if re.match(r"\d{4}-\d{2}-\d{2}", data):
                return self.perturb_date(data)
            return data
        else:
            return data

    # ========== Utility Methods ==========

    def _convert_case(self, text: str, target_case: str) -> str:
        """Convert text between naming conventions."""
        if target_case == "snake_case":
            # Already snake_case or convert from camelCase
            text = re.sub(r"(?<!^)(?=[A-Z])", "_", text).lower()
            return text

        elif target_case == "camelCase":
            # Convert snake_case to camelCase
            components = text.split("_")
            return components[0].lower() + "".join(x.title() for x in components[1:])

        elif target_case == "PascalCase":
            # Convert to PascalCase
            components = text.split("_")
            return "".join(x.title() for x in components)

        elif target_case == "kebab-case":
            # Convert to kebab-case
            text = re.sub(r"(?<!^)(?=[A-Z])", "-", text).lower()
            text = text.replace("_", "-")
            return text

        return text

    def get_perturbation_summary(self) -> Dict[str, Any]:
        """Get summary of applied perturbations."""
        summary = {
            "total_perturbations": len(self.applied_perturbations),
            "by_type": {},
        }

        for perturbation in self.applied_perturbations:
            ptype = perturbation["type"]
            if ptype not in summary["by_type"]:
                summary["by_type"][ptype] = 0
            summary["by_type"][ptype] += 1

        return summary

    def reset_tracking(self):
        """Reset perturbation tracking."""
        self.applied_perturbations = []


# ========== Environment Wrapper ==========


class PerturbedEnvironmentWrapper:
    """
    Wrapper for benchmark environments that applies structural perturbations.

    This wrapper intercepts environment interactions and applies perturbations
    to API calls, database queries, file operations, and data formats.
    """

    def __init__(self, env: Any, perturbator: StructuralPerturbator):
        """
        Initialize wrapper.

        Args:
            env: Original environment
            perturbator: Perturbator to apply
        """
        self.env = env
        self.perturbator = perturbator

    def execute_api_call(self, endpoint: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute API call with perturbations.

        Args:
            endpoint: API endpoint
            params: Request parameters

        Returns:
            Perturbed response
        """
        # Perturb request
        perturbed_endpoint = self.perturbator.perturb_api_endpoint(endpoint)
        perturbed_params = self.perturbator.perturb_api_params(params)

        # Execute (assuming env has execute method)
        if hasattr(self.env, "execute"):
            response = self.env.execute(perturbed_endpoint, perturbed_params)
        else:
            # Fallback: just perturb without executing
            response = {"status": "simulated", "data": params}

        # Perturb response
        perturbed_response = self.perturbator.perturb_api_response(response)

        return perturbed_response

    def read_file(self, path: str) -> Any:
        """
        Read file with path perturbation.

        Args:
            path: Original file path

        Returns:
            File contents
        """
        perturbed_path = self.perturbator.perturb_file_path(path)

        # Try to read from perturbed path (assuming env has read_file method)
        if hasattr(self.env, "read_file"):
            return self.env.read_file(perturbed_path)
        else:
            return None

    def query_database(self, query: str, data: Optional[Dict] = None) -> Any:
        """
        Execute database query with perturbations.

        Args:
            query: SQL query or operation
            data: Data to insert/update

        Returns:
            Query results
        """
        # Perturb table and column names in query
        # This is simplified - a real implementation would parse SQL properly
        perturbed_query = query

        # Perturb data if provided
        if data:
            data = self.perturbator.perturb_database_schema(data)
            data = self.perturbator.perturb_data(data)

        # Execute (assuming env has query method)
        if hasattr(self.env, "query"):
            return self.env.query(perturbed_query, data)
        else:
            return None


# ========== Factory Functions ==========


def create_perturbator(
    perturbation_type: str = "all",
    strength: str = "medium",
    custom_config: Optional[Dict] = None,
) -> StructuralPerturbator:
    """
    Factory function to create perturbator.

    Args:
        perturbation_type: Type of perturbations (api, database, file, data_format, all)
        strength: Perturbation strength (mild, medium, severe)
        custom_config: Custom configuration dictionary

    Returns:
        Configured StructuralPerturbator
    """
    if custom_config:
        config = PerturbationConfig.from_dict(custom_config)
    else:
        strength_enum = PerturbationStrength(strength)
        config = PerturbationConfig.get_preset(strength_enum)

    return StructuralPerturbator(perturbation_type, config)
