from dataclasses import dataclass


@dataclass
class Environment:
    task_timeout: int = 1800
    max_concurrent: int = 10

    def validate(self) -> None:
        """Raise ValueError with all problems collected if config is invalid."""
        errors: list[str] = []
        if self.task_timeout <= 0:
            errors.append(f"task_timeout must be > 0, got {self.task_timeout}")
        if self.max_concurrent <= 0:
            errors.append(f"max_concurrent must be > 0, got {self.max_concurrent}")
        if errors:
            raise ValueError(
                "Invalid environment config:\n" + "\n".join(f"  - {e}" for e in errors)
            )
