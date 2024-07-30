from abc import ABC, abstractmethod
from pydantic import TypeAdapter, ValidationError


class BaseBenchmark(ABC):
    @abstractmethod
    def run(self, agent_function):
        pass

    @abstractmethod
    def test_run(self, agent_function):
        pass
    
    @property
    @abstractmethod
    def type_adapter(self) -> TypeAdapter:
        pass

    def validate_agent_output(self, output) -> bool:
        try:
            self.type_adapter.validate_python(output)
        except ValidationError as e:
            print("Agent output does not match expected format for output! Required output format" + str(e))