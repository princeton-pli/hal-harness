from abc import ABC, abstractmethod
from pydantic import TypeAdapter, ValidationError
from ..utils.weave_utils import get_total_cost

class BaseBenchmark(ABC):

    @abstractmethod
    def run(self, agent_function, run_id: str):
        pass

    @abstractmethod
    def test_run(self, agent_function, weave_client):
        pass
    
    @property
    @abstractmethod
    def type_adapter(self) -> TypeAdapter:
        pass
    
    @abstractmethod
    def process_and_upload_results(self, results, agent_name):
        pass

    def validate_agent_output(self, output) -> bool:
        try:
            self.type_adapter.validate_python(output)
        except ValidationError as e:
            print("Agent output does not match expected format for output! Required output format" + str(e))

    def validate_cost_logging(self, weave_client):
        assert get_total_cost(weave_client) > 0, "Test run did not incur any cost"
