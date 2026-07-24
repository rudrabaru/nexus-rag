from abc import ABC, abstractmethod
from typing import Tuple, AsyncGenerator
from src.generating.models import GenerationConfig

class BaseProvider(ABC):
    def __init__(self, config: GenerationConfig):
        self.config = config
        self.client = self._init_client()

    @abstractmethod
    def _init_client(self):
        pass

    @abstractmethod
    def call(self, prompt: str, response_schema=None) -> Tuple[str, str, int, int]:
        pass

    @abstractmethod
    async def call_stream(self, prompt: str) -> AsyncGenerator[str, None]:
        pass
