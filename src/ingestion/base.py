import abc
from src.crawling.metadata import AdapterResult


class IngestionAdapter(abc.ABC):
    """
    Base class for all ingestion adapters.
    Adapters convert specific input formats (URL, PDF, DOCX, etc.)
    into standard CrawledDocument objects.
    """

    @abc.abstractmethod
    async def ingest(self, source: str, **kwargs) -> AdapterResult:
        """
        Reads from a source (URL or file path) and returns an AdapterResult.
        """
        pass

