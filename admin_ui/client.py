"""
The only place the admin UI talks to the Nexus API: one auth header, one error type, one place
that knows the routes. The tabs call these methods and never build a request themselves.
"""
import json
from typing import Dict, Iterator, List, Optional

import requests

API_KEY_HEADER = "X-API-Key"
REQUEST_TIMEOUT_SECONDS = 30
CHAT_TIMEOUT_SECONDS = 120  # an answer includes retrieval, optional reranking and the model call


class ApiError(Exception):
    """A request that failed. `status` is 0 when the API could not be reached at all."""

    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status
        self.message = message

    def __str__(self) -> str:
        return f"Error {self.status}: {self.message}" if self.status else self.message


def _message(response: requests.Response) -> str:
    """The API's error message, or the raw text when the body is not the standard error shape."""
    try:
        return response.json().get("message", response.text[:200])
    except (ValueError, AttributeError):
        return response.text[:200]


class NexusClient:
    def __init__(self, base_url: str, api_key: str = ""):
        self.base_url = base_url.rstrip("/")
        self.headers: Dict[str, str] = {API_KEY_HEADER: api_key} if api_key else {}

    def _send(self, method: str, path: str, timeout: float = REQUEST_TIMEOUT_SECONDS, **kwargs) -> requests.Response:
        try:
            response = requests.request(method, f"{self.base_url}{path}", headers=self.headers, timeout=timeout, **kwargs)
        except requests.RequestException as e:
            raise ApiError(0, f"Cannot reach the API at {self.base_url}: {e}")
        if response.status_code >= 400:
            raise ApiError(response.status_code, _message(response))
        return response

    def chat(self, payload: dict) -> dict:
        return self._send("POST", "/v1/chat", CHAT_TIMEOUT_SECONDS, json=payload).json()

    def chat_events(self, payload: dict) -> Iterator[dict]:
        """The streamed answer's events: token, sources, done, error, and optionally faithfulness."""
        response = self._send("POST", "/v1/chat/stream", CHAT_TIMEOUT_SECONDS, json=payload, stream=True)
        for line in response.iter_lines():
            if not line.startswith(b"data: "):
                continue
            try:
                yield json.loads(line[6:])
            except json.JSONDecodeError:
                continue

    def compare_retrieval(self, query: str, top_k: int) -> dict:
        return self._send("POST", "/v1/retrieval/compare", json={"query": query, "top_k": top_k}).json()

    def usage(self) -> dict:
        return self._send("GET", "/v1/usage").json()

    def documents(self) -> List[dict]:
        return self._send("GET", "/v1/documents").json()

    def ingest_url(self, url: str) -> str:
        return self._send("POST", "/v1/documents", data={"url": url}).json().get("job_id")

    def ingest_file(self, name: str, content: bytes, mime_type: Optional[str]) -> str:
        files = {"file": (name, content, mime_type)}
        return self._send("POST", "/v1/documents", files=files).json().get("job_id")

    def delete_document(self, doc_id: str) -> None:
        self._send("DELETE", f"/v1/documents/{doc_id}")

    def job(self, job_id: str) -> dict:
        return self._send("GET", f"/v1/jobs/{job_id}").json()
