"""The admin UI's API client: one auth header, one error type, and a stream reader that survives junk lines."""
import sys
from pathlib import Path

import pytest
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "admin_ui"))
from client import ApiError, NexusClient  # noqa: E402


class FakeResponse:
    def __init__(self, status=200, body=None, text="", lines=()):
        self.status_code, self._body, self.text, self._lines = status, body, text or str(body), lines

    def json(self):
        if self._body is None:
            raise ValueError("no json")
        return self._body

    def iter_lines(self):
        return iter(self._lines)


@pytest.fixture
def sent(monkeypatch):
    calls = []

    def fake(method, url, **kwargs):
        calls.append((method, url, kwargs))
        return fake.response

    fake.response = FakeResponse(body={"job_id": "j1"})
    monkeypatch.setattr(requests, "request", fake)
    return calls, fake


def test_the_key_travels_in_one_header_and_only_when_there_is_one(sent):
    calls, _ = sent
    NexusClient("http://api/", "secret").documents()
    NexusClient("http://api").documents()
    assert calls[0][1] == "http://api/v1/documents"
    assert calls[0][2]["headers"] == {"X-API-Key": "secret"}
    assert calls[1][2]["headers"] == {}


def test_an_error_response_carries_the_apis_message(sent):
    _, fake = sent
    fake.response = FakeResponse(status=429, body={"code": "rate_limited", "message": "Slow down."})
    with pytest.raises(ApiError) as error:
        NexusClient("http://api", "k").usage()
    assert (error.value.status, error.value.message) == (429, "Slow down.")
    assert str(error.value) == "Error 429: Slow down."


def test_a_body_that_is_not_the_standard_error_shape_still_gives_a_message(sent):
    _, fake = sent
    fake.response = FakeResponse(status=502, text="<html>bad gateway</html>")
    with pytest.raises(ApiError) as error:
        NexusClient("http://api", "k").usage()
    assert "bad gateway" in error.value.message


def test_an_unreachable_api_is_an_error_with_status_zero(monkeypatch):
    def refuse(*args, **kwargs):
        raise requests.ConnectionError("refused")

    monkeypatch.setattr(requests, "request", refuse)
    with pytest.raises(ApiError) as error:
        NexusClient("http://api", "k").usage()
    assert error.value.status == 0 and "http://api" in error.value.message


def test_stream_events_skip_lines_that_are_not_events(sent):
    _, fake = sent
    fake.response = FakeResponse(lines=[
        b": keep-alive", b'data: {"type": "token", "content": "Hi"}', b"data: not json", b"",
        b'data: {"type": "done"}',
    ])
    events = list(NexusClient("http://api", "k").chat_events({"query": "q"}))
    assert events == [{"type": "token", "content": "Hi"}, {"type": "done"}]


def test_an_upload_sends_the_file_and_returns_the_job_id(sent):
    calls, _ = sent
    job_id = NexusClient("http://api", "k").ingest_file("a.md", b"# A", "text/markdown")
    assert job_id == "j1"
    assert calls[0][2]["files"] == {"file": ("a.md", b"# A", "text/markdown")}
