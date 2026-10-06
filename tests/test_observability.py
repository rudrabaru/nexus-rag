"""Structured logs with request ids, one line per event, no question text in logs, and scrubbed error reports."""
import io
import json
import logging

import pytest
import structlog

from src.observability import error_reporting
from src.observability.logger import PipelineLogger, loggable
from src.observability.logging_setup import configure_logging


@pytest.fixture
def logs():
    """Configures JSON logging onto a buffer, and puts the process's logging back afterwards."""
    root = logging.getLogger()
    saved = (root.handlers[:], root.level)
    buffer = io.StringIO()
    configure_logging("json", stream=buffer)

    def lines():
        records = [json.loads(line) for line in buffer.getvalue().splitlines() if line.startswith("{")]
        buffer.seek(0)
        buffer.truncate()
        return records

    yield lines
    root.handlers[:], root.level = saved[0], saved[1]
    structlog.reset_defaults()
    structlog.contextvars.clear_contextvars()


def test_a_standard_library_log_line_is_one_json_record(logs):
    logging.getLogger("some.library").warning("disk almost full")
    [record] = logs()
    assert record["event"] == "disk almost full" and record["level"] == "warning" and record["logger"] == "some.library"
    assert "timestamp" in record


def test_a_bound_request_id_is_on_every_line_including_other_loggers(logs):
    structlog.contextvars.bind_contextvars(request_id="abc12345")
    logging.getLogger("lib").info("one")
    structlog.get_logger("ours").info("two")
    assert [r["request_id"] for r in logs()] == ["abc12345", "abc12345"]


def test_configuring_twice_does_not_stack_handlers_and_print_every_line_twice():
    root = logging.getLogger()
    saved = (root.handlers[:], root.level)
    buffer = io.StringIO()
    try:
        configure_logging("json", stream=buffer)
        configure_logging("json", stream=buffer)
        logging.getLogger("lib").info("once")
        assert len(buffer.getvalue().splitlines()) == 1
    finally:
        root.handlers[:], root.level = saved
        structlog.reset_defaults()


def test_a_pipeline_event_is_printed_once_with_its_fields(logs):
    PipelineLogger("nexus_rag").log_event("retrieval_complete", tenant_id="t1", chunk_count=3)
    [record] = logs()
    assert record["event"] == "retrieval_complete" and record["tenant_id"] == "t1" and record["chunk_count"] == 3


def test_the_log_shows_how_long_a_question_was_not_what_it_said(logs):
    PipelineLogger("nexus_rag").log_event("query_started", query_text="what is our salary policy?", tenant_id="t1")
    [record] = logs()
    assert "query_text" not in record and record["query_chars"] == len("what is our salary policy?")
    assert "salary" not in json.dumps(record)


def test_the_persisted_event_keeps_the_text_for_debugging():
    shown = loggable({"query_text": "hello", "tenant_id": "t1"})
    assert shown == {"query_chars": 5, "tenant_id": "t1"}


def test_every_request_gets_an_id_and_one_access_line_without_its_query_string(client, tenant_key, logs):
    response = client.get("/v1/documents?token=s3cret", headers={"X-API-Key": tenant_key("t1")})
    records = [r for r in logs() if r["event"] == "request"]
    assert len(records) == 1
    access = records[0]
    assert access["path"] == "/v1/documents" and access["method"] == "GET" and access["status"] == response.status_code
    assert access["request_id"] == response.headers["X-Request-ID"]
    assert "s3cret" not in json.dumps(access)


def test_a_probe_is_not_logged_at_info_level(client, logs):
    client.get("/health")
    assert [r for r in logs() if r["event"] == "request"] == []


def test_the_request_id_does_not_leak_into_the_next_request(client, logs):
    client.get("/v1/documents")
    first = [r for r in logs() if r["event"] == "request"][0]["request_id"]
    client.get("/v1/documents")
    second = [r for r in logs() if r["event"] == "request"][0]["request_id"]
    assert first != second


def test_an_error_report_carries_no_request_data():
    event = {
        "exception": {"values": []},
        "request": {"url": "https://api/v1/chat", "method": "POST", "headers": {"x-api-key": "nx_secret"},
                    "data": {"query": "private question"}, "query_string": "a=b", "cookies": {"s": "1"}},
        "user": {"ip_address": "1.2.3.4"},
    }
    scrubbed = error_reporting.scrub_event(event)
    assert scrubbed["request"] == {"url": "https://api/v1/chat", "method": "POST"}
    assert "user" not in scrubbed


def test_error_reporting_is_off_without_a_dsn(monkeypatch):
    started = []
    monkeypatch.setattr(error_reporting.sentry_sdk, "init", lambda **kw: started.append(kw))
    assert error_reporting.init_error_reporting("") is False and started == []


def test_error_reporting_is_started_without_personal_data_when_a_dsn_is_set(monkeypatch):
    started = []
    monkeypatch.setattr(error_reporting.sentry_sdk, "init", lambda **kw: started.append(kw))
    assert error_reporting.init_error_reporting("https://key@sentry.invalid/1", "production") is True
    options = started[0]
    assert options["send_default_pii"] is False and options["traces_sample_rate"] == 0.0
    assert options["max_request_body_size"] == "never" and options["include_local_variables"] is False
    assert options["before_send"] is error_reporting.scrub_event
