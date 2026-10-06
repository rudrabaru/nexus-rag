"""Bodies are bounded before parsing or authentication."""
from src.api.app import MAX_JSON_BODY_BYTES


def test_an_oversized_json_body_is_refused_before_authentication(client):
    response = client.post("/v1/chat", content=b"x" * (MAX_JSON_BODY_BYTES + 1), headers={"content-type": "application/json"})
    assert response.status_code == 413


def test_a_declared_oversized_upload_is_refused_without_a_key(client):
    big = b"x" * (21 * 1024 * 1024 + 2 * 1024 * 1024)
    response = client.post("/v1/documents", files={"file": ("a.txt", big)})
    assert response.status_code == 413


def test_a_body_that_streams_past_the_limit_is_cut_off(client):
    def chunks():
        for _ in range(MAX_JSON_BODY_BYTES // 1024 + 2):
            yield b"x" * 1024

    response = client.post("/v1/chat", content=chunks(), headers={"content-type": "application/json"})
    assert response.status_code == 413


def test_a_normal_request_is_unaffected(client, tenant_key, app_state):
    response = client.post("/v1/chat", json={"query": "hi"}, headers={"X-API-Key": "nx_wrong"})
    assert response.status_code == 401
