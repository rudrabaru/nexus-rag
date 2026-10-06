"""The committed openapi.json is the API's contract; the app must match it."""
import json

from scripts.export_openapi import OPENAPI_PATH, render


def test_the_committed_openapi_document_matches_the_app():
    committed = json.loads(OPENAPI_PATH.read_text(encoding="utf-8"))
    assert committed == json.loads(render()), "The API changed: run `python -m scripts.export_openapi` and commit openapi.json."


def test_every_route_has_a_stable_operation_id_and_a_documented_error_shape():
    spec = json.loads(render())
    operations = [(path, method, op) for path, item in spec["paths"].items() for method, op in item.items()]
    ids = [op["operationId"] for _, _, op in operations]

    assert all(not i.startswith(tuple("0123456789")) and "_v1_" not in i for i in ids), ids  # not the auto-generated path-derived names
    assert len(ids) == len(set(ids))
    assert "ErrorBody" in spec["components"]["schemas"]
    for path, method, op in operations:
        if path.startswith("/v1"):
            assert "401" in op["responses"] or path.startswith("/v1/admin"), (path, method)


def test_no_v1_route_returns_an_untyped_body():
    spec = json.loads(render())
    for path, item in spec["paths"].items():
        for method, op in item.items():
            if path == "/v1/chat/stream":
                continue  # server-sent events, documented in its description and x-events
            content = op["responses"].get("200", op["responses"].get("202", {})).get("content", {})
            assert content.get("application/json", {}).get("schema", {}).get("$ref") or \
                content.get("application/json", {}).get("schema", {}).get("items"), (path, method)
