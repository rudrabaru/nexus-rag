"""
Writes the API's OpenAPI document to openapi.json, the contract a client (the Next.js app's typed
SDK) is generated from. A test fails when the app and the committed file disagree, so a change to a
request or response shape is always a visible change to this file.

    python -m scripts.export_openapi
"""
import json
from pathlib import Path

from src.api.app import create_app

OPENAPI_PATH = Path(__file__).resolve().parents[1] / "openapi.json"


def render() -> str:
    return json.dumps(create_app().openapi(), indent=2, sort_keys=True, ensure_ascii=False) + "\n"


if __name__ == "__main__":
    OPENAPI_PATH.write_text(render(), encoding="utf-8")
    print(f"wrote {OPENAPI_PATH}")
