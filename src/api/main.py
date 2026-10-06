"""ASGI entry point: `uvicorn src.api.main:app`. The only module here that acts on import."""
from src.api.app import create_app
from src.runtime import configure_logging, load_environment

load_environment()
configure_logging()
app = create_app()
