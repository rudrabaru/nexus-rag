"""
Process start-up shared by every entry point (API, workers, command-line tools).

Importing a module never starts anything: no environment is loaded and no database is touched.
An entry point calls bootstrap() (or, for the API, load_environment() and configure_logging())
once, explicitly, as its first act. That keeps modules importable in tests and tools without side
effects, and gives configuration problems one place to be reported.
"""
import asyncio
import logging
import sys
from pathlib import Path

from dotenv import load_dotenv

from src.config import Settings, get_settings
from src.config_checks import Role, config_problems
from src.registry.engine import get_sync_engine
from src.registry.schema_version import assert_schema_current

ENV_FILE = Path(__file__).resolve().parents[1] / ".env"


class ConfigurationError(RuntimeError):
    pass


def load_environment() -> None:
    """Loads .env into the process environment. Variables already set in the real environment win over the file."""
    load_dotenv(dotenv_path=ENV_FILE, override=False)
    get_settings.cache_clear()


def configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")


def check_configuration(role: Role) -> Settings:
    settings = get_settings()
    problems = config_problems(settings, role)
    if problems:
        raise ConfigurationError(f"Missing or invalid configuration: {', '.join(problems)}. Startup aborted.")
    return settings


def bootstrap(role: Role, async_postgres_driver: bool = False) -> Settings:
    """
    Environment, logging and a configuration check; for a worker or tool also the schema check.
    The API does its schema check in its lifespan, off the event loop.

    async_postgres_driver: psycopg's async pool refuses Windows' default ProactorEventLoop, so a
    process that runs it (a worker) selects the selector loop. Local Windows development only;
    deployments are Linux.
    """
    if async_postgres_driver and sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    load_environment()
    configure_logging()
    settings = check_configuration(role)
    if role != "api":
        assert_schema_current(get_sync_engine())
    return settings
