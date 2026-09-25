import pytest

from src.registry.engine import async_connect_args, async_url, is_pooler_url, sync_url

NEON = "postgresql://user:pw@ep-cool-name-123.ap-southeast-1.aws.neon.tech/neondb?sslmode=require&channel_binding=require"


def test_sync_url_keeps_libpq_parameters_for_psycopg():
    url = sync_url(NEON)
    assert url.drivername == "postgresql+psycopg"
    assert url.query["sslmode"] == "require"
    assert url.query["channel_binding"] == "require"


def test_async_url_translates_sslmode_for_asyncpg():
    url = async_url(NEON)
    assert url.drivername == "postgresql+asyncpg"
    assert url.query == {"ssl": "require"}


def test_async_url_without_sslmode_adds_no_tls_flag():
    assert "ssl" not in async_url("postgresql://u:p@localhost/db").query


def test_postgres_scheme_alias_is_accepted():
    assert sync_url("postgres://u:p@localhost/db").drivername == "postgresql+psycopg"


def test_non_postgres_urls_are_rejected():
    with pytest.raises(ValueError):
        sync_url("mysql://u:p@localhost/db")


def test_session_settings_travel_in_the_options_parameter():
    """Neon's proxy drops individual startup parameters but forwards libpq `options`."""
    args = async_connect_args(NEON, {"search_path": "s1,public"})
    assert set(args["server_settings"]) == {"options"}
    options = args["server_settings"]["options"].split()
    assert "-chnsw.iterative_scan=relaxed_order" in options
    assert "-chnsw.ef_search=100" in options
    assert "-csearch_path=s1,public" in options


def test_pooler_endpoint_gets_no_startup_settings():
    args = async_connect_args(NEON.replace("ep-cool-name-123", "ep-cool-name-123-pooler"))
    assert "server_settings" not in args
    assert args["statement_cache_size"] == 0


def test_pooler_endpoint_is_detected():
    assert is_pooler_url(NEON.replace("ep-cool-name-123", "ep-cool-name-123-pooler"))
    assert not is_pooler_url(NEON)
