import pytest

from src.config import Settings, get_settings
from src.config_checks import MIN_ADMIN_KEY_LENGTH, config_problems


def test_the_cloudflare_provider_needs_its_account_and_token(monkeypatch):
    from src.config_checks import config_problems

    settings = make_settings(monkeypatch, EMBEDDING_PROVIDER="cloudflare")
    problems = config_problems(settings, "api")
    assert any("CLOUDFLARE_ACCOUNT_ID" in p for p in problems) and any("CLOUDFLARE_API_TOKEN" in p for p in problems)
    settings = make_settings(monkeypatch, CLOUDFLARE_ACCOUNT_ID="a", CLOUDFLARE_API_TOKEN="t")
    assert not any("CLOUDFLARE" in p for p in config_problems(settings, "api"))


def make_settings(monkeypatch, **env):
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    get_settings.cache_clear()
    return get_settings()


def problems(role="api"):
    return config_problems(get_settings(), role)


def test_complete_environment_has_no_problems_for_every_role():
    for role in ("api", "worker", "cli"):
        assert problems(role) == []


def test_all_problems_are_reported_at_once(monkeypatch):
    for key in ("DATABASE_URL", "VOYAGE_API_KEY", "GEMINI_API_KEY"):
        monkeypatch.delenv(key)
    get_settings.cache_clear()

    reported = " ".join(problems())
    assert all(name in reported for name in ("DATABASE_URL", "VOYAGE_API_KEY", "GEMINI_API_KEY"))


def test_each_role_requires_only_what_it_uses(monkeypatch):
    for key in ("ADMIN_API_KEY", "GEMINI_API_KEY", "VOYAGE_API_KEY"):
        monkeypatch.delenv(key)
    get_settings.cache_clear()

    assert "ADMIN_API_KEY" in " ".join(problems("api"))
    assert problems("cli") == []  # a command-line tool needs the database; commands ask for the rest themselves
    assert "VOYAGE_API_KEY" in " ".join(problems("worker"))
    assert "ADMIN_API_KEY" not in " ".join(problems("worker"))


def test_embedding_key_requirement_follows_the_embedding_provider(monkeypatch):
    monkeypatch.delenv("VOYAGE_API_KEY")
    assert config_problems(make_settings(monkeypatch, EMBEDDING_PROVIDER="ollama"), "api") == []
    assert any("EMBEDDING_PROVIDER" in p for p in config_problems(make_settings(monkeypatch, EMBEDDING_PROVIDER="qdrant"), "api"))
    assert any("EMBEDDING_PROVIDER" in p for p in config_problems(make_settings(monkeypatch, EMBEDDING_PROVIDER="jina"), "api"))


def test_provider_key_requirement_follows_the_chat_model(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY")
    reported = config_problems(make_settings(monkeypatch, LLM_CHAT="groq/openai/gpt-oss-20b"), "api")
    assert "GROQ_API_KEY" in reported and "GEMINI_API_KEY" not in reported


def test_the_admin_key_must_be_long_enough_to_resist_guessing(monkeypatch):
    short = "x" * (MIN_ADMIN_KEY_LENGTH - 1)
    assert any("ADMIN_API_KEY" in p for p in config_problems(make_settings(monkeypatch, ADMIN_API_KEY=short), "api"))
    assert config_problems(make_settings(monkeypatch, ADMIN_API_KEY="x" * MIN_ADMIN_KEY_LENGTH), "api") == []


@pytest.mark.parametrize("name", ["RAG_API_KEY", "ENABLE_RERANKER", "TRUST_PROXIES", "MISTRAL_API_KEY", "LLM_PROVIDER", "LLM_MODEL_NAME"])
def test_a_removed_variable_still_set_stops_the_process_instead_of_being_ignored(monkeypatch, name):
    monkeypatch.setenv(name, "false")
    reported = " ".join(problems())
    assert name in reported and "no longer supported" in reported


@pytest.mark.parametrize("url, ok", [
    ("postgresql://u:p@ep-x.neon.tech/db?sslmode=require", True),
    ("postgresql://u:p@ep-x.neon.tech/db?sslmode=verify-full", True),
    ("postgresql://u:p@ep-x.neon.tech/db", False),
    ("postgresql://u:p@ep-x.neon.tech/db?sslmode=disable", False),
    ("postgresql://u:p@localhost:5432/db", True),
    ("mysql://u:p@localhost/db", False),
])
def test_a_remote_database_must_use_an_encrypted_connection(monkeypatch, url, ok):
    reported = config_problems(make_settings(monkeypatch, DATABASE_URL=url), "api")
    assert (not any("DATABASE_URL" in p for p in reported)) is ok


def test_secrets_do_not_appear_when_settings_are_printed():
    text = repr(get_settings())
    assert "gemini-key" not in text and "voyage-key" not in text and "test:test" not in text


def test_fetch_domain_lists_are_parsed_and_lowercased(monkeypatch):
    settings = make_settings(monkeypatch, FETCH_ALLOWED_DOMAINS=" Docs.Python.org, ", FETCH_DENIED_DOMAINS="")
    assert settings.allowed_fetch_domains == ["docs.python.org"]
    assert settings.denied_fetch_domains == []
    assert "facebook.com" in Settings.model_fields["fetch_denied_domains"].default  # a denylist is on by default


def test_cors_origins_default_to_none_and_parse_lists(monkeypatch):
    assert get_settings().cors_origins == []
    settings = make_settings(monkeypatch, ALLOWED_ORIGINS=" https://a.example , https://b.example ,")
    assert settings.cors_origins == ["https://a.example", "https://b.example"]


@pytest.mark.parametrize("name, value", [("QUERY_CONCURRENCY", "0"), ("DB_POOL_SIZE", "500"), ("TRUSTED_PROXY_HOPS", "-1")])
def test_out_of_range_numbers_are_rejected_at_load(monkeypatch, name, value):
    monkeypatch.setenv(name, value)
    get_settings.cache_clear()
    with pytest.raises(Exception):
        get_settings()


def test_unsafe_defaults_are_off():
    settings = Settings(_env_file=None)
    assert settings.trusted_proxy_hops == 0
    assert settings.cors_origins == []
