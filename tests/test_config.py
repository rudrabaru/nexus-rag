from src.config import Settings, get_settings


def make_settings(monkeypatch, **env):
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    get_settings.cache_clear()
    return get_settings()


def test_complete_environment_has_no_missing_settings():
    assert get_settings().missing_required() == []


def test_missing_settings_are_all_reported_at_once(monkeypatch):
    for key in ("DATABASE_URL", "VOYAGE_API_KEY", "GEMINI_API_KEY"):
        monkeypatch.delenv(key)
    get_settings.cache_clear()

    missing = " ".join(get_settings().missing_required())
    assert all(name in missing for name in ("DATABASE_URL", "VOYAGE_API_KEY", "GEMINI_API_KEY"))


def test_embedding_key_requirement_follows_the_embedding_provider(monkeypatch):
    """Jina is only needed to query the legacy index; a local Ollama index needs no key at all."""
    assert not any("JINA" in p for p in get_settings().missing_required())

    monkeypatch.delenv("VOYAGE_API_KEY")
    assert any("JINA_API_KEY" in p for p in make_settings(monkeypatch, EMBEDDING_PROVIDER="jina").missing_required())
    assert make_settings(monkeypatch, EMBEDDING_PROVIDER="ollama").missing_required() == []
    assert any("EMBEDDING_PROVIDER" in p for p in make_settings(monkeypatch, EMBEDDING_PROVIDER="qdrant").missing_required())


def test_fetch_domain_lists_are_parsed_and_lowercased(monkeypatch):
    settings = make_settings(monkeypatch, FETCH_ALLOWED_DOMAINS=" Docs.Python.org, ", FETCH_DENIED_DOMAINS="")
    assert settings.allowed_fetch_domains == ["docs.python.org"]
    assert settings.denied_fetch_domains == []
    assert "facebook.com" in Settings.model_fields["fetch_denied_domains"].default  # a denylist is on by default



def test_provider_key_requirement_follows_llm_provider(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY")
    settings = make_settings(monkeypatch, LLM_PROVIDER="groq")
    assert "GROQ_API_KEY" in settings.missing_required()
    assert "GEMINI_API_KEY" not in settings.missing_required()


def test_legacy_rag_api_key_is_a_fallback_for_the_admin_key(monkeypatch):
    monkeypatch.delenv("ADMIN_API_KEY")
    settings = make_settings(monkeypatch, RAG_API_KEY="legacy-secret-0123456789-abcdefghijklmnop")

    assert settings.effective_admin_key == "legacy-secret-0123456789-abcdefghijklmnop"
    assert settings.missing_required() == []


def test_admin_key_takes_precedence_over_legacy(monkeypatch):
    settings = make_settings(monkeypatch, RAG_API_KEY="legacy-secret-0123456789-abcdefghijklmnop")
    assert settings.effective_admin_key != "legacy-secret-0123456789-abcdefghijklmnop"


def test_cors_origins_default_to_none_and_parse_lists(monkeypatch):
    assert get_settings().cors_origins == []
    settings = make_settings(monkeypatch, ALLOWED_ORIGINS=" https://a.example , https://b.example ,")
    assert settings.cors_origins == ["https://a.example", "https://b.example"]


def test_unsafe_defaults_are_off():
    settings = Settings()
    assert settings.trusted_proxy_hops == 0
    assert settings.cors_origins == []
