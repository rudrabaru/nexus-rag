from src.config import Settings, get_settings


def make_settings(monkeypatch, **env):
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    get_settings.cache_clear()
    return get_settings()


def test_complete_environment_has_no_missing_settings():
    assert get_settings().missing_required() == []


def test_missing_settings_are_all_reported_at_once(monkeypatch):
    for key in ("QDRANT_URL", "QDRANT_API_KEY", "JINA_API_KEY", "GEMINI_API_KEY"):
        monkeypatch.delenv(key)
    get_settings.cache_clear()

    missing = get_settings().missing_required()
    assert {"QDRANT_URL", "QDRANT_API_KEY", "JINA_API_KEY", "GEMINI_API_KEY"} <= set(missing)


def test_provider_key_requirement_follows_llm_provider(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY")
    settings = make_settings(monkeypatch, LLM_PROVIDER="groq")
    assert "GROQ_API_KEY" in settings.missing_required()
    assert "GEMINI_API_KEY" not in settings.missing_required()


def test_short_signing_secret_is_rejected(monkeypatch):
    settings = make_settings(monkeypatch, API_KEY_SIGNING_SECRET="short")
    assert any(p.startswith("API_KEY_SIGNING_SECRET") for p in settings.missing_required())


def test_legacy_rag_api_key_is_a_fallback_for_both_secrets(monkeypatch):
    monkeypatch.delenv("ADMIN_API_KEY")
    monkeypatch.delenv("API_KEY_SIGNING_SECRET")
    settings = make_settings(monkeypatch, RAG_API_KEY="legacy-secret-0123456789")

    assert settings.effective_admin_key == "legacy-secret-0123456789"
    assert settings.effective_signing_secret == "legacy-secret-0123456789"
    assert settings.missing_required() == []


def test_new_secrets_take_precedence_over_legacy(monkeypatch):
    settings = make_settings(monkeypatch, RAG_API_KEY="legacy-secret-0123456789")
    assert settings.effective_admin_key != "legacy-secret-0123456789"


def test_cors_origins_default_to_none_and_parse_lists(monkeypatch):
    assert get_settings().cors_origins == []
    settings = make_settings(monkeypatch, ALLOWED_ORIGINS=" https://a.example , https://b.example ,")
    assert settings.cors_origins == ["https://a.example", "https://b.example"]


def test_unsafe_defaults_are_off():
    settings = Settings()
    assert settings.trust_proxies is False
    assert settings.cors_origins == []
