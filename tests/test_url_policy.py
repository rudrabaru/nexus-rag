"""The URL policy and its edge cases."""
import socket
import pytest
from src.crawling.url_policy import UnsafeUrlError, validate_public_url
from tests.support.ingest_security import resolve_to


@pytest.mark.parametrize(
    "url",
    [
        "/etc/passwd.txt",
        "logs.txt",
        "C:\\Users\\someone\\notes.md",
        "file:///etc/hosts.txt",
        "ftp://example.com/a.pdf",
        "gopher://example.com/",
        "//example.com/a.md",
        "",
    ],
)
def test_non_http_sources_are_rejected_before_any_lookup(url, monkeypatch):
    monkeypatch.setattr(
        "src.crawling.url_policy.socket.getaddrinfo",
        lambda *a, **k: pytest.fail("DNS must not be consulted for a non-http URL"),
    )
    with pytest.raises(UnsafeUrlError):
        validate_public_url(url)


@pytest.mark.parametrize(
    "address",
    ["10.0.0.5", "127.0.0.1", "169.254.169.254", "192.168.1.10", "172.16.0.1", "100.64.0.1", "0.0.0.0", "::1", "fd00::1", "::ffff:127.0.0.1"],
)
def test_hosts_resolving_to_non_public_addresses_are_rejected(address, monkeypatch):
    resolve_to(monkeypatch, address)
    with pytest.raises(UnsafeUrlError):
        validate_public_url("https://internal.example/page")


def test_one_private_address_among_public_ones_is_enough_to_reject(monkeypatch):
    resolve_to(monkeypatch, "93.184.216.34", "10.0.0.5")
    with pytest.raises(UnsafeUrlError):
        validate_public_url("https://mixed.example/")


def test_embedded_credentials_are_rejected(monkeypatch):
    resolve_to(monkeypatch, "93.184.216.34")
    with pytest.raises(UnsafeUrlError):
        validate_public_url("https://user:secret@example.com/")


def test_unresolvable_hosts_are_rejected(monkeypatch):
    def fail(*args, **kwargs):
        raise socket.gaierror("no such host")

    monkeypatch.setattr("src.crawling.url_policy.socket.getaddrinfo", fail)
    with pytest.raises(UnsafeUrlError):
        validate_public_url("https://does-not-exist.invalid/")


def test_malformed_port_is_rejected_with_validation_error():
    with pytest.raises(UnsafeUrlError):
        validate_public_url("https://example.com:99999/")


@pytest.mark.parametrize("url", ["https://example.com/docs", "http://example.com/a.pdf", "HTTPS://Example.com/x"])
def test_public_http_urls_are_accepted(url, monkeypatch):
    resolve_to(monkeypatch, "93.184.216.34")
    validate_public_url(url)


@pytest.mark.parametrize("url", ["https://[x", "https://[::1", "https://exa mple.com:abc/"])
def test_malformed_urls_are_unsafe_url_errors_not_crashes(url):
    with pytest.raises(UnsafeUrlError):
        validate_public_url(url)


@pytest.mark.parametrize("address", ["64:ff9b::7f00:1", "64:ff9b::a00:5", "2002:7f00:1::1"])
def test_private_addresses_wrapped_in_nat64_or_6to4_are_rejected(address, monkeypatch):
    resolve_to(monkeypatch, address)
    with pytest.raises(UnsafeUrlError):
        validate_public_url("https://wrapped.example/")


def test_a_host_that_does_not_resolve_in_time_is_rejected(monkeypatch):
    import time

    from src.crawling import url_policy

    monkeypatch.setattr(url_policy, "DNS_TIMEOUT_SECONDS", 0.05)
    monkeypatch.setattr("src.crawling.url_policy.socket.getaddrinfo", lambda *a, **k: time.sleep(0.5))
    with pytest.raises(UnsafeUrlError, match="in time"):
        validate_public_url("https://slow.example/")


def test_redaction_keeps_the_address_and_drops_the_secret():
    from src.crawling.url_policy import redact_url

    assert redact_url("https://a.example/docs/page?token=SECRET#frag") == "https://a.example/docs/page"
