"""Which client a request is counted against."""
from starlette.requests import Request
from src.api.rate_limit import client_ip, rate_limit_key


def make_request(headers=None, client=("1.2.3.4", 5000)):
    scope = {
        "type": "http",
        "headers": [(k.encode(), v.encode()) for k, v in (headers or {}).items()],
        "client": client,
    }
    return Request(scope)


def hops(monkeypatch, count):
    from src.config import get_settings

    monkeypatch.setenv("TRUSTED_PROXY_HOPS", str(count))
    get_settings.cache_clear()


def test_forwarded_headers_are_ignored_by_default():
    assert client_ip(make_request({"x-forwarded-for": "9.9.9.9"})) == "1.2.3.4"


def test_with_one_trusted_proxy_the_last_entry_is_the_client_and_earlier_ones_are_spoofable(monkeypatch):
    hops(monkeypatch, 1)
    # The client sent "6.6.6.6" itself; the proxy appended the address it actually saw.
    assert client_ip(make_request({"x-forwarded-for": "6.6.6.6, 9.9.9.9"})) == "9.9.9.9"


def test_with_two_trusted_proxies_the_client_is_second_from_the_right(monkeypatch):
    hops(monkeypatch, 2)
    assert client_ip(make_request({"x-forwarded-for": "6.6.6.6, 9.9.9.9, 10.0.0.1"})) == "9.9.9.9"


def test_a_header_shorter_than_the_trusted_chain_or_not_an_address_falls_back_to_the_peer(monkeypatch):
    hops(monkeypatch, 2)
    assert client_ip(make_request({"x-forwarded-for": "9.9.9.9"})) == "1.2.3.4"
    hops(monkeypatch, 1)
    assert client_ip(make_request({"x-forwarded-for": "not-an-ip"})) == "1.2.3.4"


def test_authenticated_callers_are_limited_per_tenant_not_per_ip():
    request = make_request({"x-forwarded-for": "9.9.9.9"})
    request.state.tenant_id = "tenant-1"
    assert rate_limit_key(request) == "tenant:tenant-1"


def test_anonymous_callers_are_limited_per_ip():
    assert rate_limit_key(make_request()) == "ip:1.2.3.4"
