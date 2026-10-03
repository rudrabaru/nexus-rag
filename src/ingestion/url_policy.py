import ipaddress
import socket
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as LookupTimeout
from urllib.parse import urlparse

ALLOWED_SCHEMES = {"http", "https"}
DNS_TIMEOUT_SECONDS = 5.0  # a name that does not resolve in this long is treated as unresolvable

# getaddrinfo cannot be given a timeout, so it runs on a small pool and the caller stops waiting.
_resolver = ThreadPoolExecutor(max_workers=4, thread_name_prefix="dns")
_NAT64 = ipaddress.ip_network("64:ff9b::/96")


class UnsafeUrlError(ValueError):
    """Raised when a user-supplied URL must not be fetched."""


def redact_url(url: str) -> str:
    """The URL without its query string and fragment, which can carry tokens, for logs and audit rows."""
    parsed = urlparse(url.strip())
    return f"{parsed.scheme}://{parsed.hostname or ''}{parsed.path}"


def _addresses(hostname: str, port: int) -> set:
    future = _resolver.submit(lambda: socket.getaddrinfo(hostname, port, type=socket.SOCK_STREAM))
    try:
        return {info[4][0] for info in future.result(timeout=DNS_TIMEOUT_SECONDS)}
    except LookupTimeout:
        raise UnsafeUrlError("URL host did not resolve in time.")
    except socket.gaierror:
        raise UnsafeUrlError("URL host could not be resolved.")


def _public(address: str) -> bool:
    ip = ipaddress.ip_address(address.split("%")[0])
    if ip.version == 6:
        embedded = ip.ipv4_mapped
        if embedded is None and ip in _NAT64:
            embedded = ipaddress.IPv4Address(int(ip) & 0xFFFFFFFF)  # a v4 address wrapped by a NAT64 gateway
        if embedded is None and ip.sixtofour is not None:
            embedded = ip.sixtofour
        if embedded is not None:
            ip = embedded
    return ip.is_global


def validate_public_url(url: str) -> None:
    """
    Accepts only http(s) URLs whose host resolves exclusively to public addresses.

    This check is structural (scheme and resolved address), not a list of sites. It validates the
    address at check time only: redirects and DNS rebinding are not covered here, because pages
    are fetched by a hosted reader API from its own network, not by this process.
    """
    try:
        parsed = urlparse(url.strip())
        port = parsed.port or (443 if parsed.scheme.lower() == "https" else 80)
    except ValueError:
        raise UnsafeUrlError("URL is malformed or has an invalid port.")

    if parsed.scheme.lower() not in ALLOWED_SCHEMES:
        raise UnsafeUrlError("Only http and https URLs are accepted.")
    if not parsed.hostname:
        raise UnsafeUrlError("URL has no host.")
    if parsed.username or parsed.password:
        raise UnsafeUrlError("URLs with embedded credentials are not accepted.")

    for address in _addresses(parsed.hostname, port):
        if not _public(address):
            raise UnsafeUrlError("URL host resolves to a non-public address.")
