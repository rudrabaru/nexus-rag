import ipaddress
import socket
from urllib.parse import urlparse

ALLOWED_SCHEMES = {"http", "https"}


class UnsafeUrlError(ValueError):
    """Raised when a user-supplied URL must not be fetched."""


def validate_public_url(url: str) -> None:
    """
    Accepts only http(s) URLs whose host resolves exclusively to public addresses.

    Why: the ingest `url` field was previously passed to the dispatcher unchecked, so a
    filesystem path ending in .txt/.md/.pdf/.docx was read as a local file, and internal
    hosts could be requested. This check is structural (scheme and resolved address), not a
    list of sites.

    Limits: it validates the address at check time only. Redirects and DNS rebinding are
    not covered here; they are addressed by fetching through a hosted reader API instead of
    from this process.
    """
    parsed = urlparse(url.strip())

    if parsed.scheme.lower() not in ALLOWED_SCHEMES:
        raise UnsafeUrlError("Only http and https URLs are accepted.")
    if not parsed.hostname:
        raise UnsafeUrlError("URL has no host.")
    if parsed.username or parsed.password:
        raise UnsafeUrlError("URLs with embedded credentials are not accepted.")

    port = parsed.port or (443 if parsed.scheme.lower() == "https" else 80)
    try:
        addresses = {info[4][0] for info in socket.getaddrinfo(parsed.hostname, port, type=socket.SOCK_STREAM)}
    except socket.gaierror:
        raise UnsafeUrlError("URL host could not be resolved.")

    for address in addresses:
        ip = ipaddress.ip_address(address.split("%")[0])
        if ip.version == 6 and ip.ipv4_mapped:
            ip = ip.ipv4_mapped
        if not ip.is_global:
            raise UnsafeUrlError("URL host resolves to a non-public address.")
