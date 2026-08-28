#!/usr/bin/env python3
"""Small destination-allowlisting HTTP CONNECT proxy for managed Pi containers."""
from __future__ import annotations

import ipaddress
import io
import os
import re
import select
import socket
import socketserver
from typing import BinaryIO

MAX_HEADERS = 64 * 1024
MAX_CLIENT_HELLO = 128 * 1024
ALLOW_PRIVATE = os.environ.get("HERMES_PI_PROXY_TEST_ALLOW_PRIVATE") == "1"
_HEADER_NAME = re.compile(rb"^[!#$%&'*+.^_`|~0-9A-Za-z-]+$")
_HOP_BY_HOP_HEADERS = frozenset({
    b"connection",
    b"keep-alive",
    b"proxy-authenticate",
    b"proxy-authorization",
    b"proxy-connection",
    b"te",
    b"trailer",
    b"transfer-encoding",
    b"upgrade",
})


def _allowed_destinations() -> set[tuple[str, int]]:
    result: set[tuple[str, int]] = set()
    for raw in os.environ.get("HERMES_PI_PROXY_ALLOWLIST", "").split(","):
        value = raw.strip().lower()
        if not value:
            continue
        if value.count(":") == 1:
            host, raw_port = value.rsplit(":", 1)
            port = int(raw_port)
        else:
            host, port = value, 443
        result.add((host.rstrip("."), port))
    return result


ALLOWLIST = _allowed_destinations()


def _read_exact(stream: BinaryIO | io.BufferedIOBase, length: int) -> bytes:
    chunks = bytearray()
    while len(chunks) < length:
        chunk = stream.read(length - len(chunks))
        if not chunk:
            raise ValueError("truncated TLS ClientHello")
        chunks.extend(chunk)
    return bytes(chunks)


def _take(data: bytes, offset: int, length: int) -> tuple[bytes, int]:
    end = offset + length
    if length < 0 or end > len(data):
        raise ValueError("malformed TLS ClientHello")
    return data[offset:end], end


def _parse_client_hello_sni(payload: bytes) -> str:
    if len(payload) < 34:
        raise ValueError("malformed TLS ClientHello")
    offset = 34  # legacy_version + random
    session_length = payload[offset]
    _, offset = _take(payload, offset + 1, session_length)
    cipher_length_raw, offset = _take(payload, offset, 2)
    cipher_length = int.from_bytes(cipher_length_raw, "big")
    if cipher_length < 2 or cipher_length % 2:
        raise ValueError("malformed TLS cipher list")
    _, offset = _take(payload, offset, cipher_length)
    compression_length_raw, offset = _take(payload, offset, 1)
    compression_length = compression_length_raw[0]
    if compression_length < 1:
        raise ValueError("malformed TLS compression list")
    _, offset = _take(payload, offset, compression_length)
    extensions_length_raw, offset = _take(payload, offset, 2)
    extensions_length = int.from_bytes(extensions_length_raw, "big")
    extensions, offset = _take(payload, offset, extensions_length)
    if offset != len(payload):
        raise ValueError("malformed TLS ClientHello length")

    sni: str | None = None
    ext_offset = 0
    while ext_offset < len(extensions):
        header, ext_offset = _take(extensions, ext_offset, 4)
        extension_type = int.from_bytes(header[:2], "big")
        extension_length = int.from_bytes(header[2:], "big")
        extension, ext_offset = _take(extensions, ext_offset, extension_length)
        # ECH can conceal the effective destination. This proxy has no ECH
        # termination key, so both assigned and draft code points fail closed.
        if extension_type in {0xFE0D, 0xFFCE}:
            raise ValueError("encrypted ClientHello is not supported")
        if extension_type != 0:
            continue
        if sni is not None or len(extension) < 5:
            raise ValueError("malformed or duplicate TLS SNI extension")
        names_length = int.from_bytes(extension[:2], "big")
        if names_length != len(extension) - 2:
            raise ValueError("malformed TLS SNI list")
        name_offset = 2
        names: list[str] = []
        while name_offset < len(extension):
            name_type = extension[name_offset]
            raw_length, name_offset = _take(extension, name_offset + 1, 2)
            name_length = int.from_bytes(raw_length, "big")
            raw_name, name_offset = _take(extension, name_offset, name_length)
            if name_type != 0:
                raise ValueError("unknown TLS server-name type")
            try:
                names.append(raw_name.decode("ascii").lower())
            except UnicodeDecodeError as exc:
                raise ValueError("non-ASCII TLS SNI") from exc
        if len(names) != 1 or not names[0] or names[0].endswith("."):
            raise ValueError("TLS ClientHello must contain one exact DNS SNI")
        sni = names[0]
    if ext_offset != len(extensions) or sni is None:
        raise ValueError("TLS ClientHello has no usable SNI")
    return sni


def _read_client_hello(stream: BinaryIO | io.BufferedIOBase) -> tuple[str, bytes]:
    records = bytearray()
    handshake = bytearray()
    expected_handshake_length: int | None = None
    while len(records) < MAX_CLIENT_HELLO:
        header = _read_exact(stream, 5)
        content_type = header[0]
        legacy_version = header[1:3]
        record_length = int.from_bytes(header[3:], "big")
        if content_type != 22 or legacy_version[0] != 3 or not (1 <= record_length <= 65535):
            raise ValueError("CONNECT tunnel requires a TLS ClientHello")
        body = _read_exact(stream, record_length)
        records.extend(header)
        records.extend(body)
        handshake.extend(body)
        if len(records) > MAX_CLIENT_HELLO:
            raise ValueError("TLS ClientHello exceeds proxy limit")
        if expected_handshake_length is None and len(handshake) >= 4:
            if handshake[0] != 1:
                raise ValueError("first TLS handshake message is not ClientHello")
            expected_handshake_length = 4 + int.from_bytes(handshake[1:4], "big")
            if expected_handshake_length > MAX_CLIENT_HELLO:
                raise ValueError("TLS ClientHello exceeds proxy limit")
        if expected_handshake_length is not None and len(handshake) >= expected_handshake_length:
            if len(handshake) != expected_handshake_length:
                raise ValueError("unexpected TLS handshake data before SNI validation")
            return _parse_client_hello_sni(bytes(handshake[4:])), bytes(records)
    raise ValueError("TLS ClientHello exceeds proxy limit")


def _resolve(host: str, port: int) -> list[tuple]:
    rows = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    if not rows:
        raise OSError("destination did not resolve")
    if not ALLOW_PRIVATE:
        for row in rows:
            address = ipaddress.ip_address(row[4][0])
            if not address.is_global:
                raise OSError("non-public destination address denied")
    return rows


def _parse_authority(authority: str, default_port: int) -> tuple[str, int]:
    if "@" in authority or authority.startswith("["):
        raise ValueError("invalid destination authority")
    if authority.count(":") == 1:
        host, raw_port = authority.rsplit(":", 1)
        port = int(raw_port)
    elif ":" in authority:
        raise ValueError("IP literals are not permitted")
    else:
        host, port = authority, default_port
    host = host.strip().lower().rstrip(".")
    if not host or "*" in host or not (1 <= port <= 65535):
        raise ValueError("invalid destination")
    try:
        ipaddress.ip_address(host)
    except ValueError:
        pass
    else:
        raise ValueError("IP literal destinations are not permitted")
    if (host, port) not in ALLOWLIST:
        raise PermissionError("destination is not allowlisted")
    return host, port


class ProxyHandler(socketserver.StreamRequestHandler):
    timeout = 30
    rbufsize = 0

    def _error(self, code: int, message: str) -> None:
        body = (message + "\n").encode("utf-8", errors="replace")[:1024]
        self.wfile.write(
            f"HTTP/1.1 {code} Denied\r\nConnection: close\r\nContent-Length: {len(body)}\r\n\r\n".encode()
            + body
        )

    def _headers(self) -> tuple[str, list[tuple[bytes, bytes]]]:
        first = self.rfile.readline(MAX_HEADERS + 1)
        if not first or len(first) > MAX_HEADERS or not first.endswith(b"\r\n"):
            raise ValueError("request line too large")
        total = len(first)
        headers: list[tuple[bytes, bytes]] = []
        while True:
            line = self.rfile.readline(MAX_HEADERS - total + 1)
            total += len(line)
            if not line or total > MAX_HEADERS:
                raise ValueError("headers too large")
            if line == b"\r\n":
                break
            if not line.endswith(b"\r\n") or line[:1] in {b" ", b"\t"}:
                raise ValueError("malformed HTTP header")
            raw = line[:-2]
            if b":" not in raw:
                raise ValueError("malformed HTTP header")
            name, value = raw.split(b":", 1)
            if not _HEADER_NAME.fullmatch(name):
                raise ValueError("malformed HTTP header name")
            value = value.strip(b" \t")
            if any(byte < 32 and byte != 9 for byte in value) or 127 in value:
                raise ValueError("malformed HTTP header value")
            headers.append((name.lower(), value))
        return first.decode("iso-8859-1").rstrip("\r\n"), headers

    @staticmethod
    def _forward_headers(
        headers: list[tuple[bytes, bytes]], canonical_host: str,
    ) -> list[bytes]:
        host_values = [value for name, value in headers if name == b"host"]
        if len(host_values) > 1 or any(not value for value in host_values):
            raise ValueError("malformed or duplicate Host header")
        connection_tokens: set[bytes] = set()
        for name, value in headers:
            if name != b"connection":
                continue
            for raw_token in value.split(b","):
                token = raw_token.strip().lower()
                if not token or not _HEADER_NAME.fullmatch(token):
                    raise ValueError("malformed Connection header")
                connection_tokens.add(token)
        denied = _HOP_BY_HOP_HEADERS | connection_tokens | {b"host"}
        forwarded = [
            name + b": " + value + b"\r\n"
            for name, value in headers
            if name not in denied
        ]
        return [f"Host: {canonical_host}\r\n".encode("ascii"), *forwarded]

    @staticmethod
    def _connect(host: str, port: int) -> socket.socket:
        last: OSError | None = None
        for family, socktype, proto, _canon, sockaddr in _resolve(host, port):
            upstream = socket.socket(family, socktype, proto)
            upstream.settimeout(20)
            try:
                upstream.connect(sockaddr)
                return upstream
            except OSError as exc:
                last = exc
                upstream.close()
        raise last or OSError("could not connect")

    def _relay(self, upstream: socket.socket) -> None:
        peers = [self.connection, upstream]
        while True:
            readable, _, _ = select.select(peers, [], [], 60)
            if not readable:
                return
            for source in readable:
                data = source.recv(64 * 1024)
                if not data:
                    return
                target = upstream if source is self.connection else self.connection
                target.sendall(data)

    def handle(self) -> None:
        try:
            request_line, headers = self._headers()
            method, target, version = request_line.split(" ", 2)
            if not _HEADER_NAME.fullmatch(method.encode("ascii")) or version not in {
                "HTTP/1.0", "HTTP/1.1",
            }:
                raise ValueError("malformed HTTP request line")
            if method.upper() == "CONNECT":
                host, port = _parse_authority(target, 443)
                self.wfile.write(b"HTTP/1.1 200 Connection Established\r\n\r\n")
                self.wfile.flush()
                sni, client_hello = _read_client_hello(self.rfile)
                if sni != host:
                    raise PermissionError("TLS SNI does not match CONNECT authority")
                upstream = self._connect(host, port)
                try:
                    upstream.sendall(client_hello)
                    self._relay(upstream)
                finally:
                    upstream.close()
                return

            # Pi's reviewed runtime traffic is HTTPS.  Plaintext proxying is
            # deliberately unsupported: forwarding even one framed request
            # while relaying the connection would let pipelined bytes bypass
            # destination validation.
            raise PermissionError("plaintext HTTP proxying is not supported")
        except PermissionError as exc:
            self._error(403, str(exc))
        except Exception:
            self._error(502, "proxy request rejected")


class Server(socketserver.ThreadingMixIn, socketserver.TCPServer):
    allow_reuse_address = True
    daemon_threads = True


if __name__ == "__main__":
    if not ALLOWLIST:
        raise SystemExit("HERMES_PI_PROXY_ALLOWLIST must not be empty")
    with Server(("0.0.0.0", 8080), ProxyHandler) as server:
        server.serve_forever()
