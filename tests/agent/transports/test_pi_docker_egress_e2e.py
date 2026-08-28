"""Real-Docker egress proof; opt in with HERMES_TEST_PI_IMAGE=<immutable ref>."""
from __future__ import annotations

import http.server
import os
from pathlib import Path
import shutil
import socketserver
import subprocess
import tempfile
import threading
import time
import uuid

import pytest
import run_agent

from agent.transports.pi_docker_sandbox import PiDockerSandbox, is_immutable_image_ref
from agent.transports.pi_rpc_client import PiRpcClient
from agent.transports.pi_rpc_session import PiRpcSession


IMAGE = os.environ.get("HERMES_TEST_PI_IMAGE", "")
pytestmark = pytest.mark.skipif(
    not is_immutable_image_ref(IMAGE),
    reason="set HERMES_TEST_PI_IMAGE to the reviewed immutable Pi image",
)


def _docker(*args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["docker", *args], text=True, capture_output=True, timeout=90, check=check
    )


def _wait_network_endpoint(network: str, host: str, port: int) -> None:
    deadline = time.monotonic() + 30
    last = ""
    while time.monotonic() < deadline:
        probe = _docker(
            "run", "--rm", "--network", network, "--entrypoint", "python3", IMAGE,
            "-c", f"import socket;s=socket.create_connection(({host!r},{port}),2);s.close()",
            check=False,
        )
        if probe.returncode == 0:
            return
        last = probe.stderr
        time.sleep(0.25)
    pytest.fail(f"detached fixture {host}:{port} never became ready: {last}")


def test_built_image_has_no_generated_compile_or_bytecode_caches():
    probe = _docker(
        "run", "--rm", "--network", "none", "--entrypoint", "sh", IMAGE,
        "-c",
        "test \"$PYTHONDONTWRITEBYTECODE\" = 1"
        " && test -z \"$(find /tmp -path '/tmp/node-compile-cache*' -print -quit)\""
        " && test -z \"$(find /opt/hermes/verify-venv -type d -name __pycache__ -print -quit)\""
        " && test -z \"$(find /opt/hermes/verify-venv -type f -name '*.pyc' -print -quit)\""
        " && test ! -e /root/.cache/uv",
        check=False,
    )
    assert probe.returncode == 0, probe.stderr


def test_internal_pi_network_allows_only_allowlisted_proxy_destination():
    suffix = uuid.uuid4().hex[:12]
    internal = f"hermes-pi-e2e-internal-{suffix}"
    egress = f"hermes-pi-e2e-egress-{suffix}"
    fixture = f"hermes-pi-e2e-fixture-{suffix}"
    proxy = f"hermes-pi-e2e-proxy-{suffix}"
    client = f"hermes-pi-e2e-client-{suffix}"

    class Quiet(http.server.SimpleHTTPRequestHandler):
        def log_message(self, *_args):
            pass

    host_server = socketserver.TCPServer(("0.0.0.0", 0), Quiet)
    host_port = host_server.server_address[1]
    host_thread = threading.Thread(target=host_server.serve_forever, daemon=True)
    host_thread.start()
    cert_root = Path(tempfile.mkdtemp(prefix="hermes-pi-tls-"))
    try:
        openssl = shutil.which("openssl")
        assert openssl is not None, "real Pi egress E2E requires openssl"
        certificate = cert_root / "cert.pem"
        key = cert_root / "key.pem"
        generated = subprocess.run(
            [
                openssl, "req", "-x509", "-newkey", "rsa:2048", "-nodes",
                "-keyout", str(key), "-out", str(certificate), "-days", "1",
                "-subj", "/CN=fixture.test",
            ],
            text=True, capture_output=True, timeout=30,
        )
        assert generated.returncode == 0, generated.stderr
        _docker("network", "create", "--internal", internal)
        _docker("network", "create", egress)
        _docker(
            "run", "--detach", "--rm", "--name", fixture,
            "--network", egress, "--network-alias", "fixture.test",
            "--mount", f"type=bind,src={cert_root},dst=/fixture-certs,ro",
            "--entrypoint", "python3", IMAGE,
            "-c",
            "import http.server,ssl;"
            "s=http.server.ThreadingHTTPServer(('0.0.0.0',8443),http.server.SimpleHTTPRequestHandler);"
            "c=ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER);"
            "c.load_cert_chain('/fixture-certs/cert.pem','/fixture-certs/key.pem');"
            "s.socket=c.wrap_socket(s.socket,server_side=True);s.serve_forever()",
        )
        _docker(
            "run", "--detach", "--rm", "--name", proxy,
            "--network", internal, "--network-alias", "pi-egress-proxy",
            "--read-only", "--cap-drop=ALL", "--security-opt=no-new-privileges:true",
            "--user", "65532:65532",
            "--env", "HERMES_PI_PROXY_ALLOWLIST=fixture.test:8443",
            "--env", "HERMES_PI_PROXY_TEST_ALLOW_PRIVATE=1",
            "--entrypoint", "python3", IMAGE, "/opt/hermes/pi_egress_proxy.py",
        )
        _docker("network", "connect", egress, proxy)
        _wait_network_endpoint(egress, "fixture.test", 8443)
        _wait_network_endpoint(internal, "pi-egress-proxy", 8080)

        metadata = _docker("network", "inspect", internal).stdout
        assert '"Internal": true' in metadata
        proxy_metadata = _docker("container", "inspect", proxy).stdout
        assert '"Mounts": []' in proxy_metadata

        allowed = _docker(
            "run", "--rm", "--name", client, "--network", internal,
            "--env", "HTTPS_PROXY=http://pi-egress-proxy:8080",
            "--env", "https_proxy=http://pi-egress-proxy:8080",
            "--entrypoint", "python3", IMAGE, "-c",
            "import ssl,urllib.request;"
            "c=ssl._create_unverified_context();"
            "h=urllib.request.HTTPSHandler(context=c);"
            "r=urllib.request.build_opener(h).open('https://fixture.test:8443/',timeout=15);"
            "assert r.status==200",
            check=False,
        )
        assert allowed.returncode == 0, allowed.stderr

        fronted = _docker(
            "run", "--rm", "--network", internal,
            "--entrypoint", "python3", IMAGE, "-c",
            "import socket,ssl;"
            "s=socket.create_connection(('pi-egress-proxy',8080),5);"
            "s.sendall(b'CONNECT fixture.test:8443 HTTP/1.1\\r\\nHost: fixture.test:8443\\r\\n\\r\\n');"
            "h=b'';"
            "\nwhile b'\\r\\n\\r\\n' not in h: h+=s.recv(4096);"
            "\nassert b' 200 ' in h;"
            "\nc=ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT);c.check_hostname=False;c.verify_mode=ssl.CERT_NONE;"
            "\ntry: c.wrap_socket(s,server_hostname='fronted.test').do_handshake()"
            "\nexcept (ssl.SSLError,EOFError,ConnectionError,OSError): raise SystemExit(0)"
            "\nraise SystemExit(9)",
            check=False,
        )
        assert fronted.returncode == 0, fronted.stderr

        non_tls = _docker(
            "run", "--rm", "--network", internal,
            "--entrypoint", "python3", IMAGE, "-c",
            "import errno,socket;"
            "s=socket.create_connection(('pi-egress-proxy',8080),5);"
            "s.sendall(b'CONNECT fixture.test:8443 HTTP/1.1\\r\\nHost: fixture.test:8443\\r\\n\\r\\n');"
            "h=b'';"
            "\nwhile b'\\r\\n\\r\\n' not in h: h+=s.recv(4096);"
            "\nassert b' 200 ' in h;"
            "\ntry:s.sendall(b'not tls');s.shutdown(socket.SHUT_WR)"
            "\nexcept OSError as e:"
            "\n assert e.errno in {errno.ECONNRESET,errno.ENOTCONN,errno.EPIPE};raise SystemExit(0)"
            "\ntry:r=s.recv(4096)"
            "\nexcept OSError as e:"
            "\n assert e.errno in {errno.ECONNRESET,errno.ENOTCONN};raise SystemExit(0)"
            "\nassert r == b'' or r.startswith(b'HTTP/1.1 502 Denied')",
            check=False,
        )
        assert non_tls.returncode == 0, non_tls.stderr

        direct = _docker(
            "run", "--rm", "--network", internal,
            "--entrypoint", "python3", IMAGE, "-c",
            "import socket;socket.create_connection(('api.openai.com',443),5)",
            check=False,
        )
        assert direct.returncode != 0

        host = _docker(
            "run", "--rm", "--network", internal,
            "--add-host", "host.docker.internal:host-gateway",
            "--entrypoint", "python3", IMAGE, "-c",
            f"import socket;socket.create_connection(('host.docker.internal',{host_port}),5)",
            check=False,
        )
        assert host.returncode != 0
    finally:
        host_server.shutdown()
        host_server.server_close()
        for name in (client, proxy, fixture):
            _docker("rm", "--force", name, check=False)
        for name in (internal, egress):
            _docker("network", "rm", name, check=False)
        shutil.rmtree(cert_root, ignore_errors=True)
    leftovers = _docker(
        "container", "ls", "-a", "--filter", f"name={suffix}", "--format", "{{.Names}}",
        check=False,
    )
    assert leftovers.returncode == 0 and leftovers.stdout.strip() == ""
    for name in (internal, egress):
        assert _docker("network", "inspect", name, check=False).returncode != 0


def test_real_runtime_mounts_deny_host_and_git_common_dir_escape(tmp_path):
    work = tmp_path / "work"
    config = tmp_path / "config"
    sessions = tmp_path / "sessions"
    outside = tmp_path / "outside"
    for directory in (work, config, sessions, outside):
        directory.mkdir()
    (config / "hermes_bridge.ts").write_text("stub", encoding="utf-8")
    secret = outside / "secret.txt"
    secret.write_text("must-not-cross-boundary", encoding="utf-8")
    (work / "escape-link").symlink_to(secret)
    (work / ".git").write_text(f"gitdir: {outside / 'git-common'}\n", encoding="utf-8")

    from agent.transports.pi_docker_sandbox import PiDockerSandbox

    sandbox = PiDockerSandbox(
        image=IMAGE, worktree=work, config_dir=config, session_dir=sessions,
        extension_file=config / "hermes_bridge.ts",
        egress_allowlist={"fixture.test:8443"},
    )
    sandbox.inspect(timeout=20)
    command = [
        *sandbox._base_run_args(
            workspace_read_only=True, network_none=True, include_provider_env=False,
        ),
        "--entrypoint", "python3", IMAGE, "-c",
        "from pathlib import Path;"
        "p=Path('/workspace/escape-link');"
        "g=Path('/workspace/.git').read_text().split(':',1)[1].strip();"
        "assert not p.exists();"
        "assert not Path(g).exists();"
        "assert 'must-not-cross-boundary' not in str(list(Path('/workspace').rglob('*')))",
    ]
    result = subprocess.run(command, text=True, capture_output=True, timeout=60)
    assert result.returncode == 0, result.stderr
    joined = " ".join(command)
    assert str(outside) not in joined
    assert "docker.sock" not in joined

    no_git = _docker(
        "run", "--rm", "--network", "none", "--entrypoint", "sh", IMAGE,
        "-c", "! command -v git",
        check=False,
    )
    assert no_git.returncode == 0, no_git.stderr


def test_built_image_native_rpc_smoke_is_unauthenticated_and_schema_clean(tmp_path):
    work = tmp_path / "work"
    config = tmp_path / "config"
    sessions = tmp_path / "sessions"
    for directory in (work, config, sessions):
        directory.mkdir()
    source = Path("agent/transports/pi_assets/hermes_bridge.ts").resolve(strict=True)
    extension = config / "hermes_bridge.ts"
    extension.write_bytes(source.read_bytes())
    sandbox = PiDockerSandbox(
        image=IMAGE, worktree=work, config_dir=config, session_dir=sessions,
        extension_file=extension, read_only_workspace=True,
    )
    sandbox.inspect(timeout=20)
    pi_args = [
        "--mode", "rpc", "--no-extensions", "--no-skills",
        "--no-prompt-templates", "--no-themes", "--no-context-files",
        "--no-approve", "--offline", "--no-session", "--tools", "read,edit,write",
    ]
    command = [
        *sandbox._base_run_args(
            workspace_read_only=True, network_none=True, include_provider_env=False,
        ),
        "--entrypoint", "pi", IMAGE, *pi_args,
    ]
    client = PiRpcClient(None, cwd=str(work), launch_command=command)
    try:
        state = client.request("get_state", timeout=20)
        models = client.request("get_available_models", timeout=20)
    finally:
        client.close(force=True)
    assert state.get("command") == "get_state" and isinstance(state.get("data"), dict)
    assert models.get("command") == "get_available_models" and isinstance(models.get("data"), dict)


def test_public_agent_path_uses_real_contained_session_without_credentials(
    tmp_path, monkeypatch
):
    work = tmp_path / "work"
    runtime = tmp_path / "runtime"
    config = runtime / "managed-config"
    sessions = runtime / "sessions" / "public-agent"
    bindings = runtime / "bindings"
    home = runtime / "home"
    for directory in (work, config, sessions, bindings, home):
        directory.mkdir(parents=True, exist_ok=True)
    extension = config / "hermes_bridge.ts"
    extension.write_bytes(
        Path("agent/transports/pi_assets/hermes_bridge.ts").resolve(strict=True).read_bytes()
    )
    sandbox = PiDockerSandbox(
        image=IMAGE,
        worktree=work,
        config_dir=config,
        session_dir=sessions,
        binding_root=bindings,
        sensitive_roots=(runtime, sessions.parent, home),
        extension_file=extension,
        read_only_workspace=True,
        provider_env_names=set(),
        egress_allowlist={"api.openai.com"},
    )
    session = PiRpcSession(
        executable=None,
        cwd=str(work),
        config_dir=config,
        session_dir=sessions,
        binding_path=bindings / "public-agent.json",
        silence_timeout=20,
        turn_timeout=60,
        mode="inspect",
        sandbox=sandbox,
        allowed_provider_vars=set(),
    )
    monkeypatch.setattr("agent.pi_runtime.get_or_create_pi_session", lambda _agent: session)
    agent = run_agent.AIAgent(
        api_key="non-provider-placeholder",
        base_url="https://invalid.local",
        provider="openai",
        api_mode="pi_rpc",
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
        session_id="docker-no-auth-public-path",
    )
    try:
        result = agent.run_conversation("Report whether a model is available; do not edit files.")
    finally:
        session.close()
        sandbox.cleanup()

    assert isinstance(result, dict)
    external = result.get("external_runtime")
    assert isinstance(external, dict)
    assert external["runtime"] == "pi_rpc"
    assert external["status"] in {"completed", "error", "reconciliation_required"}
    assert external["prompt_accepted"] in {True, False, None}
    if external["prompt_accepted"] is None:
        assert external["reconciliation_required"] is True
    if not result.get("completed"):
        error = str(result.get("error") or "").lower()
        assert error
        assert any(word in error for word in ("auth", "model", "provider", "credential"))
