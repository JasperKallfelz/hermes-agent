import hashlib
import argparse
import base64
import csv
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from agent import pi_runtime
from agent.external_runtime.binding import RuntimeBinding, save_binding, worktree_identity
from agent.transports import pi_runtime_doctor
from agent.transports.pi_docker_sandbox import PiSandboxError
from agent.transports.pi_assets.verify_npm_lock import (
    IMAGE_NPM_TREE_SHA256,
    NPM_TREE_SHA256,
    verify_lock_integrities,
    verify_npm_tree,
    verify_runtime_bundle,
)
from hermes_cli.pi_runtime_cli import register_cli
from scripts.generate_pi_runtime_evidence import generate


def test_doctor_runs_schema_queries_inside_read_only_docker(monkeypatch, tmp_path):
    root = tmp_path / "runtime"
    config = root / "managed-config"
    sessions = root / "sessions"
    bindings = root / "bindings"
    home = root / "home"
    for directory in (root, config, sessions, bindings, home):
        directory.mkdir(exist_ok=True)
    executable = root / "pi"
    executable.write_text("stub", encoding="utf-8")
    extension = config / "hermes_bridge.ts"
    extension.write_text("stub", encoding="utf-8")
    work = tmp_path / "work"
    work.mkdir()
    paths = SimpleNamespace(
        root=root,
        executable=executable,
        config_dir=config,
        session_root=sessions,
        binding_root=bindings,
        home_dir=home,
        extension_file=extension,
    )
    monkeypatch.setattr(pi_runtime_doctor, "managed_pi_paths", lambda **kwargs: paths)
    seen = {}

    def verify_host(paths, run_version=True):
        seen["host_run_version"] = run_version
        return str(executable)

    monkeypatch.setattr(pi_runtime_doctor, "verify_managed_pi", verify_host)

    class Sandbox:
        def __init__(self, **kwargs):
            seen["sandbox"] = kwargs

        def verify_version(self, **kwargs):
            seen["verified"] = True

        def prepare_egress(self, **kwargs):
            seen["egress_prepared"] = True

        def inspect_egress(self, **kwargs):
            seen["egress_inspected"] = True
            return {}

        def pi_launch_command(self, args):
            seen["pi_args"] = args
            return ["docker", "run", "contained-pi"]

        def cleanup(self, **kwargs):
            seen["cleaned"] = True

    monkeypatch.setattr(pi_runtime_doctor, "PiDockerSandbox", Sandbox)

    class Client:
        def __init__(self, exe, **kwargs):
            assert exe is None
            seen.update(kwargs)

        def request(self, command, **kwargs):
            return {"type": "response", "command": command, "success": True, "data": {}}

        def close(self, **kwargs):
            pass

    monkeypatch.setattr(pi_runtime_doctor, "PiRpcClient", Client)
    report = pi_runtime_doctor.check_pi_runtime(
        cwd=str(work), hermes_home=tmp_path,
        docker_image="sha256:" + "a" * 64,
    )
    assert report.ok is True
    assert report.rpc_ok is True
    assert "RPC handshake" in report.checks
    assert seen["host_run_version"] is False
    assert seen["sandbox"]["read_only_workspace"] is True
    assert seen["sandbox"]["image"] == "sha256:" + "a" * 64
    assert seen["egress_prepared"] is True
    assert seen["egress_inspected"] is True
    assert seen["launch_command"] == ["docker", "run", "contained-pi"]
    assert "--no-extensions" in seen["pi_args"]
    assert seen["cleaned"] is True


def test_pi_runtime_cli_exposes_install_and_interactive_auth_actions():
    parser = argparse.ArgumentParser()
    register_cli(parser.add_subparsers(dest="command"))
    install = parser.parse_args(["pi-runtime", "install", "--npm", "fake-npm"])
    with pytest.raises(SystemExit):
        parser.parse_args(["pi-runtime", "auth"])
    auth = parser.parse_args([
        "pi-runtime", "auth", "--acknowledge-trusted-local",
    ])
    assert install.pi_runtime_command == "install"
    assert install.npm == "fake-npm"
    assert auth.pi_runtime_command == "auth"
    assert auth.acknowledge_trusted_local is True


def test_doctor_fails_closed_without_interactive_image(monkeypatch, tmp_path):
    root = tmp_path / "runtime"
    root.mkdir()
    paths = SimpleNamespace(root=root, executable=root / "pi")
    monkeypatch.setattr(pi_runtime_doctor, "managed_pi_paths", lambda **kwargs: paths)
    monkeypatch.setattr(pi_runtime_doctor, "verify_managed_pi", lambda paths, run_version=True: "pi")
    report = pi_runtime_doctor.check_pi_runtime(cwd=str(tmp_path), hermes_home=tmp_path)
    assert report.ok is False
    assert "interactive Pi requires" in str(report.error)


def test_all_pi_runtime_assets_are_declared_as_package_data():
    pyproject = Path("pyproject.toml").read_text(encoding="utf-8")
    assert '"transports/pi_assets/*.ts"' in pyproject
    assert '"transports/pi_assets/Dockerfile"' in pyproject
    assert '"transports/pi_assets/*.sh"' in pyproject
    assets = Path("agent/transports/pi_assets")
    assert (assets / "hermes_bridge.ts").is_file()
    dockerfile = (assets / "Dockerfile").read_text(encoding="utf-8")
    helper = (assets / "build-image.sh").read_text(encoding="utf-8")
    base_stages = [line for line in dockerfile.splitlines() if line.startswith("FROM ")]
    assert len(base_stages) == 2
    assert all("@sha256:" in line for line in base_stages)
    assert "FROM node:26-bookworm-slim@sha256:" in dockerfile
    assert '@earendil-works/pi-coding-agent@0.84.3' in dockerfile
    assert 'org.hermes.pi.version="0.84.3"' in dockerfile
    assert 'test "$(npm --version)" = "11.17.0"' in dockerfile
    assert "npm ci --omit=dev" in dockerfile
    assert "npm-shrinkwrap.json" in dockerfile
    assert "package.json package-lock.json" in dockerfile
    for forbidden in ("apt-get", "build-essential", "curl", "python3-pip", "--system"):
        assert forbidden not in dockerfile
    assert "uv venv" in dockerfile
    assert "--require-hashes" in dockerfile
    assert "/opt/hermes/verify-venv/bin/pytest" in dockerfile
    assert (assets / "package-lock.json").is_file()
    bridge = (assets / "hermes_bridge.ts").read_text(encoding="utf-8")
    assert "realpathSync" in bridge
    assert "lstatSync" in bridge
    assert "readlinkSync" in bridge
    assert 'event.toolName === "bash"' in bridge
    assert "DANGEROUS" not in bridge
    assert "image inspect" in helper
    assert "Configure this immutable local image ID" in helper
    assert "source_date_epoch=1704067200" in helper
    assert 'SOURCE_DATE_EPOCH=$source_date_epoch DOCKER_BUILDKIT=1' in helper
    assert 'buildx build' in helper
    assert '--build-arg "SOURCE_DATE_EPOCH=$source_date_epoch"' in helper
    assert 'target_platform=linux/arm64' in helper
    assert '--platform "$target_platform"' in helper
    assert "--provenance=false" in helper
    assert 'type=docker,dest=$image_archive,rewrite-timestamp=true' in helper
    assert '"$docker_bin" load --input "$image_archive" >/dev/null' in helper
    assert "--no-cache" in helper
    assert "ARG SOURCE_DATE_EPOCH=1704067200" in dockerfile
    assert "ENV SOURCE_DATE_EPOCH=${SOURCE_DATE_EPOCH}" in dockerfile
    assert "PYTHONDONTWRITEBYTECODE=1" in dockerfile
    integrity = json.loads((assets / "runtime-integrity.json").read_text(encoding="utf-8"))
    assert integrity["version"] == pi_runtime.PINNED_PI_VERSION
    assert integrity["source_date_epoch"] == 1704067200
    assert integrity["tarball_sha512"] == pi_runtime.PINNED_TARBALL_SHA512
    assert integrity["package_json_sha256"] == pi_runtime.PINNED_PACKAGE_JSON_SHA256
    assert integrity["cli_sha256"] == pi_runtime.PINNED_CLI_SHA256
    assert integrity["shrinkwrap_sha256"] == pi_runtime.PINNED_SHRINKWRAP_SHA256
    assert integrity["manifest_sha256"] == pi_runtime.PINNED_MANIFEST_SHA256
    assert integrity["manifest_sha256"] == hashlib.sha256(
        (assets / "package-lock.json").read_bytes()
    ).hexdigest()
    assert integrity["egress_proxy_sha256"] == hashlib.sha256(
        (assets / "egress_proxy.py").read_bytes()
    ).hexdigest()
    assert (
        f'org.hermes.pi.egress-proxy.sha256="{integrity["egress_proxy_sha256"]}"'
        in dockerfile
    )
    assert integrity["bridge_sha256"] == hashlib.sha256(
        (assets / "hermes_bridge.ts").read_bytes()
    ).hexdigest()
    assert integrity["npm_lock_helper_sha256"] == hashlib.sha256(
        (assets / "verify_npm_lock.py").read_bytes()
    ).hexdigest()
    assert integrity["verification_requirements_sha256"] == hashlib.sha256(
        (assets / "verification-requirements.txt").read_bytes()
    ).hexdigest()
    assert integrity["registry_npm_tree_sha256"] == NPM_TREE_SHA256
    assert integrity["image_npm_tree_sha256"] == IMAGE_NPM_TREE_SHA256
    verify_runtime_bundle(assets)


def test_pi_runtime_ci_compares_deterministic_ids_before_post_merge_evidence():
    workflow = Path(".github/workflows/pi-runtime.yml").read_text(encoding="utf-8")
    first = "IMAGE_ID=$(agent/transports/pi_assets/build-image.sh docker hermes-pi-ci:first"
    second = "SECOND_IMAGE_ID=$(agent/transports/pi_assets/build-image.sh docker hermes-pi-ci:second"
    equality = 'if [ "$IMAGE_ID" != "$SECOND_IMAGE_ID" ]; then'
    assert workflow.index(first) < workflow.index(second) < workflow.index(equality)
    assert 'echo "image_id=$IMAGE_ID" >> "$GITHUB_OUTPUT"' in workflow
    assert "Generate complete post-merge SBOM and source provenance" in workflow
    assert "github.event_name == 'push' && github.ref == 'refs/heads/main'" in workflow
    assert "pi-runtime-sbom-provenance-${{ github.sha }}" in workflow
    generator = Path("scripts/generate_pi_runtime_evidence.py").read_text(encoding="utf-8")
    assert '"source_date_epoch": 1704067200' in generator


def test_pi_runtime_ci_requires_full_npm_lock_regeneration_and_result_wiring():
    workflow = Path(".github/workflows/pi-runtime.yml").read_text(encoding="utf-8")
    lock_step = workflow.split(
        "- name: Fully regenerate and compare the Pi npm lock", 1,
    )[1].split("- name:", 1)[0]
    assert "npm install --global npm@11.17.0" in lock_step
    assert "verify_npm_lock.py --npm \"$NPM_CLI\"" in lock_step
    assert "--integrity-only" not in lock_step
    assert "continue-on-error" not in lock_step
    assert "needs: [unit-and-package, windows-path-parser]" in workflow

    ci = Path(".github/workflows/ci.yaml").read_text(encoding="utf-8")
    pi_job = ci.split("\n  pi-runtime:\n", 1)[1].split("\n  docker:\n", 1)[0]
    gate = ci.split("\n  all-checks-pass:\n", 1)[1]
    assert "uses: ./.github/workflows/pi-runtime.yml" in pi_job
    assert "id-token: write" in pi_job
    assert "attestations: write" in pi_job
    assert "      - pi-runtime" in gate


def test_pi_runtime_ci_captures_complete_sbom_from_image_and_attests_it():
    workflow = Path(".github/workflows/pi-runtime.yml").read_text(encoding="utf-8")
    for required in (
        "docker image inspect \"$RUNTIME_IMAGE_ID\"",
        "--entrypoint dpkg-query \"$RUNTIME_IMAGE_ID\"",
        "image_run cat /opt/hermes/pi-build/package-lock.json",
        "image_run cat /opt/hermes/pi-build/verification-requirements.txt",
        "image_run uv pip list --python /opt/hermes/verify-venv/bin/python --format json",
        "image_run node --version",
        "image_run npm --version",
        "image_run python3 --version",
        "image_run uv --version",
        "docker version --format '{{json .}}'",
        "docker buildx version",
        "scripts/generate_pi_runtime_evidence.py",
    ):
        assert required in workflow
    assert 'runs-on: ubuntu-24.04-arm' in workflow
    assert "trusted-provenance-attestation:" in workflow
    attestation = workflow.split("\n  trusted-provenance-attestation:\n", 1)[1]
    assert "needs: docker-containment" in attestation
    assert "github.event_name == 'push' && github.ref == 'refs/heads/main'" in attestation
    assert "id-token: write" in attestation
    assert "attestations: write" in attestation
    assert "actions/attest-build-provenance@a2bbfa25375fe432b6a289bc6b6cd05ecd0c4c32" in attestation
    assert "subject-name: hermes-pi-runtime" in attestation
    assert "subject-digest: ${{ needs.docker-containment.outputs.image_id }}" in attestation
    for action in re.findall(r"\buses:\s*([^\s]+)", workflow):
        if action.startswith("./"):
            continue
        reference = action.rsplit("@", 1)[-1]
        assert re.fullmatch(r"[0-9a-f]{40}", reference), action


def _write_pi_evidence_fixture(root: Path) -> None:
    image_id = "sha256:" + "a" * 64
    (root / "image-inspect.json").write_text(json.dumps([{
        "Id": image_id, "Os": "linux", "Architecture": "arm64",
    }]), encoding="utf-8")
    (root / "dpkg-packages.tsv").write_text(
        "base-files\t13.8+deb13u2\tarm64\n", encoding="utf-8",
    )
    (root / "package-lock.json").write_text(json.dumps({
        "lockfileVersion": 3,
        "packages": {
            "": {"name": "hermes-pi-runtime", "version": "0.84.3"},
            "node_modules/@earendil-works/pi-coding-agent": {
                "version": "0.84.3",
            },
        },
    }), encoding="utf-8")
    (root / "verification-requirements.txt").write_text(
        "iniconfig==2.3.0 \\\n    --hash=sha256:" + "a" * 64 + "\n",
        encoding="utf-8",
    )
    (root / "verify-python-packages.json").write_text(json.dumps([
        {"name": "iniconfig", "version": "2.3.0"},
    ]), encoding="utf-8")
    for name, value in {
        "node-version.txt": "v26.0.0\n",
        "npm-version.txt": "11.17.0\n",
        "python-version.txt": "Python 3.13.11\n",
        "uv-version.txt": "uv 0.11.6\n",
        "buildkit-version.txt": "github.com/docker/buildx v0.30.1 buildkit v0.27.0\n",
    }.items():
        (root / name).write_text(value, encoding="utf-8")
    (root / "docker-version.json").write_text(json.dumps({
        "Client": {"Version": "29.1.5"},
        "Server": {"Version": "29.1.5"},
    }), encoding="utf-8")


def _set_pi_evidence_environment(monkeypatch) -> None:
    for name, value in {
        "RUNTIME_IMAGE_ID": "sha256:" + "a" * 64,
        "SOURCE_REPOSITORY": "example/hermes",
        "SOURCE_REVISION": "b" * 40,
        "SOURCE_REF": "refs/heads/main",
        "WORKFLOW_RUN_ID": "123",
        "WORKFLOW_RUN_ATTEMPT": "1",
    }.items():
        monkeypatch.setenv(name, value)


def test_pi_runtime_evidence_has_every_required_category_and_tool_version(
    tmp_path, monkeypatch,
):
    _write_pi_evidence_fixture(tmp_path)
    _set_pi_evidence_environment(monkeypatch)
    sbom, provenance = generate(tmp_path)
    categories = {
        property_["value"]
        for component in sbom["components"]
        for property_ in component["properties"]
        if property_["name"] == "hermes:category"
    }
    assert categories == {
        "image-dpkg", "pi-npm-lock", "runtime-tool", "verify-python-lock",
    }
    assert provenance["target"] == {"os": "linux", "architecture": "arm64"}
    assert set(provenance["tools"]) == {
        "docker_client", "docker_server", "buildkit", "node", "npm", "python", "uv",
    }


@pytest.mark.parametrize(
    ("filename", "replacement"),
    [
        ("dpkg-packages.tsv", ""),
        ("package-lock.json", '{"lockfileVersion": 3, "packages": {}}'),
        ("verify-python-packages.json", "[]"),
        ("node-version.txt", ""),
    ],
)
def test_pi_runtime_evidence_fails_closed_when_a_category_is_absent(
    tmp_path, monkeypatch, filename, replacement,
):
    _write_pi_evidence_fixture(tmp_path)
    (tmp_path / filename).write_text(replacement, encoding="utf-8")
    _set_pi_evidence_environment(monkeypatch)
    with pytest.raises(RuntimeError):
        generate(tmp_path)


def test_pi_runtime_desktop_warning_gate_catches_unawaited_act():
    workflow = Path(".github/workflows/pi-runtime.yml").read_text(encoding="utf-8")
    assert "not wrapped in act" in workflow
    assert r"You called act\(" in workflow
    assert "without await" in workflow


def test_pi_runtime_dockerfile_removes_inherited_git_without_installing_it():
    dockerfile = Path("agent/transports/pi_assets/Dockerfile").read_text(
        encoding="utf-8"
    )
    lowered = dockerfile.lower()
    assert "apt-get" not in lowered
    assert "apt install" not in lowered
    assert "aptitude" not in lowered
    assert "rm -f /usr/bin/git /usr/bin/git-* /usr/bin/scalar" in dockerfile
    for path in (
        "/usr/lib/git-core",
        "/usr/libexec/git-core",
        "/usr/share/git-core",
        "/usr/share/gitweb",
        "/usr/share/perl5/Git",
        "/usr/share/perl5/Git.pm",
    ):
        assert f"test ! -e {path}" in dockerfile
    assert "! command -v git" in dockerfile
    assert "! command -v scalar" in dockerfile
    assert "-name 'git-*'" in dockerfile


def test_pi_runtime_dockerfile_removes_generated_caches_in_the_creating_runs():
    dockerfile = Path("agent/transports/pi_assets/Dockerfile").read_text(
        encoding="utf-8"
    )
    instructions = dockerfile.replace("\\\n", " ").splitlines()
    npm_run = next(
        line for line in instructions if line.startswith("RUN ") and "npm pack" in line
    )
    venv_run = next(
        line for line in instructions if line.startswith("RUN ") and "uv venv" in line
    )

    assert npm_run.index('test "$(pi --version)" = "0.84.3"') < npm_run.index(
        "rm -rf /root/.npm /tmp/node-compile-cache /tmp/npm-*"
    )
    assert venv_run.count(
        "PYTHONDONTWRITEBYTECODE=1 /opt/hermes/verify-venv/bin/"
    ) == 3
    assert "find /opt/hermes/verify-venv -type d -name __pycache__" in venv_run
    assert "find /opt/hermes/verify-venv -type f -name '*.pyc' -delete" in venv_run
    assert "rm -rf /root/.cache/uv /tmp/uv-*" in venv_run
    assert venv_run.index("/opt/hermes/verify-venv/bin/ty --version") < venv_run.index(
        "find /opt/hermes/verify-venv -type d -name __pycache__"
    )


def _write_fake_docker_for_archive_test(path: Path) -> None:
    path.write_text(
        "#!/bin/sh\n"
        "printf '%s|%s\\n' \"${SOURCE_DATE_EPOCH:-}\" \"$*\" >> \"$FAKE_DOCKER_LOG\"\n"
        "if [ \"$1 $2\" = 'buildx build' ]; then\n"
        "  image_archive=\n"
        "  for argument in \"$@\"; do\n"
        "    case \"$argument\" in\n"
        "      type=docker,dest=*,rewrite-timestamp=true)\n"
        "        image_archive=${argument#type=docker,dest=}\n"
        "        image_archive=${image_archive%,rewrite-timestamp=true}\n"
        "        ;;\n"
        "    esac\n"
        "  done\n"
        "  test -n \"$image_archive\"\n"
        "  printf 'private docker archive\\n' > \"$image_archive\"\n"
        "  printf '%s\\n' \"$image_archive\" > \"$FAKE_DOCKER_ARCHIVE_PATH\"\n"
        "  if [ \"${FAKE_DOCKER_FAIL_BUILD:-0}\" = 1 ]; then exit 17; fi\n"
        "  exit 0\n"
        "fi\n"
        "if [ \"$1\" = load ]; then\n"
        "  test \"$2\" = --input\n"
        "  test \"$3\" = \"$(cat \"$FAKE_DOCKER_ARCHIVE_PATH\")\"\n"
        "  test -s \"$3\"\n"
        "  archive_mode=$(stat -c %a \"$3\" 2>/dev/null || stat -f %Lp \"$3\")\n"
        "  directory_mode=$(stat -c %a \"$(dirname \"$3\")\" 2>/dev/null || stat -f %Lp \"$(dirname \"$3\")\")\n"
        "  test \"$archive_mode\" = 600\n"
        "  test \"$directory_mode\" = 700\n"
        "  echo 'Loaded image ID: sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb'\n"
        "  exit 0\n"
        "fi\n"
        "if [ \"$1 $2 $3\" = 'image inspect --format' ]; then\n"
        "  case \"$4\" in\n"
        "    *org.hermes.pi.version*) echo 0.84.3;;\n"
        "    *) echo sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa;;\n"
        "  esac\n"
        "  exit 0\n"
        "fi\n"
        "if [ \"$1\" = run ]; then echo 0.84.3; exit 0; fi\n"
        "exit 2\n",
        encoding="utf-8",
    )
    path.chmod(0o700)


def test_build_image_exports_loads_inspects_and_cleans_private_archive(tmp_path):
    helper = Path("agent/transports/pi_assets/build-image.sh").resolve(strict=True)
    fake_docker = tmp_path / "selected-docker"
    fake_docker_log = tmp_path / "fake-docker.log"
    archive_path_log = tmp_path / "archive-path.log"
    private_temp_root = tmp_path / "private temp root"
    private_temp_root.mkdir()
    _write_fake_docker_for_archive_test(fake_docker)
    env = {
        **os.environ,
        "TMPDIR": str(private_temp_root),
        "FAKE_DOCKER_LOG": str(fake_docker_log),
        "FAKE_DOCKER_ARCHIVE_PATH": str(archive_path_log),
    }

    result = subprocess.run(
        [str(helper), str(fake_docker), "exact-pi:0.84.3"],
        env=env,
        text=True,
        capture_output=True,
        timeout=30,
    )

    assert result.returncode == 0, result.stderr
    assert "sha256:" + "b" * 64 not in result.stdout
    assert result.stdout.splitlines()[-1] == "sha256:" + "a" * 64
    invocations = fake_docker_log.read_text(encoding="utf-8").splitlines()
    build = next(line for line in invocations if "|buildx build " in line)
    assert build.startswith("1704067200|buildx build ")
    assert " --no-cache " in build
    assert " --platform linux/arm64 " in build
    assert " --build-arg SOURCE_DATE_EPOCH=1704067200 " in build
    assert " --provenance=false " in build
    assert "type=docker,dest=" in build
    assert ",rewrite-timestamp=true " in build
    archive = Path(archive_path_log.read_text(encoding="utf-8").strip())
    assert any(f"|load --input {archive}" == line for line in invocations)
    immutable = "sha256:" + "a" * 64
    assert any(
        line.endswith("image inspect --format {{.Id}} exact-pi:0.84.3")
        for line in invocations
    )
    assert any(
        line.endswith(
            'image inspect --format {{index .Config.Labels "org.hermes.pi.version"}} '
            + immutable
        )
        for line in invocations
    )
    assert any(
        line.endswith(f"run --rm --network none --entrypoint pi {immutable} --version")
        for line in invocations
    )
    assert not archive.exists()
    assert not archive.parent.exists()

    failed_log = tmp_path / "failed-docker.log"
    failed_archive_log = tmp_path / "failed-archive-path.log"
    failed = subprocess.run(
        [str(helper), str(fake_docker), "exact-pi:failed"],
        env={
            **env,
            "FAKE_DOCKER_LOG": str(failed_log),
            "FAKE_DOCKER_ARCHIVE_PATH": str(failed_archive_log),
            "FAKE_DOCKER_FAIL_BUILD": "1",
        },
        text=True,
        capture_output=True,
        timeout=30,
    )
    assert failed.returncode == 17
    failed_archive = Path(failed_archive_log.read_text(encoding="utf-8").strip())
    assert not failed_archive.exists()
    assert not failed_archive.parent.exists()


def test_registry_and_image_npm_tree_reviews_are_distinct_and_not_cross_accepted(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(
        "agent.transports.pi_assets.verify_npm_lock.npm_package_root",
        lambda _path: tmp_path,
    )
    monkeypatch.setattr(
        "agent.transports.pi_assets.verify_npm_lock.canonical_tree_hash",
        lambda _path: IMAGE_NPM_TREE_SHA256,
    )
    assert verify_npm_tree(tmp_path / "npm-cli.js", source="image") == tmp_path
    with pytest.raises(RuntimeError, match="registry package tree"):
        verify_npm_tree(tmp_path / "npm-cli.js", source="registry")

    monkeypatch.setattr(
        "agent.transports.pi_assets.verify_npm_lock.canonical_tree_hash",
        lambda _path: NPM_TREE_SHA256,
    )
    assert verify_npm_tree(tmp_path / "npm-cli.js", source="registry") == tmp_path
    with pytest.raises(RuntimeError, match="image package tree"):
        verify_npm_tree(tmp_path / "npm-cli.js", source="image")


def test_every_remote_production_npm_lock_entry_has_sha512_integrity(tmp_path):
    lock = Path("agent/transports/pi_assets/package-lock.json")
    verify_lock_integrities(lock)
    payload = json.loads(lock.read_text(encoding="utf-8"))
    remote = {
        name: entry for name, entry in payload["packages"].items()
        if name and str(entry.get("resolved") or "").startswith("https://")
    }
    assert remote
    assert all(str(entry.get("integrity") or "").startswith("sha512-") for entry in remote.values())
    assert {
        "node_modules/@earendil-works/pi-agent-core",
        "node_modules/@earendil-works/pi-ai",
        "node_modules/@earendil-works/pi-client",
        "node_modules/@earendil-works/pi-protocol",
        "node_modules/@earendil-works/pi-telemetry",
        "node_modules/@earendil-works/pi-tui",
    } <= set(remote)

    broken = tmp_path / "package-lock.json"
    broken.write_text(json.dumps({
        "lockfileVersion": 3,
        "packages": {
            "node_modules/remote": {
                "resolved": "https://registry.npmjs.org/remote/-/remote-1.0.0.tgz"
            }
        },
    }), encoding="utf-8")
    with pytest.raises(RuntimeError, match="lack sha512 integrity"):
        verify_lock_integrities(broken)


def test_lock_regeneration_starts_without_committed_candidate(monkeypatch, tmp_path):
    from agent.transports.pi_assets import verify_npm_lock

    assets = tmp_path / "assets"
    assets.mkdir()
    (assets / "package.json").write_text('{"name":"fixture"}', encoding="utf-8")
    (assets / "package-lock.json").write_text("committed candidate", encoding="utf-8")
    seen = {}

    def fake_run(command, *, cwd, **kwargs):
        seen["candidate_exists"] = (cwd / "package-lock.json").exists()
        (cwd / "package-lock.json").write_text(
            '{"lockfileVersion":3,"packages":{}}', encoding="utf-8"
        )
        return type("Result", (), {"returncode": 0, "stderr": ""})()

    monkeypatch.setattr(verify_npm_lock.subprocess, "run", fake_run)
    generated = verify_npm_lock.regenerate(assets, tmp_path / "npm-cli.js")
    assert seen["candidate_exists"] is False
    assert json.loads(generated)["lockfileVersion"] == 3


@pytest.mark.skipif(
    shutil.which("node") is None,
    reason="TypeScript extension regression requires Node.js",
)
@pytest.mark.live_system_guard_bypass
def test_bridge_realpath_mediation_rejects_existing_and_parent_symlink_escape(tmp_path):
    workspace = tmp_path / "workspace"
    outside = tmp_path / "outside"
    workspace.mkdir()
    outside.mkdir()
    secret = outside / "secret"
    secret.write_text("opaque", encoding="utf-8")
    (workspace / "escape").symlink_to(secret)
    (workspace / "linked-parent").symlink_to(outside, target_is_directory=True)
    (workspace / "dangling-leaf").symlink_to(outside / "future-secret")
    (workspace / "dangling-parent").symlink_to(outside / "future-dir", target_is_directory=True)
    (workspace / "chain-b").symlink_to("../outside", target_is_directory=True)
    (workspace / "chain-a").symlink_to("chain-b", target_is_directory=True)
    before = sorted(path.name for path in outside.iterdir())
    script = (
        "import {canonicalInside} from './agent/transports/pi_assets/hermes_bridge.ts';"
        "const denied=['escape','linked-parent/new.txt','dangling-leaf','dangling-parent/new.txt',"
        "'chain-a/new.txt','/pi-config','/pi-sessions/new.jsonl','/proc/self/environ'];"
        "if(denied.some(p=>canonicalInside(p))) process.exit(7);"
        "if(!canonicalInside('new/inside.txt')) process.exit(8);"
    )
    node = shutil.which("node")
    assert node is not None
    result = subprocess.run(
        [node, "--experimental-strip-types", "--input-type=module", "--eval", script],
        cwd=Path.cwd(),
        env={
            **os.environ,
            "HERMES_PI_WORKSPACE": str(workspace),
            "TMPDIR": str(tmp_path),
        },
        text=True, capture_output=True, timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert sorted(path.name for path in outside.iterdir()) == before


@pytest.mark.skipif(
    shutil.which("node") is None or shutil.which("git") is None,
    reason="linked-worktree bridge regression requires Node.js and Git",
)
@pytest.mark.live_system_guard_bypass
def test_bridge_denies_linked_worktree_git_pointer_and_symlink_alias(tmp_path):
    main = tmp_path / "main"
    workspace = tmp_path / "linked"
    main.mkdir()

    def git(cwd: Path, *args: str) -> str:
        result = subprocess.run(
            ["git", *args], cwd=cwd, text=True, capture_output=True, timeout=30,
        )
        assert result.returncode == 0, result.stderr
        return result.stdout

    git(main, "init")
    git(main, "config", "user.email", "bridge@example.invalid")
    git(main, "config", "user.name", "Bridge Test")
    (main / "tracked.txt").write_text("base\n", encoding="utf-8")
    git(main, "add", "tracked.txt")
    git(main, "commit", "-m", "base")
    git(main, "worktree", "add", "-b", "bridge-test", str(workspace))
    assert (workspace / ".git").is_file()
    (workspace / ".github").mkdir()
    (workspace / ".env.example").write_text("SAFE=1\n", encoding="utf-8")
    (workspace / "git-alias").symlink_to(".git")
    (workspace / "nested").mkdir()
    status_before = git(workspace, "status", "--porcelain")

    script = (
        "import {canonicalInside} from './agent/transports/pi_assets/hermes_bridge.ts';"
        "const denied=['.git','.git/config','nested/../.git','git-alias',"
        "'git-alias/config',process.env.HERMES_PI_WORKSPACE+'/.git'];"
        "if(denied.some(p=>canonicalInside(p))) process.exit(17);"
        "const allowed=['.github/config.yml','.env.example','nested/../tracked.txt'];"
        "if(allowed.some(p=>!canonicalInside(p))) process.exit(18);"
    )
    node = shutil.which("node")
    assert node is not None
    result = subprocess.run(
        [node, "--experimental-strip-types", "--input-type=module", "--eval", script],
        cwd=Path.cwd(),
        env={**os.environ, "HERMES_PI_WORKSPACE": str(workspace)},
        text=True,
        capture_output=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert git(workspace, "status", "--porcelain") == status_before
    (workspace / "tracked.txt").write_text("host edit\n", encoding="utf-8")
    assert "host edit" in git(workspace, "diff", "--", "tracked.txt")


def test_managed_runtime_rejects_writable_path_component(monkeypatch, tmp_path):
    root = tmp_path / "0.84.3"
    package = root / "node_modules" / "@earendil-works" / "pi-coding-agent"
    bin_dir = root / "node_modules" / ".bin"
    package.mkdir(parents=True)
    bin_dir.mkdir(parents=True)
    package_json = package / "package.json"
    package_json.write_text(
        json.dumps({"name": "@earendil-works/pi-coding-agent", "version": "0.84.3"}),
        encoding="utf-8",
    )
    executable = bin_dir / "pi"
    executable.write_text("stub", encoding="utf-8")
    cli_artifact = package / "dist" / "bundle" / "cli.js"
    cli_artifact.parent.mkdir(parents=True)
    cli_artifact.write_text("stub", encoding="utf-8")
    extension = root / "config" / "hermes_bridge.ts"
    extension.parent.mkdir()
    extension.write_text("bridge", encoding="utf-8")
    shrinkwrap_path = package / "npm-shrinkwrap.json"
    shrinkwrap_path.write_text(json.dumps({"packages": {
        "": {"version": "0.84.3"}
    }}), encoding="utf-8")
    package_lock = root / "package-lock.json"
    package_lock.write_text(json.dumps({
        "packages": {
            "": {"dependencies": {"@earendil-works/pi-coding-agent": "0.84.3"}},
            "node_modules/@earendil-works/pi-coding-agent": {
                "version": "0.84.3", "integrity": pi_runtime.PINNED_NPM_INTEGRITY,
            },
        }
    }), encoding="utf-8")
    for directory in (
        root, root / "node_modules", root / "node_modules" / "@earendil-works",
        package, package / "dist", package / "dist" / "bundle", bin_dir,
        extension.parent,
    ):
        directory.chmod(0o700)
    for file_path in (
        package_json, executable, cli_artifact, package_lock, shrinkwrap_path, extension,
    ):
        file_path.chmod(0o600)
    monkeypatch.setattr(
        pi_runtime,
        "PINNED_PACKAGE_JSON_SHA256",
        hashlib.sha256(package_json.read_bytes()).hexdigest(),
    )
    monkeypatch.setattr(
        pi_runtime,
        "PINNED_CLI_SHA256",
        hashlib.sha256(cli_artifact.read_bytes()).hexdigest(),
    )
    monkeypatch.setattr(
        pi_runtime,
        "PINNED_SHRINKWRAP_SHA256",
        hashlib.sha256(shrinkwrap_path.read_bytes()).hexdigest(),
    )
    monkeypatch.setattr(
        pi_runtime,
        "PINNED_MANIFEST_SHA256",
        hashlib.sha256(package_lock.read_bytes()).hexdigest(),
    )
    monkeypatch.setattr(
        pi_runtime, "PINNED_BRIDGE_SHA256", hashlib.sha256(extension.read_bytes()).hexdigest(),
    )
    paths = pi_runtime.ManagedPiPaths(
        root=root, executable=executable, package_json=package_json,
        package_lock=package_lock, config_dir=root / "config",
        session_root=root / "sessions", binding_root=root / "bindings",
        home_dir=root / "home",
        extension_file=extension,
    )
    assert pi_runtime.verify_managed_pi(paths, run_version=False) == str(executable)
    (root / "node_modules").chmod(0o770)
    with pytest.raises(RuntimeError, match="group/world writable"):
        pi_runtime.verify_managed_pi(paths, run_version=False)


def test_clean_home_installer_uses_verified_tarball_and_npm_ci(monkeypatch, tmp_path):
    package_json = json.dumps({
        "name": "@earendil-works/pi-coding-agent", "version": "0.84.3",
    }).encode()
    shrinkwrap = json.dumps({"packages": {"": {"version": "0.84.3"}}}).encode()
    cli = b"#!/bin/sh\necho 0.84.3\n"
    tarball = tmp_path / "fixture.tgz"
    with tarfile.open(tarball, "w:gz") as archive:
        for name, content, mode in (
            ("package/package.json", package_json, 0o600),
            ("package/npm-shrinkwrap.json", shrinkwrap, 0o600),
            ("package/dist/bundle/cli.js", cli, 0o700),
        ):
            info = tarfile.TarInfo(name)
            info.size = len(content)
            info.mode = mode
            archive.addfile(info, io.BytesIO(content))
    npm_log = tmp_path / "npm.log"
    fake_npm = tmp_path / "fake-npm"
    fake_npm.write_text(
        "#!/bin/sh\n"
        f"printf '%s\\n' \"$*\" >> '{npm_log}'\n"
        "if [ \"$1\" = pack ]; then\n"
        "  shift\n"
        "  while [ \"$1\" != --pack-destination ]; do shift; done\n"
        "  shift\n"
        f"  cp '{tarball}' \"$1/earendil-works-pi-coding-agent-0.84.3.tgz\"\n"
        "  exit 0\n"
        "fi\n"
        "if [ \"$1\" = ci ]; then\n"
        "  mkdir -p node_modules/@earendil-works/pi-coding-agent node_modules/.bin\n"
        f"  tar -xzf '{tarball}' -C node_modules/@earendil-works/pi-coding-agent --strip-components=1\n"
        "  chmod 700 node_modules/@earendil-works/pi-coding-agent/dist/bundle/cli.js\n"
        "  ln -s ../@earendil-works/pi-coding-agent/dist/bundle/cli.js node_modules/.bin/pi\n"
        "  exit 0\n"
        "fi\n"
        "exit 2\n",
        encoding="utf-8",
    )
    fake_npm.chmod(0o700)
    monkeypatch.setattr(
        "agent.transports.pi_assets.verify_npm_lock.npm_package_root",
        lambda _path: tmp_path,
    )
    monkeypatch.setattr(
        "agent.transports.pi_assets.verify_npm_lock.canonical_tree_hash",
        lambda _path: __import__(
            "agent.transports.pi_assets.verify_npm_lock", fromlist=["NPM_TREE_SHA256"]
        ).NPM_TREE_SHA256,
    )
    monkeypatch.setattr(pi_runtime, "PINNED_TARBALL_SHA512", hashlib.sha512(tarball.read_bytes()).hexdigest())
    monkeypatch.setattr(pi_runtime, "PINNED_PACKAGE_JSON_SHA256", hashlib.sha256(package_json).hexdigest())
    monkeypatch.setattr(pi_runtime, "PINNED_SHRINKWRAP_SHA256", hashlib.sha256(shrinkwrap).hexdigest())
    monkeypatch.setattr(pi_runtime, "PINNED_CLI_SHA256", hashlib.sha256(cli).hexdigest())
    home = tmp_path / "clean-hermes-home"
    paths = pi_runtime.install_managed_pi(
        hermes_home=home, npm_bin=str(fake_npm), force=False
    )
    assert paths.root == home / "runtimes" / "pi" / "0.84.3"
    assert paths.executable.resolve() == paths.package_json.parent / "dist" / "bundle" / "cli.js"
    assert "ci --omit=dev --no-audit --no-fund" in npm_log.read_text(encoding="utf-8")
    assert pi_runtime.verify_managed_pi(paths) == str(paths.executable.resolve())


def test_auth_launcher_uses_only_managed_locations(monkeypatch, tmp_path):
    paths = pi_runtime.managed_pi_paths(hermes_home=tmp_path, provision=True)
    paths.executable.parent.mkdir(parents=True, exist_ok=True)
    paths.executable.write_text("stub", encoding="utf-8")
    monkeypatch.setattr(pi_runtime, "verify_managed_pi", lambda _paths: str(paths.executable))
    monkeypatch.setenv("HERMES_GATEWAY_TOKEN", "must-not-pass")
    monkeypatch.setenv("CODEX_HOME", "/must-not-pass")
    seen = {}
    monkeypatch.setattr(
        pi_runtime.subprocess, "call",
        lambda argv, **kwargs: seen.update(argv=argv, **kwargs) or 0,
    )
    with pytest.raises(RuntimeError, match="explicitly acknowledging"):
        pi_runtime.launch_managed_pi_auth(hermes_home=tmp_path)
    assert pi_runtime.launch_managed_pi_auth(
        hermes_home=tmp_path, acknowledge_trusted_local=True,
    ) == 0
    assert seen["env"]["HOME"] == str(paths.home_dir)
    assert seen["env"]["PI_CODING_AGENT_DIR"] == str(paths.config_dir)
    assert "HERMES_GATEWAY_TOKEN" not in seen["env"]
    assert "CODEX_HOME" not in seen["env"]
    assert "--no-session" in seen["argv"]


def test_built_wheel_contains_and_resolves_all_pi_runtime_assets(tmp_path):
    uv = shutil.which("uv")
    assert uv is not None, "artifact packaging tests require the project's uv build tool"
    wheels = []
    for build_number in (1, 2):
        wheel_dir = tmp_path / f"wheel-{build_number}"
        wheel_dir.mkdir()
        source = tmp_path / f"source-{build_number}"
        shutil.copytree(
            Path.cwd(),
            source,
            ignore=shutil.ignore_patterns(
                ".git", ".venv", "venv", "node_modules", "build", "dist",
                "__pycache__", ".pytest_cache", ".ruff_cache", ".ty_cache",
                "test_durations.json", "*.tsbuildinfo", "*.whl",
            ),
        )
        proc = subprocess.run(
            [uv, "build", "--wheel", "--no-build-isolation", "--offline",
             "--python", sys.executable, "--out-dir", str(wheel_dir), str(source)],
            env={
                **os.environ,
                "HERMES_NIX_BUILD": "1",
                "SOURCE_DATE_EPOCH": "1704067200",
                "UV_CACHE_DIR": str(tmp_path / "uv-cache"),
            },
            text=True, capture_output=True, timeout=300,
        )
        assert proc.returncode == 0, proc.stderr
        wheels.append(next(wheel_dir.glob("hermes_agent-*.whl")))
    assert wheels[0].read_bytes() == wheels[1].read_bytes()
    assert hashlib.sha256(wheels[0].read_bytes()).hexdigest() == hashlib.sha256(
        wheels[1].read_bytes()
    ).hexdigest()
    wheel = wheels[0]
    required = {
        "agent/transports/pi_assets/Dockerfile",
        "agent/transports/pi_assets/build-image.sh",
        "agent/transports/pi_assets/hermes_bridge.ts",
        "agent/transports/pi_assets/egress_proxy.py",
        "agent/transports/pi_assets/package.json",
        "agent/transports/pi_assets/package-lock.json",
        "agent/transports/pi_assets/runtime-integrity.json",
        "agent/transports/pi_assets/verification-requirements.in",
        "agent/transports/pi_assets/verification-requirements.txt",
        "agent/transports/pi_assets/verify_npm_lock.py",
    }
    with zipfile.ZipFile(wheel) as archive:
        assert required <= set(archive.namelist())
        record_name = next(name for name in archive.namelist() if name.endswith(".dist-info/RECORD"))
        record = {
            row[0]: row[1]
            for row in csv.reader(io.StringIO(archive.read(record_name).decode("utf-8")))
        }
        for name in required:
            algorithm, encoded = record[name].split("=", 1)
            assert algorithm == "sha256"
            actual = base64.urlsafe_b64encode(
                hashlib.sha256(archive.read(name)).digest()
            ).decode("ascii").rstrip("=")
            assert actual == encoded
        helper_info = archive.getinfo("agent/transports/pi_assets/build-image.sh")
        assert (helper_info.external_attr >> 16) & 0o111
        helper = tmp_path / "build-image.sh"
        helper.write_bytes(archive.read("agent/transports/pi_assets/build-image.sh"))
        extracted = tmp_path / "wheel-extracted"
        archive.extractall(extracted)
    verify_runtime_bundle(extracted / "agent" / "transports" / "pi_assets")
    assert subprocess.run(["sh", "-n", str(helper)], check=False).returncode == 0
    target = tmp_path / "installed"
    python_311 = shutil.which("python3.11")
    assert python_311 is not None, "Pi wheel probe requires an explicit Python 3.11"
    install_argv = [
        uv, "pip", "install", "--python", python_311,
        "--offline", "--no-deps", "--target", str(target), str(wheel),
    ]
    assert install_argv[3:5] == ["--python", python_311]
    install = subprocess.run(
        install_argv,
        env={**os.environ, "UV_CACHE_DIR": str(tmp_path / "uv-cache")},
        text=True, capture_output=True, timeout=300,
    )
    assert install.returncode == 0, install.stderr
    installed_helper = target / "agent" / "transports" / "pi_assets" / "build-image.sh"
    verify_runtime_bundle(target / "agent" / "transports" / "pi_assets")
    assert installed_helper.stat().st_mode & 0o111
    assert subprocess.run(["sh", "-n", str(installed_helper)], check=False).returncode == 0
    fake_docker = tmp_path / "fake-docker"
    fake_docker_log = tmp_path / "fake-docker.log"
    fake_docker.write_text(
        "#!/bin/sh\n"
        "if [ \"$1 $2\" = 'buildx build' ]; then\n"
        "  printf '%s|%s\\n' \"${SOURCE_DATE_EPOCH:-}\" \"$*\" > \"$FAKE_DOCKER_LOG\"\n"
        "  for argument in \"$@\"; do\n"
        "    case \"$argument\" in\n"
        "      type=docker,dest=*,rewrite-timestamp=true)\n"
        "        archive=${argument#type=docker,dest=}\n"
        "        archive=${archive%,rewrite-timestamp=true}\n"
        "        printf 'archive\\n' > \"$archive\"\n"
        "        ;;\n"
        "    esac\n"
        "  done\n"
        "  exit 0\n"
        "fi\n"
        "if [ \"$1\" = load ]; then exit 0; fi\n"
        "if [ \"$1 $2\" = 'image inspect' ]; then\n"
        "  case \"$3\" in\n"
        "    --format) case \"$4\" in *org.hermes.pi.version*) echo 0.84.3;; *) echo sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa;; esac;;\n"
        "    *) echo sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa;;\n"
        "  esac\n"
        "  exit 0\n"
        "fi\n"
        "if [ \"$1\" = run ]; then echo 0.84.3; exit 0; fi\n"
        "exit 2\n",
        encoding="utf-8",
    )
    fake_docker.chmod(0o700)
    helper_run = subprocess.run(
        [str(installed_helper), str(fake_docker), "test-pi:0.84.3"],
        env={**os.environ, "FAKE_DOCKER_LOG": str(fake_docker_log)},
        text=True, capture_output=True, timeout=30,
    )
    assert helper_run.returncode == 0, helper_run.stderr
    assert "Configure this immutable local image ID" in helper_run.stdout
    build_invocation = fake_docker_log.read_text(encoding="utf-8")
    assert build_invocation.startswith("1704067200|buildx build ")
    assert " --no-cache " in build_invocation
    assert " --platform linux/arm64 " in build_invocation
    assert " --build-arg SOURCE_DATE_EPOCH=1704067200 " in build_invocation
    assert " --provenance=false " in build_invocation
    assert " --output type=docker,dest=" in build_invocation
    assert ",rewrite-timestamp=true " in build_invocation
    probe = subprocess.run(
        [sys.executable, "-c", (
            "from pathlib import Path; import agent.pi_runtime as p; "
            "a=Path(p.__file__).with_name('transports')/'pi_assets'; "
            "assert (a/'egress_proxy.py').is_file(); assert (a/'runtime-integrity.json').is_file()"
        )],
        env={**os.environ, "PYTHONPATH": str(target)}, text=True, capture_output=True,
        cwd=tmp_path,
    )
    assert probe.returncode == 0, probe.stderr


def test_interactive_pi_is_contained_read_only_with_host_only_binding(
    monkeypatch, tmp_path
):
    root = tmp_path / "runtime"
    config = root / "managed-config"
    sessions = root / "sessions"
    bindings = root / "bindings"
    home = root / "home"
    work = tmp_path / "work"
    for directory in (config, sessions, bindings, home, work):
        directory.mkdir(parents=True, exist_ok=True)
    extension = config / "hermes_bridge.ts"
    extension.write_text("stub", encoding="utf-8")
    paths = SimpleNamespace(
        root=root, executable=root / "pi", package_json=root / "package.json",
        package_lock=root / "package-lock.json", config_dir=config,
        session_root=sessions, binding_root=bindings, home_dir=home,
        extension_file=extension,
    )
    seen = {}

    class Sandbox:
        def __init__(self, **kwargs):
            seen["sandbox"] = self
            self.kwargs = kwargs

        def verify_version(self, **kwargs):
            seen["verified"] = True

        def cleanup(self, **kwargs):
            seen["cleaned"] = True

    class Session:
        def __init__(self, **kwargs):
            seen["session"] = self
            self.kwargs = kwargs

    monkeypatch.setattr(pi_runtime, "managed_pi_paths", lambda provision=True: paths)
    def verify_host(paths, run_version=True):
        seen["host_run_version"] = run_version
        return "host-pi"

    monkeypatch.setattr(pi_runtime, "verify_managed_pi", verify_host)
    monkeypatch.setattr(pi_runtime, "PiDockerSandbox", Sandbox)
    monkeypatch.setattr(pi_runtime, "PiRpcSession", Session)
    monkeypatch.setattr("hermes_cli.config.load_config", lambda: {
        "agent": {"pi_rpc": {
            "enabled": True,
            "docker_image": "sha256:" + "b" * 64,
            "interactive_mode": "inspect",
        }}
    })
    agent = SimpleNamespace(session_id="conversation", session_cwd=str(work))
    session = pi_runtime.get_or_create_pi_session(agent)
    sandbox_kwargs = seen["sandbox"].kwargs
    session_kwargs = getattr(session, "kwargs")
    assert seen["verified"] is True
    assert seen["host_run_version"] is False
    assert sandbox_kwargs["read_only_workspace"] is True
    assert sandbox_kwargs["worktree"] == str(work.resolve())
    assert session_kwargs["executable"] is None
    assert session_kwargs["sandbox"] is seen["sandbox"]
    binding_path = Path(session_kwargs["binding_path"])
    assert binding_path.is_relative_to(bindings)
    for mounted in (work, config, Path(session_kwargs["session_dir"])):
        assert not binding_path.is_relative_to(mounted)


def test_interactive_pi_fails_closed_when_exact_image_is_unavailable(monkeypatch, tmp_path):
    root = tmp_path / "runtime"
    config = root / "config"
    sessions = root / "sessions"
    bindings = root / "bindings"
    home = root / "home"
    work = tmp_path / "work"
    for directory in (config, sessions, bindings, home, work):
        directory.mkdir(parents=True, exist_ok=True)
    extension = config / "hermes_bridge.ts"
    extension.write_text("stub", encoding="utf-8")
    paths = SimpleNamespace(
        root=root, config_dir=config, session_root=sessions, binding_root=bindings,
        home_dir=home,
        extension_file=extension,
    )

    class MissingImageSandbox:
        def __init__(self, **kwargs):
            pass

        def verify_version(self, **kwargs):
            raise PiSandboxError("configured exact image is unavailable")

        def cleanup(self, **kwargs):
            pass

    monkeypatch.setattr(pi_runtime, "managed_pi_paths", lambda provision=True: paths)
    monkeypatch.setattr(
        pi_runtime, "verify_managed_pi",
        lambda paths, run_version=True: "host-pi",
    )
    monkeypatch.setattr(pi_runtime, "PiDockerSandbox", MissingImageSandbox)
    monkeypatch.setattr("hermes_cli.config.load_config", lambda: {
        "agent": {"pi_rpc": {
            "enabled": True, "docker_image": "sha256:" + "c" * 64,
        }}
    })
    with pytest.raises(PiSandboxError, match="unavailable"):
        pi_runtime.get_or_create_pi_session(
            SimpleNamespace(session_id="conversation", session_cwd=str(work))
        )


@pytest.mark.parametrize("workspace_case", ["ancestor", "symlink"])
def test_public_interactive_startup_rejects_control_overlap_before_provision_or_docker(
    monkeypatch, tmp_path, workspace_case
):
    root = tmp_path / "hermes" / "runtimes" / "pi" / pi_runtime.PINNED_PI_VERSION
    config = root / "managed-config"
    sessions = root / "sessions"
    bindings = root / "bindings"
    home = root / "home"
    for directory in (config, sessions, bindings, home):
        directory.mkdir(parents=True, exist_ok=True)
    extension = config / "hermes_bridge.ts"
    extension.write_text("stub", encoding="utf-8")
    paths = SimpleNamespace(
        root=root,
        config_dir=config,
        session_root=sessions,
        binding_root=bindings,
        home_dir=home,
        extension_file=extension,
    )
    if workspace_case == "ancestor":
        workspace = tmp_path / "hermes"
    else:
        workspace = tmp_path / "workspace-link"
        workspace.symlink_to(config, target_is_directory=True)
    provisions = []
    monkeypatch.setattr(
        pi_runtime,
        "managed_pi_paths",
        lambda *, provision=False: provisions.append(provision) or paths,
    )
    monkeypatch.setattr(
        pi_runtime,
        "verify_managed_pi",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("overlap must fail before managed-runtime verification")
        ),
    )
    monkeypatch.setattr(
        pi_runtime,
        "PiDockerSandbox",
        lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("overlap must fail before sandbox allocation")
        ),
    )
    monkeypatch.setattr("hermes_cli.config.load_config", lambda: {
        "agent": {"pi_rpc": {
            "enabled": True,
            "docker_image": "sha256:" + "c" * 64,
        }}
    })

    with pytest.raises(PiSandboxError, match="overlaps"):
        pi_runtime.get_or_create_pi_session(
            SimpleNamespace(session_id="conversation", session_cwd=str(workspace))
        )
    assert provisions == [False]


def test_shared_model_selection_is_exact_and_synchronizes_agent(monkeypatch):
    class Session:
        def available_models(self):
            return [
                {"provider": "one", "id": "shared"},
                {"provider": "two", "id": "shared"},
            ]

        def set_model(self, provider, model_id):
            return SimpleNamespace(
                model_provider=provider, model_id=model_id, thinking_level="high",
            )

    agent = SimpleNamespace(provider="old", model="old")
    monkeypatch.setattr(pi_runtime, "get_or_create_pi_session", lambda _agent: Session())
    state = pi_runtime.select_pi_model(agent, "two", "shared")
    assert (state.model_provider, state.model_id) == ("two", "shared")
    assert (agent.provider, agent.model) == ("two", "shared")
    with pytest.raises(ValueError, match="exactly match"):
        pi_runtime.select_pi_model(agent, "missing", "shared")


@pytest.mark.parametrize(
    ("target", "provider", "expected"),
    [
        ("openai/gpt-test", "", ("openai", "gpt-test")),
        (
            "openrouter/anthropic/claude-sonnet",
            "",
            ("openrouter", "anthropic/claude-sonnet"),
        ),
        ("gpt-test", "openai", ("openai", "gpt-test")),
        (
            "anthropic/claude-sonnet",
            "openrouter",
            ("openrouter", "anthropic/claude-sonnet"),
        ),
    ],
)
def test_shared_pi_model_parser_accepts_both_documented_forms(target, provider, expected):
    assert pi_runtime.parse_pi_model_selection(target, provider) == expected


@pytest.mark.parametrize("target", ["", "gpt-test", "/gpt", "openai/"])
def test_shared_pi_model_parser_rejects_ambiguous_targets_with_guidance(target):
    with pytest.raises(ValueError, match="provider/model"):
        pi_runtime.parse_pi_model_selection(target)


def test_cli_pi_model_command_preserves_nested_model_id(monkeypatch):
    import cli as cli_mod

    class Session:
        def ensure_started(self):
            return None

        def available_models(self):
            return [
                {"provider": "openrouter", "id": "anthropic/claude-sonnet"}
            ]

        def set_model(self, provider, model_id):
            return SimpleNamespace(
                model_provider=provider,
                model_id=model_id,
                thinking_level="high",
            )

    native = Session()
    agent = SimpleNamespace(api_mode="pi_rpc", provider="old", model="old")
    output = []
    monkeypatch.setattr(pi_runtime, "get_or_create_pi_session", lambda _agent: native)
    monkeypatch.setattr(cli_mod, "_cprint", output.append)

    cli_mod.HermesCLI._handle_model_switch(
        SimpleNamespace(agent=agent),
        "/model openrouter/anthropic/claude-sonnet --session",
    )

    assert (agent.provider, agent.model) == (
        "openrouter",
        "anthropic/claude-sonnet",
    )
    assert any("runtime=pi" in line for line in output)


def test_inspect_reconciliation_never_mutates_agent_metadata(monkeypatch):
    class Session:
        def reconcile(self, action):
            assert action == "inspect"
            return {
                "action": "inspect",
                "model": {"provider": "native", "id": "observed"},
                "replayed": False,
            }

    agent = SimpleNamespace(
        api_mode="pi_rpc", provider="hermes-before", model="model-before",
    )
    monkeypatch.setattr(pi_runtime, "get_or_create_pi_session", lambda _agent: Session())
    report = pi_runtime.reconcile_pi_agent(agent, "inspect")
    assert report["model"] == {"provider": "native", "id": "observed"}
    assert (agent.provider, agent.model) == ("hermes-before", "model-before")


def test_reconciliation_messages_make_advanced_cursor_context_loss_explicit():
    inspected = pi_runtime.format_pi_reconciliation_report({
        "action": "inspect",
        "prompt_state": "reconciliation_required",
        "previous_cursor": "before",
        "cursor": "after",
        "cursor_advanced": True,
    })
    assert "Adopt unavailable" in inspected
    assert "archiv" in inspected
    assert "loses native continuity" in inspected
    assert "independent approval" in inspected

    abandoned = pi_runtime.format_pi_reconciliation_report({
        "action": "abandon",
        "prompt_state": "abandoned",
        "previous_cursor": "before",
        "cursor": "before",
    })
    assert "CONTEXT LOSS" in abandoned
    assert "will not resume" in abandoned
    assert "independent approval" in abandoned


def test_interactive_abandon_after_retirement_is_host_control_plane_only(
    monkeypatch, tmp_path
):
    session_id = "conversation"
    slug = hashlib.sha256(session_id.encode()).hexdigest()[:32]
    root = tmp_path / "runtime"
    sessions = root / "sessions" / slug
    binding_dir = root / "bindings" / "interactive" / slug
    sessions.mkdir(parents=True)
    binding_dir.mkdir(parents=True)
    binding_path = binding_dir / "binding.json"
    save_binding(
        binding_path,
        RuntimeBinding(
            runtime="pi_rpc",
            native_session_file=str(sessions / "missing-after-retirement.jsonl"),
            native_session_id="retired-native",
            last_entry_id="leaf",
            cwd=str(tmp_path.resolve()),
            runtime_version=pi_runtime.PINNED_PI_VERSION,
            worktree=worktree_identity(tmp_path),
            prompt_state="reconciliation_required",
        ),
    )
    paths = SimpleNamespace(
        session_root=root / "sessions", binding_root=root / "bindings",
    )
    provisions = []
    monkeypatch.setattr(
        pi_runtime,
        "managed_pi_paths",
        lambda *, provision=False, **_kwargs: provisions.append(provision) or paths,
    )
    monkeypatch.setattr(
        pi_runtime,
        "get_or_create_pi_session",
        lambda _agent: (_ for _ in ()).throw(AssertionError("must not provision or inspect Docker")),
    )
    agent = SimpleNamespace(
        api_mode="pi_rpc", session_id=session_id, session_cwd=str(tmp_path),
        _pi_rpc_session=None,
    )
    report = pi_runtime.reconcile_pi_agent(agent, "abandon")
    assert report["slot_bound"] is False
    assert provisions == [False]
    assert not binding_path.exists()


def test_interactive_abandon_archives_recoverable_native_context(monkeypatch, tmp_path):
    session_id = "conversation"
    slug = hashlib.sha256(session_id.encode()).hexdigest()[:32]
    root = tmp_path / "runtime"
    sessions = root / "sessions" / slug
    binding_dir = root / "bindings" / "interactive" / slug
    sessions.mkdir(parents=True)
    binding_dir.mkdir(parents=True)
    native = sessions / "recoverable.jsonl"
    native.write_text("{}\n", encoding="utf-8")
    binding_path = binding_dir / "binding.json"
    save_binding(
        binding_path,
        RuntimeBinding(
            runtime="pi_rpc",
            native_session_file=str(native),
            native_session_id="recoverable-native",
            last_entry_id="leaf",
            cwd=str(tmp_path.resolve()),
            runtime_version=pi_runtime.PINNED_PI_VERSION,
            worktree=worktree_identity(tmp_path),
            prompt_state="reconciliation_required",
        ),
    )
    paths = SimpleNamespace(
        session_root=root / "sessions", binding_root=root / "bindings",
    )
    monkeypatch.setattr(
        pi_runtime, "managed_pi_paths", lambda *, provision=False, **_kwargs: paths,
    )
    agent = SimpleNamespace(
        api_mode="pi_rpc", session_id=session_id, session_cwd=str(tmp_path),
        _pi_rpc_session=None,
    )
    original = binding_path.read_bytes()

    report = pi_runtime.reconcile_pi_agent(agent, "abandon")
    archive = binding_path.with_name(report["archived_binding"])

    assert not binding_path.exists()
    assert archive.read_bytes() == original
    assert report["context_loss"] is True
    assert report["requires_independent_approval"] is True
    assert report["replayed"] is False
