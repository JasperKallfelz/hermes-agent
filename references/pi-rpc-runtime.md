# Pi RPC coding runtime (experimental)

Hermes can use pinned Pi Coding Agent 0.84.3 as an external coding engine.
Hermes remains responsible for runtime selection, worktree assignment,
approvals, visible accounting, and independent delivery verification. Pi owns
its native coding conversation, retry, tree, and compaction.

The runtime is off by default. Opt into the experiment in `config.yaml` before
selecting it:

```yaml
agent:
  pi_rpc:
    enabled: true
    # Required for every Pi session. Tags are rejected. Use either a registry
    # digest or the exact local image ID printed by build-image.sh.
    docker_image: sha256:<64 lowercase hex digits>
    # Credential *names* only. Omit this list when managed Pi auth is enough.
    provider_env_allowlist: []
    # Exact endpoints for the selected native provider, optionally with a
    # fixed port. URLs, IP literals, and wildcards are rejected. There is no
    # broad default union; an empty list fails closed before a model turn.
    egress_allowlist: [api.openai.com:443]
```

Then use `/runtime pi` and start a new session. `/model` queries Pi's RPC
catalog in that live Pi session; Pi is intentionally not listed as a Hermes
model provider. Disable with `/runtime hermes` (or the legacy
`/codex-runtime auto`) and start another new session.

## Build the reviewed Pi image

Install the separately managed host runtime first. This downloads the exact
reviewed npm tarball, verifies its SHA-512, installs its upstream pinned
dependency graph through Hermes' reviewed `package-lock.json` with `npm ci`,
then verifies the upstream shrinkwrap/package/CLI/bridge hashes,
and writes only below the active `$HERMES_HOME`:

```bash
hermes pi-runtime install
hermes pi-runtime auth --acknowledge-trusted-local
```

Pi 0.84.3 cannot perform its interactive authentication inside the normal
contained RPC profile. Authentication is therefore a separate elevated
trusted-local operation, never auto-run, and requires the explicit
`--acknowledge-trusted-local` flag. It uses a sanitized environment and managed
HOME/config/session directories, but the third-party CLI runs on the host; do
not describe this step as contained Pi execution.

The image definition is shipped at
`agent/transports/pi_assets/Dockerfile`. Its Node and uv base stages are pinned
by digest, Pi is pinned to 0.84.3, and the image label
`org.hermes.pi.version=0.84.3` is checked before launch. From a source checkout,
the exact build and inspection workflow is:

```bash
agent/transports/pi_assets/build-image.sh docker hermes-pi:0.84.3
docker image inspect --format '{{.Id}}' hermes-pi:0.84.3
```

The helper builds the packaged Dockerfile, inspects the label, executes
`pi --version` without networking, and prints the immutable local
`sha256:<64hex>` ID to place in `agent.pi_rpc.docker_image`. A local tag such as
`hermes-pi:0.84.3` is deliberately not accepted at runtime. A pushed image may
instead be configured as `repository@sha256:<64hex>` when that exact value is
present in Docker's `RepoDigests` inspection output.

Local and release builds use no cache, a fixed `SOURCE_DATE_EPOCH=1704067200`,
BuildKit layer-timestamp rewriting, and `--provenance=false`. This makes the
runtime image ID a deterministic function of the reviewed inputs instead of a
function of the build clock. On post-merge CI, the SBOM and source provenance
are generated after the two-image equality gate and uploaded as a separate
artifact; timestamped attestations are never children of the runtime image.

The build verifies the same complete Pi tarball, then uses the packaged
production-only `package-lock.json` with pinned npm 11 and `npm ci --omit=dev`.
It verifies the manifest, upstream shrinkwrap, package, CLI, proxy, bridge, and
runtime version hashes. The image includes Node/npm, Python 3, pip/venv, build essentials, curl,
CA certificates, and uv. Updating a base digest, Pi version, package label, or
toolchain is a security-review event.

The managed host executable is installed at
`$HERMES_HOME/runtimes/pi/0.84.3/node_modules/.bin/pi`. Hermes never reuses
Codex credentials. Project extensions, skills, settings, and context files are
disabled in managed runs. Provider credentials are passed only by an explicit
allowlist; gateway and messaging secrets are never inherited.

Every interactive Pi session requires a reachable, immutable-reference Docker
image reporting Pi 0.84.3. Interactive modes are limited to `review`, `inspect`,
and `plan`; `/workspace` is always mounted read-only. `implement` is exclusive
to the one-shot `coding_task`, which additionally requires an isolated linked
Git worktree and mounts only that assigned worktree read-write.

Model-bearing containers attach only to a per-session Docker network verified
as `Internal=true`; mandatory HTTP(S) proxy variables point to a separate
allowlisting sidecar. The sidecar uses the same immutable reviewed image, has
no host mounts or Docker socket, and alone also attaches to the outbound bridge.
Direct internet, LAN, metadata, and host routes from Pi are unavailable.
Provider destinations require exact hostname opt-in; broad schemes, IP
literals, and wildcards fail closed. HTTPS CONNECT is relayed only after the
first TLS ClientHello contains an exact matching SNI. Missing/malformed SNI,
non-TLS payloads, mismatches, and ECH fail closed. This is hostname containment,
not full L7 policy: the proxy does not terminate TLS and therefore cannot
inspect encrypted HTTP paths, headers, or request bodies.

Both paths use a read-only container root, bounded resources, dropped
capabilities, no privilege escalation, no Docker socket, and exactly three bind
mounts: assigned workspace (`ro` interactive/read-only, `rw` implement), Pi
config (`ro`), and the native Pi session/brief directory (`rw`). Hermes'
`binding.json` is stored under the separate mode-0700 host `bindings/` directory
and is never inside any container mount, so Pi cannot change Hermes' acceptance,
cursor, worktree, or resume decision. The TypeScript extension is policy
mediation, not containment. Unattended approval requests fail closed. A worker's
successful response is evidence, not approval: inspect the diff and rerun every
requested check independently. At least one host verification command is
required before a coding task can return `completed`; more than 20 are rejected,
and none are silently omitted from a verdict. Model-issued `bash` and `powershell`
are disabled in every Pi tool set; only structured realpath-mediated file tools
remain. Verification uses a separate no-network container with only the
read-only worktree mount and fixed reviewed-image executables. Git is not
available to Pi or verification in linked worktrees: Hermes supplies status and
diff evidence and performs Git operations outside the container. The host Git
common directory is never mounted.

Docker credential exposure is intentionally narrow. The host Docker CLI may
consult its normal Docker config while pulling the pinned base during an
operator-initiated build, but Hermes never mounts Docker's config directory or
socket into Pi. At runtime only the three mounts above exist. Pi provider
environment variables are forwarded by name only through
`provider_env_allowlist`; omit it to forward none. Do not put Docker auth,
gateway secrets, or unrelated credentials in the Pi config/session directories.

Recovery never automatically repeats a prompt whose acceptance is uncertain.
Use `/runtime reconcile inspect` to query a redacted cursor/entry summary from
the exact bound native session. `/runtime reconcile adopt` is available only
when the native cursor did **not** advance beyond Hermes' canonical checkpoint;
it never projects or invents native transcript entries. If the cursor advanced,
exact canonical transcript recovery is not available in this release and adopt
fails closed. The recovery action is an explicit `/runtime reconcile abandon`:
it archives the complete prior binding/session metadata, retires old runtime
ownership, never replays the uncertain prompt, and leaves the slot unbound. The
next run allocates a fresh native session and loses native conversation
continuity. This archival context loss is surfaced as a warning and always
requires independent approval. The `coding_task`
tool exposes the same no-prompt operations through `action=inspect|adopt|abandon`
with the original `resume_id`. Inspect/adopt revalidate the session file, ID,
exact worktree identity, native activity/queue state, cursor lineage, and Pi
version. Abandon intentionally validates only host-side binding/worktree
ownership, so missing sessions or an unavailable Docker daemon cannot wedge
recovery. None replay the prior prompt. A completed `coding_task` is marked
settled only after a host-only durable record binds its exact native result to
evidence for every requested independent verification command; failure to land
that record returns reconciliation-required instead of completed.
Removing `agent.runtime: pi_rpc` (or
using `/runtime hermes`) is the complete rollback; native Hermes and Codex
session data are unaffected.

The authenticated provider/model E2E and the real-Docker egress test remain
operator-gated because they require external credentials and Docker access.
Run the latter with `HERMES_TEST_PI_IMAGE=sha256:<reviewed-id>` (or an exact
repository digest) against `tests/agent/transports/test_pi_docker_egress_e2e.py`.
