#!/bin/sh
set -eu

repo_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$repo_root"

if [ "${1:-}" = "--docker" ]; then
  image=${HERMES_TEST_PI_IMAGE:-}
  digest=${image#sha256:}
  if [ "$image" = "$digest" ] || [ "${#digest}" -ne 64 ]; then
    echo "HERMES_TEST_PI_IMAGE must be an immutable sha256 image ID" >&2
    exit 2
  fi
  case "$digest" in
    *[!0-9a-f]*)
      echo "HERMES_TEST_PI_IMAGE must be an immutable sha256 image ID" >&2
      exit 2
      ;;
  esac
  HERMES_PI_EXPLICIT_DOCKER_GATE=1 exec scripts/run_tests.sh \
    --file-retries 0 --require-no-skips \
    tests/agent/transports/test_pi_docker_egress_e2e.py
fi

# This Windows-only assertion is exercised by the required
# pi-runtime.yml/windows-path-parser job with this same zero-retry/no-skip
# contract.  Classifier/workflow tests make that job and its aggregation
# load-bearing.  It is explicitly deselected here so the non-Windows release
# gate stays a genuine no-skips signal rather than accepting a platform skip.
scripts/run_tests.sh \
  --file-retries 0 --require-no-skips \
  --deselect=tests/test_run_tests_parallel.py::test_drive_letter_colon_is_not_a_path_separator \
  tests/acp/test_session.py \
  tests/acp_adapter/test_acp_commands.py \
  tests/agent/transports/test_pi_egress_proxy.py \
  tests/agent/transports/test_pi_event_projector.py \
  tests/agent/transports/test_pi_rpc_client.py \
  tests/agent/transports/test_pi_rpc_session.py \
  tests/ci/test_classify_changes.py \
  tests/cli/test_cli_steer_busy_path.py \
  tests/cron/test_run_one_job.py \
  tests/gateway/test_busy_session_ack.py \
  tests/gateway/test_session_model_override_routing.py \
  tests/hermes_cli/test_commands.py \
  tests/hermes_cli/test_model_switch_parsing.py \
  tests/hermes_cli/test_pi_runtime_doctor.py \
  tests/hermes_cli/test_runtime_provider_resolution.py \
  tests/hermes_cli/test_runtime_switch.py \
  tests/run_agent/test_pi_rpc_integration.py \
  tests/run_agent/test_background_review_cost_controls.py \
  tests/test_run_tests_parallel.py \
  tests/test_tui_gateway_queue_on_busy.py \
  tests/test_tui_gateway_server.py \
  tests/tui_gateway/test_compute_host_phase1.py \
  tests/tools/test_coding_task_tool.py \
  tests/tui_gateway/test_pi_runtime_surface.py \
  tests/agent/test_codex_app_server_event_bridge.py \
  tests/agent/test_codex_app_server_persist.py \
  tests/agent/test_codex_runtime_live_events.py \
  tests/agent/transports/test_codex_app_server_runtime.py \
  tests/agent/transports/test_codex_app_server_session.py \
  tests/hermes_cli/test_codex_runtime_plugin_migration.py \
  tests/hermes_cli/test_codex_runtime_switch.py \
  tests/run_agent/test_codex_app_server_compaction.py \
  tests/run_agent/test_codex_app_server_integration.py \
  tests/tui_gateway/test_codex_app_server_live_events.py
