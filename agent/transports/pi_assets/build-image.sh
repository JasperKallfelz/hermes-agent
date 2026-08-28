#!/bin/sh
set -eu

docker_bin=${1:-docker}
image_tag=${2:-hermes-pi:0.84.3}
script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
source_date_epoch=1704067200
target_platform=linux/arm64

# A docker exporter with timestamp rewriting cannot also use the docker
# driver's implicit unpack path. Export to a private archive, load it through
# the caller-selected CLI, and discard it on success, error, or interruption.
# The load command's output is informational and is never an image identity.
umask 077
archive_dir=$(mktemp -d "${TMPDIR:-/tmp}/hermes-pi-image.XXXXXX")
image_archive=$archive_dir/runtime-image.tar
cleanup_archive() {
  rm -f -- "$image_archive"
  rmdir -- "$archive_dir" 2>/dev/null || true
}
trap cleanup_archive 0
trap 'exit 129' HUP
trap 'exit 130' INT
trap 'exit 143' TERM
chmod 0700 "$archive_dir"

# The runtime ID is a release contract: identical reviewed inputs must produce
# the same local image ID. BuildKit rewrites every exported layer timestamp to
# this fixed epoch, while the build arg exposes the same value to npm/uv and
# any other package step. Provenance/SBOM evidence is generated separately in
# post-merge CI so timestamped attestations never change this runtime ID.
SOURCE_DATE_EPOCH=$source_date_epoch DOCKER_BUILDKIT=1 "$docker_bin" buildx build \
  --pull=false \
  --no-cache \
  --platform "$target_platform" \
  --build-arg "SOURCE_DATE_EPOCH=$source_date_epoch" \
  --provenance=false \
  --output "type=docker,dest=$image_archive,rewrite-timestamp=true" \
  --tag "$image_tag" \
  --file "$script_dir/Dockerfile" \
  "$script_dir"

# BuildKit creates this file under the private directory; pin its permissions
# before handing it to Docker and reject missing/empty exporter output.
test -f "$image_archive"
test -s "$image_archive"
chmod 0600 "$image_archive"
"$docker_bin" load --input "$image_archive" >/dev/null

# Resolve the caller's exact requested tag after loading, then use only that
# immutable ID for every subsequent inspection and execution. In particular,
# do not parse or trust implementation-specific `docker load` output.
image_id=$("$docker_bin" image inspect --format '{{.Id}}' "$image_tag")
case "$image_id" in
  sha256:????????????????????????????????????????????????????????????????) ;;
  *)
    echo "Docker returned a non-immutable image ID: $image_id" >&2
    exit 1
    ;;
esac

declared_version=$(
  "$docker_bin" image inspect \
    --format '{{index .Config.Labels "org.hermes.pi.version"}}' \
    "$image_id"
)
if [ "$declared_version" != "0.84.3" ]; then
  echo "Built image has the wrong org.hermes.pi.version label" >&2
  exit 1
fi

reported_version=$(
  "$docker_bin" run --rm --network none --entrypoint pi "$image_id" --version
)
if [ "$reported_version" != "0.84.3" ]; then
  echo "Built image executed an unexpected Pi version" >&2
  exit 1
fi

echo "Built and inspected $image_tag"
echo "Configure this immutable local image ID: $image_id"
printf '%s\n' "$image_id"
