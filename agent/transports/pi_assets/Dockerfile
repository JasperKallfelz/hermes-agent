# Hermes-reviewed runtime for Pi RPC 0.84.3.
# Keep both upstream stages immutable: updating either digest is a review event.
ARG SOURCE_DATE_EPOCH=1704067200
FROM node:26-bookworm-slim@sha256:9e6f9357d371591e32ab6f2d8a26d63bdd0d17c29eee3f4f3e7e454d9634bf73 AS node_source

FROM ghcr.io/astral-sh/uv:0.11.6-python3.13-trixie@sha256:b3c543b6c4f23a5f2df22866bd7857e5d304b67a564f4feab6ac22044dde719b

# Keep package-generated metadata on the same deterministic clock used by
# BuildKit's layer timestamp rewriter. The build helper always passes the
# reviewed epoch explicitly; this default also makes direct builds stable.
ARG SOURCE_DATE_EPOCH=1704067200
ENV SOURCE_DATE_EPOCH=${SOURCE_DATE_EPOCH} \
    PYTHONDONTWRITEBYTECODE=1

LABEL org.hermes.pi.version="0.84.3" \
      org.hermes.pi.tarball.sha512="62bda9f4fb9badb15999810f608f82f0a9993fdc6516e2c39c01bae11b54d190e0add8976166becbb586c893b93a5a8f9623a454231df48cd5b333b7fedd03d3" \
      org.hermes.pi.package-json.sha256="6c0bfa4558a0de32fb54fa97ca73397753ec7d4ad7d0ba607790920168de751f" \
      org.hermes.pi.cli.sha256="1c3a5094b54aae9ae98c66516ce8c6578140363d081471ca7e91f9cb8c23dc8a" \
      org.hermes.pi.shrinkwrap.sha256="df18751951cef578a1e5d809b33774bb385f69539f197bade0e70f1944aa3de7" \
      org.hermes.pi.manifest.sha256="9fef223236a34bcab3a6926dcc1f3cd1f4d3504e3ce5eb37ac840b5ea0b280f2" \
      org.hermes.pi.egress-proxy.sha256="0d8f3d724b70d141072fc51c42ccd6cc17296a5775d0bf68c2496b3de7882aab" \
      org.hermes.pi.bridge.sha256="26058182f2aa31e5e3fd46160143e07cd3f6d506b8898970755128d2faac6bf5" \
      org.hermes.pi.registry-npm-tree.sha256="1951e4981b80b2ffd23f8f6c26f91598b15e3892a94108f9b91a321c0f805716" \
      org.hermes.pi.image-npm-tree.sha256="f845429e8c8fa262101900dd8eddaf43193ed9c0db51ab25085c6bedefc51370" \
      org.hermes.pi.verification-requirements.sha256="18ceee9cfc1b05a75456c272adb20dfda896bd46064f105d0357898db9911857"

COPY --from=node_source /usr/local/bin/node /usr/local/bin/node
COPY --from=node_source /usr/local/lib/node_modules /usr/local/lib/node_modules
RUN ln -s /usr/local/lib/node_modules/npm/bin/npm-cli.js /usr/local/bin/npm \
    && ln -s /usr/local/lib/node_modules/npm/bin/npx-cli.js /usr/local/bin/npx

COPY Dockerfile build-image.sh egress_proxy.py hermes_bridge.ts package.json package-lock.json runtime-integrity.json verification-requirements.in verification-requirements.txt verify_npm_lock.py /opt/hermes/pi-build/
COPY egress_proxy.py /opt/hermes/pi_egress_proxy.py
COPY hermes_bridge.ts /opt/hermes/hermes_bridge.ts

# The registry response is not trusted by version alone. The build-context lock
# is derived from the reviewed production shrinkwrap but has a complete root
# package entry, which npm 11 requires for npm ci.
RUN cd /opt/hermes/pi-build \
    && test "$(npm --version)" = "11.17.0" \
    && python3 verify_npm_lock.py --npm /usr/local/bin/npm --npm-tree-source image --verify-bundle --integrity-only \
    && npm pack @earendil-works/pi-coding-agent@0.84.3 --pack-destination /tmp \
    && test "$(sha512sum /tmp/earendil-works-pi-coding-agent-0.84.3.tgz | cut -d' ' -f1)" = "62bda9f4fb9badb15999810f608f82f0a9993fdc6516e2c39c01bae11b54d190e0add8976166becbb586c893b93a5a8f9623a454231df48cd5b333b7fedd03d3" \
    && test "$(sha256sum package-lock.json | cut -d' ' -f1)" = "9fef223236a34bcab3a6926dcc1f3cd1f4d3504e3ce5eb37ac840b5ea0b280f2" \
    && test "$(sha256sum verification-requirements.txt | cut -d' ' -f1)" = "18ceee9cfc1b05a75456c272adb20dfda896bd46064f105d0357898db9911857" \
    && npm ci --omit=dev --no-audit --no-fund \
    && pi_package=/opt/hermes/pi-build/node_modules/@earendil-works/pi-coding-agent \
    && test "$(sha256sum "$pi_package/npm-shrinkwrap.json" | cut -d' ' -f1)" = "df18751951cef578a1e5d809b33774bb385f69539f197bade0e70f1944aa3de7" \
    && test "$(sha256sum "$pi_package/package.json" | cut -d' ' -f1)" = "6c0bfa4558a0de32fb54fa97ca73397753ec7d4ad7d0ba607790920168de751f" \
    && test "$(sha256sum "$pi_package/dist/bundle/cli.js" | cut -d' ' -f1)" = "1c3a5094b54aae9ae98c66516ce8c6578140363d081471ca7e91f9cb8c23dc8a" \
    && test "$(sha256sum /opt/hermes/pi_egress_proxy.py | cut -d' ' -f1)" = "0d8f3d724b70d141072fc51c42ccd6cc17296a5775d0bf68c2496b3de7882aab" \
    && test "$(sha256sum /opt/hermes/hermes_bridge.ts | cut -d' ' -f1)" = "26058182f2aa31e5e3fd46160143e07cd3f6d506b8898970755128d2faac6bf5" \
    && ln -s /opt/hermes/pi-build/node_modules/.bin/pi /usr/local/bin/pi \
    && chmod 0555 /opt/hermes/pi_egress_proxy.py /opt/hermes/hermes_bridge.ts \
    && test "$(pi --version)" = "0.84.3" \
    && rm -f /tmp/earendil-works-pi-coding-agent-0.84.3.tgz \
    && rm -rf /root/.npm /tmp/node-compile-cache /tmp/npm-*

RUN uv venv --python /usr/local/bin/python3 /opt/hermes/verify-venv \
    && uv pip install --python /opt/hermes/verify-venv/bin/python \
        --no-cache --require-hashes \
        --requirements /opt/hermes/pi-build/verification-requirements.txt \
    && test "$(PYTHONDONTWRITEBYTECODE=1 /opt/hermes/verify-venv/bin/pytest --version | awk '{print $2}')" = "9.1.1" \
    && test "$(PYTHONDONTWRITEBYTECODE=1 /opt/hermes/verify-venv/bin/ruff --version | awk '{print $2}')" = "0.15.10" \
    && test "$(PYTHONDONTWRITEBYTECODE=1 /opt/hermes/verify-venv/bin/ty --version | awk '{print $2}')" = "0.0.21" \
    && find /opt/hermes/verify-venv -type d -name __pycache__ -prune -exec rm -rf {} + \
    && find /opt/hermes/verify-venv -type f -name '*.pyc' -delete \
    && rm -rf /root/.cache/uv /tmp/uv-*

# The digest-pinned uv base currently carries Git. Pi must not receive a VCS
# client or its directly invokable helpers/data merely because that base does.
# Remove the payload explicitly instead of performing a mutable apt operation.
RUN rm -f /usr/bin/git /usr/bin/git-* /usr/bin/scalar \
    && rm -rf /usr/lib/git-core /usr/libexec/git-core \
        /usr/share/git-core /usr/share/gitweb \
        /usr/share/perl5/Git /usr/share/perl5/Git.pm \
    && ! command -v git \
    && ! command -v scalar \
    && test -z "$(find /usr/bin -maxdepth 1 \
        \( -name git -o -name 'git-*' \) -print -quit)" \
    && test ! -e /usr/lib/git-core \
    && test ! -e /usr/libexec/git-core \
    && test ! -e /usr/share/git-core \
    && test ! -e /usr/share/gitweb \
    && test ! -e /usr/share/perl5/Git \
    && test ! -e /usr/share/perl5/Git.pm

WORKDIR /workspace
ENTRYPOINT ["pi"]
