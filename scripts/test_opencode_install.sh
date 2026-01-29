#!/usr/bin/env bash
#
# Run the OpenCode install test inside a clean Docker container.
#
# Usage:
#   ./test_opencode_install.sh                             # install-only
#   ./test_opencode_install.sh --run-query                 # full test (needs API key)
#   ./test_opencode_install.sh --run-query --json          # JSON report
#   ./test_opencode_install.sh -v                          # verbose
#
# API keys are forwarded from the host environment:
#   OPENAI_API_KEY=sk-...  ./test_opencode_install.sh --run-query
#   ANTHROPIC_API_KEY=... ./test_opencode_install.sh --run-query
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGE_NAME="opencode-install-test"

echo "══════════════════════════════════════════════════════════"
echo "  Building Docker image: ${IMAGE_NAME}"
echo "══════════════════════════════════════════════════════════"

docker build -t "${IMAGE_NAME}" -f - "${SCRIPT_DIR}" <<'DOCKERFILE'
FROM python:3.12-slim

# Minimal bootstrap – the test script installs everything else
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

COPY test_opencode_install.py /test_opencode_install.py

# -u = unbuffered stdout so logs stream in real-time
ENTRYPOINT ["python", "-u", "/test_opencode_install.py"]
DOCKERFILE

echo ""
echo "══════════════════════════════════════════════════════════"
echo "  Running test container"
echo "══════════════════════════════════════════════════════════"
echo ""

docker run --rm \
    --network=host \
    -e OPENAI_API_KEY="${OPENAI_API_KEY:-}" \
    -e ANTHROPIC_API_KEY="${ANTHROPIC_API_KEY:-}" \
    "${IMAGE_NAME}" "$@"
