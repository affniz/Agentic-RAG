#!/usr/bin/env bash
# Smoke-test a built image in demo mode (no API key): health, UI, ingest of the sample docs,
# a search, and that the index survives a restart.  Usage: scripts/docker_smoke_test.sh [image]
set -euo pipefail
IMAGE="${1:-agentic-rag}"
NAME="rag-smoke-$$"
PORT="${PORT:-8000}"
URL="http://127.0.0.1:$PORT"
cleanup() { docker rm -f "$NAME" >/dev/null 2>&1 || true; }
trap cleanup EXIT

fail() { echo "FAIL: $*" >&2; docker logs "$NAME" 2>&1 | tail -30 >&2; exit 1; }
wait_healthy() {
  for _ in $(seq 1 90); do curl -sf "$URL/api/health" >/dev/null && return 0; sleep 1; done
  fail "server did not become healthy"
}
json() { python3 -c "import sys, json; d = json.load(sys.stdin); print($1)"; }

docker run -d --name "$NAME" -p "127.0.0.1:$PORT:8000" -e RAG_FAKE_LLM=1 -e RAG_RERANK=0 -e HF_HUB_OFFLINE=1 "$IMAGE" >/dev/null
wait_healthy
curl -sf "$URL/" | grep -q "<title>" || fail "UI not served"

curl -sf -X POST "$URL/api/ingest" -H 'content-type: application/json' -d '{"rebuild": false}' >/dev/null
for _ in $(seq 1 120); do
  state=$(curl -sf "$URL/api/ingest" | json 'd.get("state")')
  [ "$state" = done ] && break
  [ "$state" = error ] && fail "ingest failed: $(curl -s "$URL/api/ingest")"
  sleep 1
done
[ "$state" = done ] || fail "ingest did not finish"

top=$(curl -sf "$URL/api/search?q=carry%20over&k=1" | json 'd["results"][0]["source"]')
[ "$top" = acme_vacation_policy.md ] || fail "unexpected top search result: $top"

docker restart "$NAME" >/dev/null
wait_healthy
chunks=$(curl -sf "$URL/api/health" | json 'd["chunks"]')
[ "$chunks" -gt 0 ] || fail "index lost after restart"

echo "OK: healthy, UI served, ingested and searched, index kept across restart ($chunks chunks)"
