#!/usr/bin/env bash
# Smoke-test a built image in demo mode (no API key): health, UI, ingest of the sample docs,
# a search, and that the index survives a restart; then, with RAG_PASSWORD set, that Docker's
# HEALTHCHECK still passes and the API is locked until login.  Usage: scripts/docker_smoke_test.sh [image]
set -euo pipefail
IMAGE="${1:-agentic-rag}"
NAME="rag-smoke-$$"
PORT="${PORT:-8000}"
URL="http://127.0.0.1:$PORT"
JAR="$(mktemp)"
cleanup() { docker rm -f "$NAME" >/dev/null 2>&1 || true; rm -f "$JAR"; }
trap cleanup EXIT

fail() { echo "FAIL: $*" >&2; docker logs "$NAME" 2>&1 | tail -30 >&2; exit 1; }
wait_healthy() {
  for _ in $(seq 1 90); do curl -sf "$URL/api/health" >/dev/null && return 0; sleep 1; done
  fail "server did not become healthy"
}
json() { python3 -c "import sys, json; d = json.load(sys.stdin); print($1)"; }
code() { curl -s -o /dev/null -w '%{http_code}' "$@"; }

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

# ---- login (RAG_PASSWORD): a fresh container with a password
docker rm -f "$NAME" >/dev/null
PASSWORD="smoke test passphrase"
docker run -d --name "$NAME" -p "127.0.0.1:$PORT:8000" -e RAG_FAKE_LLM=1 -e RAG_RERANK=0 -e HF_HUB_OFFLINE=1 \
  -e RAG_PASSWORD="$PASSWORD" --health-interval=2s "$IMAGE" >/dev/null
status=starting
for _ in $(seq 1 90); do
  status=$(docker inspect -f '{{.State.Health.Status}}' "$NAME")
  [ "$status" = healthy ] && break
  [ "$status" = unhealthy ] && fail "HEALTHCHECK unhealthy with RAG_PASSWORD set"
  sleep 1
done
[ "$status" = healthy ] || fail "HEALTHCHECK not healthy with RAG_PASSWORD set (last: $status)"

[ "$(code "$URL/api/health")" = 401 ] || fail "/api/health answered without a login"
curl -sf "$URL/" | grep -q "<title>" || fail "UI not served before login"
login() { code -c "$JAR" -X POST "$URL/api/auth/login" -H 'content-type: application/json' -d "{\"password\": \"$1\"}"; }
[ "$(login wrong)" = 401 ] || fail "wrong password accepted"
[ "$(login "$PASSWORD")" = 200 ] || fail "login failed"
TOKEN=$(awk '$6 == "rag_session" {print $7}' "$JAR")
[ -n "$TOKEN" ] || fail "login set no rag_session cookie"
[ "$(code -H "Cookie: rag_session=$TOKEN" "$URL/api/health")" = 200 ] || fail "session cookie refused"
# With a password, other hostnames are allowed (the cookie guards them instead).
[ "$(code -H "Cookie: rag_session=$TOKEN" -H 'Host: rag.lan:8000' "$URL/api/health")" = 200 ] || fail "other hostname refused"
[ "$(code -H 'Host: rag.lan:8000' "$URL/api/health")" = 401 ] || fail "other hostname answered without a login"

docker restart "$NAME" >/dev/null
wait_up() { for _ in $(seq 1 90); do [ "$(code "$URL/api/ping")" = 200 ] && return 0; sleep 1; done; fail "server did not come back"; }
wait_up
[ "$(code -H "Cookie: rag_session=$TOKEN" "$URL/api/health")" = 200 ] || fail "session lost across restart"

echo "OK: with RAG_PASSWORD, healthy, API locked until login, session kept across restart"
