#!/usr/bin/env bash

set -Eeuo pipefail
case "$-" in *x*) set +x ;; esac
umask 077

SCRIPT_DIR="$(cd -- "$(dirname -- "$0")" && pwd -P)"
source "$SCRIPT_DIR/storage_env.sh"
PYTHON_BIN="${RSIFLOW_PYTHON:-/root/data/conda/envs/sia/bin/python}"
PYTHON_BIN="$(readlink -f -- "$PYTHON_BIN")"
BWRAP_BIN="${RSIFLOW_BWRAP:-/usr/bin/bwrap}"
CODEX_BIN="${RSIFLOW_CODEX_BIN:-/root/data/RSI_iclr2027/rsiH/third_party/codex-bin/codex-x86_64-unknown-linux-musl}"
CATALOG="${RSIFLOW_CODEX_CATALOG:-/root/data/RSI_iclr2027/rsiH/configs/catalog-meta-reload-providers-20260912.json}"
BRIDGE="$SCRIPT_DIR/runtime/sia/task_meta/meta_backends/bridge.py"
WORKER_SOURCE="$SCRIPT_DIR/runtime/sia/task_meta/meta_backends/remote_worker.py"
WORKER_ROOT="${RSIFLOW_META_WORKER_ROOT:-$SCRIPT_DIR/runtime/meta_worker}"
WORKER_PORT="${RSIFLOW_META_WORKER_PORT:-19071}"
WORKER_SOCKET="${RSIFLOW_META_WORKER_SOCKET:-/tmp/rsi_meta_joint_20260918_worker.sock}"
RELAY_SOCKET="${RSIFLOW_META_RELAY_SOCKET:-/tmp/rsi_meta_joint_20260918_relay.sock}"
STATE_DIR="${RSIFLOW_META_WORKER_STATE:-/root/data/RSI_iclr2027/.state/RSIFlow_4B}"
TOKEN_FILE="${RSI_REMOTE_WORKER_TOKEN_FILE:-$STATE_DIR/meta_worker_token}"
WORKER_PID_FILE="$STATE_DIR/meta_worker.pid"
PROXY_PID_FILE="$STATE_DIR/meta_worker_proxy.pid"
LOG_FILE="$STATE_DIR/meta_worker.log"
PROXY_LOG="$STATE_DIR/meta_worker_proxy.log"

die() { printf 'Meta worker 启动失败：%s\n' "$*" >&2; exit 2; }
for path in "$PYTHON_BIN" "$BWRAP_BIN" "$CODEX_BIN" "$CATALOG" "$BRIDGE" "$WORKER_SOURCE"; do
  [[ -f "$path" && ! -L "$path" ]] || die "缺少或拒绝符号链接：$path"
done
command -v socat >/dev/null || die "缺少 socat"
command -v openssl >/dev/null || die "缺少 openssl"
[[ "$WORKER_PORT" =~ ^[0-9]+$ ]] || die "worker port 必须是整数"

mkdir -p "$STATE_DIR" "$WORKER_ROOT"
chmod 700 "$STATE_DIR" "$WORKER_ROOT"

stop_pid_file() {
  local file="$1" marker="$2" pid command
  [[ -f "$file" ]] || return 0
  IFS= read -r pid < "$file"
  [[ "$pid" =~ ^[0-9]+$ ]] || die "无效 PID 文件：$file"
  if [[ -r "/proc/$pid/cmdline" ]]; then
    command="$(tr '\0' ' ' < "/proc/$pid/cmdline")"
    [[ "$command" == *"$marker"* ]] || die "PID $pid 不属于预期进程；拒绝终止"
    kill "$pid"
    for _ in {1..50}; do
      kill -0 "$pid" 2>/dev/null || break
      sleep 0.1
    done
    kill -0 "$pid" 2>/dev/null && die "进程 $pid 未能安全停止"
  fi
  rm -f "$file"
}

stop_pid_file "$PROXY_PID_FILE" "socat"
stop_pid_file "$WORKER_PID_FILE" "sia.task_meta.meta_backends.remote_worker"
rm -f "$WORKER_SOCKET"

"$BWRAP_BIN" --unshare-all --unshare-user --die-with-parent --ro-bind / / /bin/true ||
  die "bubblewrap 隔离探针失败；worker 宿主必须允许 user/mount namespace"

read -r expected_worker expected_bwrap < <(
  "$PYTHON_BIN" - "$SCRIPT_DIR/runtime/configs/base.json" <<'PY'
import json, sys
value=json.load(open(sys.argv[1]))
meta=value["meta"]
print(meta["remote_worker_sha256"], meta["remote_bwrap_sha256"])
PY
)
actual_worker="$(sha256sum "$WORKER_SOURCE" | awk '{print $1}')"
actual_bwrap="$(sha256sum "$BWRAP_BIN" | awk '{print $1}')"
[[ "$actual_worker" == "$expected_worker" ]] || die "worker 源码哈希与配置 pin 不一致"
[[ "$actual_bwrap" == "$expected_bwrap" ]] || die "bubblewrap 哈希与配置 pin 不一致"

if [[ ! -f "$TOKEN_FILE" ]]; then
  token_tmp="$TOKEN_FILE.tmp.$$"
  openssl rand -hex 32 > "$token_tmp"
  chmod 600 "$token_tmp"
  mv "$token_tmp" "$TOKEN_FILE"
fi
[[ -f "$TOKEN_FILE" && ! -L "$TOKEN_FILE" ]] || die "worker token 文件不安全"
[[ "$(wc -l < "$TOKEN_FILE")" -eq 1 ]] || die "worker token 必须是单行"
worker_token="$(<"$TOKEN_FILE")"
[[ "${#worker_token}" -ge 24 ]] || die "worker token 太短"

env RSI_REMOTE_WORKER_TOKEN="$worker_token" PYTHONPATH="$SCRIPT_DIR/runtime" \
  nohup "$PYTHON_BIN" -m sia.task_meta.meta_backends.remote_worker \
    --root "$WORKER_ROOT" --codex-executable "$CODEX_BIN" --catalog "$CATALOG" \
    --bridge "$BRIDGE" --bwrap "$BWRAP_BIN" --relay-socket "$RELAY_SOCKET" \
    --port "$WORKER_PORT" >"$LOG_FILE" 2>&1 &
worker_pid=$!
printf '%s\n' "$worker_pid" > "$WORKER_PID_FILE"

ready=false
for _ in {1..50}; do
  if WORKER_TOKEN_FILE="$TOKEN_FILE" WORKER_PORT="$WORKER_PORT" "$PYTHON_BIN" - <<'PY'
import http.client, json, os
token=open(os.environ["WORKER_TOKEN_FILE"]).read().strip()
conn=http.client.HTTPConnection("127.0.0.1", int(os.environ["WORKER_PORT"]), timeout=1)
try:
    conn.request("GET", "/health", headers={"Authorization":"Bearer "+token})
    response=conn.getresponse()
    value=json.loads(response.read())
    if response.status != 200 or value.get("ready") is not True:
        raise SystemExit(1)
finally:
    conn.close()
PY
  then ready=true; break
  fi
  kill -0 "$worker_pid" 2>/dev/null || break
  sleep 0.2
done
[[ "$ready" == true ]] || {
  tail -40 "$LOG_FILE" >&2 || true
  die "worker 未通过 ready/isolation/Codex 健康检查"
}

nohup socat "UNIX-LISTEN:$WORKER_SOCKET,fork,unlink-early,mode=600" \
  "TCP:127.0.0.1:$WORKER_PORT" >"$PROXY_LOG" 2>&1 &
proxy_pid=$!
printf '%s\n' "$proxy_pid" > "$PROXY_PID_FILE"
for _ in {1..30}; do
  [[ -S "$WORKER_SOCKET" ]] && break
  kill -0 "$proxy_pid" 2>/dev/null || break
  sleep 0.1
done
[[ -S "$WORKER_SOCKET" ]] || die "worker Unix socket 代理未就绪"

unset worker_token
printf 'Meta worker 已就绪。token 文件：%s\n' "$TOKEN_FILE"
printf '启动训练：RSI_REMOTE_WORKER_TOKEN_FILE=%q %q/start_3round_training.sh\n' "$TOKEN_FILE" "$SCRIPT_DIR"
