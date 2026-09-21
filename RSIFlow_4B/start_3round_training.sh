#!/usr/bin/env bash

set -Eeuo pipefail

# Secrets must never be emitted, including when a caller invokes this with bash -x.
case "$-" in
  *x*) set +x ;;
esac
umask 077

# Each rollout worker must not create a host-sized BLAS/tokenizer pool.
# GPU serving sets its own four intra-op threads; Meta has a separate bounded pool.
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false RAYON_NUM_THREADS=4 TOKIO_WORKER_THREADS=4

SCRIPT_DIR="$(cd -- "$(dirname -- "$0")" && pwd -P)"
source "$SCRIPT_DIR/storage_env.sh"
CONFIG_PATH="${RSIFLOW_CONFIG:-$SCRIPT_DIR/configs/train.json}"
KEY_FILE="${RSIFLOW_API_KEY_FILE:-$SCRIPT_DIR/API_key.md}"
PYTHON_BIN="${RSIFLOW_PYTHON:-/root/data/conda/envs/sia/bin/python}"
META_WORKER_SCRIPT="$SCRIPT_DIR/start_meta_worker.sh"
export RSI_REMOTE_WORKER_TOKEN_FILE="${RSI_REMOTE_WORKER_TOKEN_FILE:-/root/data/RSI_iclr2027/.state/RSIFlow_4B/meta_worker_token}"
CURRENT_UID="$(id -u)"

# The token file is authoritative. Do not let stale inherited credentials reach
# preflight or worker-start subprocesses before the checked files are loaded.
unset RSI_REMOTE_WORKER_TOKEN AUTODL_API_KEY ACE_USER_API_KEY

die() {
  printf '启动失败：%s\n' "$*" >&2
  exit 2
}

assert_secure_secret_file() {
  local path="$1"
  local description="$2"
  local metadata owner mode links kind

  [[ ! -L "$path" ]] || die "$description 不能是符号链接：$path"
  [[ -e "$path" ]] || die "$description不存在：$path"
  metadata="$(LC_ALL=C stat -c '%u:%a:%h:%F' -- "$path")" ||
    die "无法检查$description：$path"
  IFS=: read -r owner mode links kind <<< "$metadata"
  [[ "$kind" == "regular file" ]] || die "$description不是普通文件：$path"
  [[ "$owner" == "$CURRENT_UID" ]] || die "$description必须归当前用户所有：$path"
  [[ "$mode" == "600" ]] || die "$description权限必须严格为 0600：$path"
  [[ "$links" == "1" ]] || die "$description必须只有一个硬链接：$path"
}

ensure_output_directory() {
  local path="$1"
  local metadata owner mode kind

  [[ ! -L "$path" ]] || die "output_root 不能是符号链接：$path"
  mkdir -p -- "$path" || die "无法创建 output_root：$path"
  [[ ! -L "$path" ]] || die "output_root 不能是符号链接：$path"
  metadata="$(LC_ALL=C stat -c '%u:%a:%F' -- "$path")" ||
    die "无法检查 output_root：$path"
  IFS=: read -r owner mode kind <<< "$metadata"
  [[ "$kind" == "directory" ]] || die "output_root 不是目录：$path"
  [[ "$owner" == "$CURRENT_UID" ]] || die "output_root 必须归当前用户所有：$path"
  chmod 700 -- "$path" || die "无法收紧 output_root 权限：$path"
  metadata="$(LC_ALL=C stat -c '%u:%a:%F' -- "$path")" ||
    die "无法复查 output_root：$path"
  IFS=: read -r owner mode kind <<< "$metadata"
  [[ "$owner" == "$CURRENT_UID" && "$mode" == "700" && "$kind" == "directory" ]] ||
    die "output_root 必须是当前用户拥有的 0700 目录：$path"
  [[ -w "$path" && -x "$path" ]] || die "output_root 不可写：$path"
}

trim_secret() {
  local value="$1"
  value="${value#"${value%%[![:space:]]*}"}"
  value="${value%"${value##*[![:space:]]}"}"
  printf '%s' "$value"
}

read_labeled_secret() {
  local label="$1"
  local output_name="$2"
  local line candidate value=""
  local matches=0

  assert_secure_secret_file "$KEY_FILE" "API 密钥文件"
  while IFS= read -r line || [[ -n "$line" ]]; do
    line="${line%$'\r'}"
    if [[ "$line" == "$label："* ]]; then
      candidate="${line#"$label："}"
    elif [[ "$line" == "$label:"* ]]; then
      candidate="${line#"$label:"}"
    else
      continue
    fi
    matches=$((matches + 1))
    candidate="$(trim_secret "$candidate")"
    [[ -n "$candidate" ]] || die "密钥文件中的 $label 值为空"
    value="$candidate"
  done < "$KEY_FILE"

  (( matches == 1 )) || die "密钥文件必须恰好包含一条 $label 记录"
  printf -v "$output_name" '%s' "$value"
}

[[ -x "$PYTHON_BIN" ]] || die "Python 不可执行：$PYTHON_BIN"
[[ -f "$CONFIG_PATH" ]] || die "训练配置不存在：$CONFIG_PATH"
SYSTEM_USED_BYTES="$(df -B1 --output=used / | tail -n 1)"
(( SYSTEM_USED_BYTES + 4000000000 <= 20000000000 )) || die "系统盘容量检查未通过：已用 $SYSTEM_USED_BYTES 字节，需为运行时预留 4 GB，使用上限 20 GB。请先完成旧环境迁移。"

OUTPUT_ROOT="$("$PYTHON_BIN" -c '
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
config = json.loads(path.read_text())
if config.get("rounds") != 3:
    raise SystemExit("配置必须固定为 3 轮")
if config.get("pause_after_round") is not None:
    raise SystemExit("连续三轮启动不允许设置 pause_after_round")
if config.get("training_tasks_per_pass") not in (180, 360):
    raise SystemExit("每轮必须是三个领域混合的 180 或 360 条任务")
if config.get("candidate_policy") != "single_candidate_strict_positive_gain":
    raise SystemExit("候选接受策略必须是 single_candidate_strict_positive_gain")
output = config.get("output_root")
if not isinstance(output, str) or not output:
    raise SystemExit("配置必须提供 output_root")
configured = Path(output)
if not configured.is_absolute():
    raise SystemExit("output_root 必须是绝对路径")
raw = configured.absolute()
project = Path(sys.argv[2]).resolve()
resolved = raw.resolve()
if resolved == Path("/") or resolved == project or resolved.is_relative_to(project) or project.is_relative_to(resolved):
    raise SystemExit("output_root 必须是项目树之外的专用目录")
print(raw)
' "$CONFIG_PATH" "$SCRIPT_DIR")"

ensure_output_directory "$OUTPUT_ROOT"
export RSIFLOW_OUTPUT_ROOT="$OUTPUT_ROOT"

assert_secure_secret_file "$KEY_FILE" "API 密钥文件"
EXECUTION_LOCATION="$("$PYTHON_BIN" -c '
import json, sys
from pathlib import Path
config = json.loads(Path(sys.argv[1]).read_text())
source = Path(config.get("runtime_source", "runtime"))
if not source.is_absolute():
    source = Path(sys.argv[2]) / source
base = source / config.get("base_config", "configs/base.json")
print(json.loads(base.read_text())["meta"]["execution_location"] if base.is_file() else "ssh_worker")
' "$CONFIG_PATH" "$SCRIPT_DIR")"

if [[ "$EXECUTION_LOCATION" == "ssh_worker" ]]; then
[[ -f "$META_WORKER_SCRIPT" && ! -L "$META_WORKER_SCRIPT" && -x "$META_WORKER_SCRIPT" ]] ||
  die "Meta worker 启动脚本缺失、不可执行或是符号链接：$META_WORKER_SCRIPT"

# Ensure/reuse the worker before reading any secret into this shell. The worker
# creates the token file on first launch and must return only after it is healthy.
"$META_WORKER_SCRIPT" --ensure

assert_secure_secret_file "$RSI_REMOTE_WORKER_TOKEN_FILE" "Meta worker token 文件"
mapfile -t worker_token_lines < "$RSI_REMOTE_WORKER_TOKEN_FILE"
(( ${#worker_token_lines[@]} == 1 )) ||
  die "Meta worker token 文件必须只包含一行"
[[ "${worker_token_lines[0]}" =~ ^[[:xdigit:]]{64}$ ]] ||
  die "Meta worker token 必须是 64 位十六进制字符"
export RSI_REMOTE_WORKER_TOKEN="${worker_token_lines[0]}"
unset worker_token_lines
elif [[ "$EXECUTION_LOCATION" == "local_chroot" ]]; then
  printf '%s\n' "Meta 使用本机按次启动的 Codex；无需常驻 Worker。"
else
  die "不支持的 Meta execution_location：$EXECUTION_LOCATION"
fi

read_labeled_secret "openrouter" openrouter_key
read_labeled_secret "autodl.art" autodl_key
export AUTODL_API_KEY="$autodl_key"
export ACE_USER_API_KEY="$openrouter_key"
export ACE_USER_BASE_URL="${RSIFLOW_ACE_BASE_URL:-https://openrouter.ai/api/v1}"
unset openrouter_key autodl_key

printf '%s\n' "开始或恢复连续三轮实验：Task1→Meta1→Task2→Meta2→Task3→Meta3"
printf '运行输出目录：%s\n' "$OUTPUT_ROOT"
printf '进度：%s；训练日志：%s\n' "$OUTPUT_ROOT/active_run.json" "$OUTPUT_ROOT/logs/"
printf '%s\n' "若进程中断，使用同一命令重跑本脚本即可从已有 receipts 恢复。"

if [[ -n "${RSIFLOW_RESUME_DEPLOYED_RUN:-}" ]]; then
  exec "$PYTHON_BIN" -u "$SCRIPT_DIR/resume_experiment.py" --run-dir "$RSIFLOW_RESUME_DEPLOYED_RUN"
fi
exec "$PYTHON_BIN" -u "$SCRIPT_DIR/launch.py" --config "$CONFIG_PATH" --execute
