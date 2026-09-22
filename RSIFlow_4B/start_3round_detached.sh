#!/usr/bin/env bash
set -Eeuo pipefail
umask 077

script_dir="$(cd -- "$(dirname -- "$0")" && pwd -P)"
log_dir="/root/data/RSI_iclr2027/rsiH/Rollout_logs/logs"
mkdir -p -- "$log_dir"
launcher_log="$log_dir/rsiflow_4b_detached_launcher.log"
pid_file="$log_dir/rsiflow_4b_detached_launcher.pid"

if [[ -f "$pid_file" ]]; then
  read -r existing_pid < "$pid_file"
  if [[ "$existing_pid" =~ ^[0-9]+$ ]] && kill -0 "$existing_pid" 2>/dev/null; then
    printf 'Experiment launcher already running: PID %s\n' "$existing_pid"
    exit 0
  fi
fi

# The controller must outlive this Codex/IDE terminal. The original script
# still owns preflight, frozen config, training, and resume semantics.
nohup setsid bash "$script_dir/start_3round_training.sh" >> "$launcher_log" 2>&1 < /dev/null &
launcher_pid=$!
printf '%s\n' "$launcher_pid" > "$pid_file"
sleep 2
if ! kill -0 "$launcher_pid" 2>/dev/null; then
  printf 'Detached launcher exited during startup; inspect %s\n' "$launcher_log" >&2
  exit 1
fi
printf 'Detached launcher PID: %s\nLauncher log: %s\n' "$launcher_pid" "$launcher_log"
