#!/usr/bin/env bash
# Start/stop the validated two-host GLM engine together. Never kill an unrecorded server.
set -euo pipefail
remote_stage_root="${MAYA_HOME_STAGE_ROOT:-$HOME/fn}"
remote_stage_config="$remote_stage_root/maya-config/remote-prefill-20261030"
remote_stage_active="$remote_stage_root/remote-prefill-active-run.txt"
case "${1:-start}" in
  start|restart)
    if [[ -n "$(ss -ltnpH 'sport = :18100')" ]]; then
      [[ -f "$remote_stage_active" ]] || { echo 'No recorded server to replace.' >&2; exit 1; }
      python3 "$remote_stage_root/stop_remote_stage.py" "$(cat "$remote_stage_active")"
    fi
    remote_stage_run="$remote_stage_root/remote-prefill-runs/$(date +%Y%m%d-%H%M%S)-$$"
    MAYA_STAGE_BENCH_BINARY="$remote_stage_root/bin/maya-remote-prefill-20261030" \
    MAYA_STAGE_TEST_GATE="$remote_stage_config" \
    MAYA_STAGE_CONFIG_TEMPLATE="$remote_stage_config/template.json" \
    MAYA_STAGE_SKIP_HARNESS=1 \
    MAYA_STAGE_SLOT_RAM_GB=2 \
    MAYA_STAGE_BENCH_FRONT_EXTRA='{"MAYA_GLM_PREFILL_MIN":"32","MAYA_GLM_PREFILL_PIPE_ROWS":"8192"}' \
    MAYA_STAGE_BENCH_EXTRA='{"MAYA_GLM_PREFILL_MIN":"32","MAYA_GLM_PREFILL_PIPE_ROWS":"8192"}' \
      python3 "$remote_stage_root/run_remote_stage.py" "$remote_stage_run" 2
    printf '%s\n' "$remote_stage_run" > "$remote_stage_active.tmp"
    mv "$remote_stage_active.tmp" "$remote_stage_active"
    echo "Active configuration: $remote_stage_run/config.json"
    ;;
  stop)
    if [[ -n "$(ss -ltnpH 'sport = :18100')" ]]; then
      [[ -f "$remote_stage_active" ]] || { echo 'No recorded server to stop.' >&2; exit 1; }
      python3 "$remote_stage_root/stop_remote_stage.py" "$(cat "$remote_stage_active")"
    fi
    ;;
  status)
    [[ ! -f "$remote_stage_active" ]] || cat "$remote_stage_active"
    python3 -c 'import urllib.request;print(urllib.request.urlopen("http://127.0.0.1:18100/health",timeout=3).read().decode())'
    ;;
  *) echo "Usage: $0 [start|restart|stop|status]" >&2; exit 2 ;;
esac
