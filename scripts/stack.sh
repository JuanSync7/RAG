#!/usr/bin/env bash
# @summary
# RagWeave stack control: status table + start/stop/restart per service,
# full up/down, logs. CLI companion to Dozzle (http://localhost:9999).
#
# Usage:
#   ./scripts/stack.sh                      # interactive menu
#   ./scripts/stack.sh status               # status table only
#   ./scripts/stack.sh start <service>      # start one service
#   ./scripts/stack.sh stop <service>       # stop one service
#   ./scripts/stack.sh restart <service>    # restart one service
#   ./scripts/stack.sh logs <service>       # tail logs (Ctrl+C to exit)
#   ./scripts/stack.sh up                   # bring everything up (all profiles, no rebuild)
#   ./scripts/stack.sh down                 # bring everything down
#   ./scripts/stack.sh restart              # base + workers, rebuild + force-recreate (mirrors `make start`)
#   ./scripts/stack.sh restart-all          # all profiles, rebuild + force-recreate (mirrors `make start-all`)
#   ./scripts/stack.sh ui                   # open Dozzle in browser
# @end-summary
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# All profiles defined in docker-compose.yml. `up`/`down` activate every one.
ALL_PROFILES=(app workers monitoring observability gateway temporal)

# tty colors
if [[ -t 1 ]]; then
    G=$'\e[32m'; R=$'\e[31m'; Y=$'\e[33m'; B=$'\e[34m'; D=$'\e[2m'; X=$'\e[0m'
else
    G=""; R=""; Y=""; B=""; D=""; X=""
fi

_compose() {
    bash "$REPO_ROOT/scripts/compose.sh" "$@"
}

_compose_with_all_profiles() {
    local args=()
    for p in "${ALL_PROFILES[@]}"; do args+=(--profile "$p"); done
    _compose "${args[@]}" "$@"
}

# All container names declared in compose (kept in sync manually).
_all_containers() {
    cat <<'EOF'
rag-postgres
rag-pg-maintenance
rag-temporal-db
rag-temporal
rag-temporal-ui
rag-api
rag-worker
rag-monitor
rag-minio
rag-weaviate
rag-redis
rag-prometheus
rag-alertmanager
rag-grafana
rag-langfuse-postgres
rag-langfuse-redis
rag-langfuse-clickhouse
rag-langfuse-minio
rag-langfuse-worker
rag-langfuse-web
rag-embed
rag-rerank
rag-ollama
rag-nginx
EOF
}

cmd_status() {
    printf "\n%sRagWeave stack status%s  %s(Dozzle UI: http://localhost:%s)%s\n\n" \
        "$B" "$X" "$D" "${DOZZLE_HOST_PORT:-9999}" "$X"
    printf "  %-28s %s\n" "SERVICE" "STATUS"
    printf "  %-28s %s\n" "-------" "------"
    local running
    running=$(docker ps --format '{{.Names}}|{{.Status}}' 2>/dev/null || true)
    while IFS= read -r name; do
        local line status
        line=$(grep "^${name}|" <<<"$running" || true)
        if [[ -n "$line" ]]; then
            status="${G}● ${line#*|}${X}"
        elif docker ps -a --format '{{.Names}}' 2>/dev/null | grep -qx "$name"; then
            status="${Y}○ stopped${X}"
        else
            status="${D}· not created${X}"
        fi
        printf "  %-28s %s\n" "$name" "$status"
    done < <(_all_containers)
    echo ""
}

cmd_start()   { docker start "$1" >/dev/null && echo "${G}started${X} $1" || echo "${R}failed${X} $1"; }
cmd_stop()    { docker stop  "$1" >/dev/null && echo "${Y}stopped${X} $1" || echo "${R}failed${X} $1"; }
cmd_restart() { docker restart "$1" >/dev/null && echo "${G}restarted${X} $1" || echo "${R}failed${X} $1"; }
cmd_logs()    { docker logs -f --tail 100 "$1"; }

cmd_up()   { _compose_with_all_profiles up -d; }
cmd_down() { _compose_with_all_profiles down --remove-orphans; }

# `restart` = same scope as `make start` (base + workers), but rebuilt.
cmd_restart_base() {
    _compose up -d --build --force-recreate
    _compose --profile workers up -d --build --force-recreate
}

# `restart-all` = same scope as `make start-all` (all profiles), but rebuilt.
cmd_restart_all() {
    _compose_with_all_profiles up -d --build --force-recreate
}

cmd_ui() {
    local url="http://localhost:${DOZZLE_HOST_PORT:-9999}"
    echo "Opening $url"
    if command -v xdg-open >/dev/null; then xdg-open "$url" >/dev/null 2>&1
    elif command -v open >/dev/null;     then open "$url"
    elif command -v wslview >/dev/null;  then wslview "$url"
    else echo "No opener found — visit $url manually."; fi
}

interactive() {
    while true; do
        cmd_status
        echo "Actions:"
        echo "  [s] start   [x] stop   [r] restart   [l] logs"
        echo "  [u] up all  [d] down all   [w] open Dozzle UI   [q] quit"
        read -r -p "> " action svc
        case "$action" in
            s) [[ -n "${svc:-}" ]] && cmd_start   "$svc" || echo "usage: s <service>";;
            x) [[ -n "${svc:-}" ]] && cmd_stop    "$svc" || echo "usage: x <service>";;
            r) [[ -n "${svc:-}" ]] && cmd_restart "$svc" || echo "usage: r <service>";;
            l) [[ -n "${svc:-}" ]] && cmd_logs    "$svc" || echo "usage: l <service>";;
            u) cmd_up;;
            d) cmd_down;;
            w) cmd_ui;;
            q|"") return 0;;
            *) echo "unknown: $action";;
        esac
        echo ""
    done
}

main() {
    local cmd="${1:-menu}"; shift || true
    case "$cmd" in
        menu)    interactive;;
        status)  cmd_status;;
        start)   cmd_start   "${1:?service name required}";;
        stop)    cmd_stop    "${1:?service name required}";;
        restart)
            if [[ -n "${1:-}" ]]; then cmd_restart "$1"
            else cmd_restart_base; fi;;
        logs)    cmd_logs    "${1:?service name required}";;
        up)          cmd_up;;
        down)        cmd_down;;
        restart-all) cmd_restart_all;;
        ui)          cmd_ui;;
        services) _all_containers;;
        -h|--help|help)
            sed -n '2,20p' "$0" | sed 's/^# \?//';;
        *) echo "unknown command: $cmd (try --help)"; exit 2;;
    esac
}

main "$@"
