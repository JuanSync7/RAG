#!/bin/sh
# Durable, portable GOMEMLIMIT for Weaviate.
# Caps the Go heap at a fraction of the memory THIS container can actually use,
# computed at startup. Works on cgroup v1 rootless (where mem_limit is ignored,
# e.g. ai03), cgroup v2, and orchestrated deployments alike.
# Single policy knob: RAG_WEAVIATE_MEM_FRACTION = percent of detected RAM (default 50).
LIM=""
[ -r /sys/fs/cgroup/memory.max ] && LIM=$(cat /sys/fs/cgroup/memory.max)
if [ -z "$LIM" ] || [ "$LIM" = max ]; then
  [ -r /sys/fs/cgroup/memory/memory.limit_in_bytes ] && LIM=$(cat /sys/fs/cgroup/memory/memory.limit_in_bytes)
fi
# unset / "max" / absurd v1 "unlimited" -> fall back to host MemTotal
if [ -z "$LIM" ] || [ "$LIM" = max ] || { [ "$LIM" -gt 1000000000000000 ] 2>/dev/null; }; then
  LIM=$(awk '/MemTotal/{print $2*1024}' /proc/meminfo)
fi
FRAC=${RAG_WEAVIATE_MEM_FRACTION:-50}
export GOMEMLIMIT=$(( LIM * FRAC / 100 ))
export GOGC=${RAG_WEAVIATE_GOGC:-75}
echo "[weaviate-mem] GOMEMLIMIT=${GOMEMLIMIT}B GOGC=${GOGC} (frac=${FRAC}% of detected ${LIM}B)"
exec /bin/weaviate "$@"
