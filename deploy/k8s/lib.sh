# Shared helpers for run.sh / bench.sh. Source, don't execute.
#
# Inputs (env): BOOK -- a file inside deploy/k8s/data/ (default bench.txt).

set -euo pipefail

K8S_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$K8S_DIR/../.." && pwd)"
export DATA_DIR="$K8S_DIR/data"
export BOOK="${BOOK:-bench.txt}"
export STEM="${BOOK%.*}"
export HOST_UID="$(id -u)" HOST_GID="$(id -g)"
NS=alebrijes
CLUSTER=alebrijes
IMAGE=alebrijes:dev
SUBST_VARS='${DATA_DIR} ${BOOK} ${STEM} ${HOST_UID} ${HOST_GID} ${CHAPTERS} ${PARALLELISM}'

log() { printf '\n[%s] %s\n' "$(date +%T)" "$*"; }

render_manifest() { envsubst "$SUBST_VARS" < "$K8S_DIR/$1"; }

ensure_cluster() {
  mkdir -p "$DATA_DIR"
  if ! kind get clusters 2>/dev/null | grep -qx "$CLUSTER"; then
    log "creating kind cluster '$CLUSTER' (1 control plane + 3 workers)"
    kind create cluster --config <(render_manifest kind-config.yaml)
  fi
  kubectl config use-context "kind-$CLUSTER" >/dev/null
  render_manifest pv.yaml | kubectl apply -f - >/dev/null
}

build_image() {
  log "building $IMAGE"
  docker build -f "$REPO_ROOT/deploy/docker/Dockerfile" -t "$IMAGE" "$REPO_ROOT"
}

load_image() {
  log "loading $IMAGE into every kind node"
  kind load docker-image "$IMAGE" --name "$CLUSTER"
}

chapter_count() {
  docker run --rm -v "$DATA_DIR:/data" "$IMAGE" chapters "/data/$BOOK" | tail -1
}

# wait_job NAME -- block until the Job completes; dump logs and fail if it fails.
wait_job() {
  local name=$1 conditions
  while true; do
    conditions=$(kubectl -n "$NS" get job "$name" \
      -o jsonpath='{range .status.conditions[*]}{.type}={.status}{"\n"}{end}')
    if grep -q '^Complete=True' <<<"$conditions"; then return 0; fi
    # A pod that can never start (e.g. image not loaded into kind) leaves
    # the Job neither Complete nor Failed -- fail fast instead of hanging.
    if kubectl -n "$NS" get pods -l "job-name=$name" \
        -o jsonpath='{.items[*].status.containerStatuses[*].state.waiting.reason}' \
        | grep -qE 'ErrImageNeverPull|ImagePullBackOff|ErrImagePull|InvalidImageName'; then
      kubectl -n "$NS" get pods -l "job-name=$name"
      echo "Job $name: pods cannot start (image missing from kind nodes? run load_image)" >&2
      return 1
    fi
    if grep -q '^Failed=True' <<<"$conditions"; then
      kubectl -n "$NS" logs "job/$name" --tail=40 || true
      return 1
    fi
    sleep 3
  done
}

# run_job MANIFEST NAME -- (re)create a Job, wait for it, print its wall time.
run_job() {
  local manifest=$1 name=$2 start
  kubectl -n "$NS" delete job "$name" --ignore-not-found --wait=true >/dev/null
  start=$(date +%s)
  render_manifest "$manifest" | kubectl apply -f - >/dev/null
  wait_job "$name"
  JOB_SECONDS=$(( $(date +%s) - start ))
  log "$name finished in ${JOB_SECONDS}s"
}

audio_seconds() {
  python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['chapters'][-1]['end_ms'] / 1000)" \
    "$DATA_DIR/out/$STEM.transcript.json"
}
