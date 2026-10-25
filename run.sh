#!/bin/bash
# Launch training and inference with public config resolution owned by Python.
set -euo pipefail
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${PROJECT_DIR}"

action="${1:-}"
config="${2:-}"
if [[ "$action" == -h || "$action" == --help ]]; then
    echo 'Usage: bash run.sh <train|test|test_all> <config> [entry arguments or key=value ...]'
    exit 0
fi
case "$action" in
    train|test|test_all) ;;
    *) echo "Unknown action '${action}'; expected train, test or test_all." >&2; exit 1 ;;
esac
if [[ -z "$config" ]]; then
    echo 'A configuration stem or path is required.' >&2
    exit 1
fi
shift 2
extra_args=("$@")

# Use the activated conda/venv interpreter throughout, including torchrun.
# Without an activated environment, retain the uv-managed project default.
runtime="${LOOKING_AHEAD_ENV:-auto}"
if [[ "$runtime" == auto ]]; then
    if [[ -n "${CONDA_PREFIX:-}" || -n "${VIRTUAL_ENV:-}" ]]; then
        runtime=active
    else
        runtime=uv
    fi
fi
case "$runtime" in
    active)
        if [[ -z "${CONDA_PREFIX:-}" && -z "${VIRTUAL_ENV:-}" ]]; then
            echo 'Activate your conda or virtual environment before using LOOKING_AHEAD_ENV=active.' >&2
            exit 1
        fi
        python_cmd=(python) ;;
    uv)
        if ! command -v uv &>/dev/null; then
            echo 'Activate the project conda/virtual environment, or install uv.' >&2
            exit 1
        fi
        python_cmd=(uv run python) ;;
    *)
        echo 'LOOKING_AHEAD_ENV must be auto, active or uv.' >&2
        exit 1 ;;
esac

get_num_gpus() {
    if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
        local devices
        IFS=',' read -ra devices <<< "$CUDA_VISIBLE_DEVICES"
        echo "${#devices[@]}"
    elif command -v nvidia-smi &>/dev/null; then
        local gpu_names
        gpu_names="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || true)"
        if [[ -n "$gpu_names" ]]; then
            printf '%s\n' "$gpu_names" | awk 'NF {count++} END {print count + 0}'
        else
            echo 1
        fi
    else
        echo 1
    fi
}

build_rdzv_id() {
    local requested_config="$2" canonical_config readable_config digest
    if [[ "$requested_config" == */configs/* ]]; then
        canonical_config="${requested_config##*/configs/}"
    else
        canonical_config="$(basename "$requested_config")"
    fi
    readable_config="${canonical_config%.yaml}"
    readable_config="$(printf '%s' "$readable_config" | tr -c '[:alnum:]_.-' '_')"
    digest="$(printf '%s' "$1:$canonical_config" | sha256sum)"
    digest="${digest%% *}"
    printf '%s-%s-%s' "$1" "${readable_config:0:72}" "${digest:0:12}"
}

positive_integer() {
    if [[ ! "$2" =~ ^[1-9][0-9]*$ ]]; then
        echo "$1 must be a positive integer; got '$2'." >&2
        exit 1
    fi
}

export DEBUG="${DEBUG:-0}"
export WAN_VIS_CAUSAL_MASK="${WAN_VIS_CAUSAL_MASK:-0}"
if [[ "$DEBUG" == 1 ]]; then
    export WAN_VIS_CAUSAL_MASK=1
fi
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS-8}"
positive_integer OMP_NUM_THREADS "$OMP_NUM_THREADS"
NUM_GPUS=$(get_num_gpus)
WORLD_SIZE="${WORLD_SIZE:-1}"
RANK="${RANK:-0}"
MASTER_ADDR="${MASTER_ADDR:-localhost}"
MASTER_PORT="${MASTER_PORT:-29500}"
positive_integer WORLD_SIZE "$WORLD_SIZE"
if [[ ! "$RANK" =~ ^(0|[1-9][0-9]*)$ ]] || (( RANK >= WORLD_SIZE )); then
    echo 'RANK must be a nonnegative node rank below WORLD_SIZE.' >&2
    exit 1
fi
if [[ "$DEBUG" == 1 ]] && (( WORLD_SIZE > 1 )); then
    echo 'DEBUG=1 requires WORLD_SIZE=1; multi-node standalone jobs are unsupported.' >&2
    exit 1
fi
if (( WORLD_SIZE > 1 )); then
    RDZV_TIMEOUT_SEC="${RDZV_TIMEOUT_SEC-86400}"
    positive_integer RDZV_TIMEOUT_SEC "$RDZV_TIMEOUT_SEC"
    case "${MASTER_ADDR,,}" in
        localhost|localhost.*|127.*|0.0.0.0|::1|\[::1\])
            echo 'MASTER_ADDR must be a routable rank-0 address for multi-node runs.' >&2
            exit 1 ;;
    esac
fi

if [[ "$action" == train ]]; then
    cmd=(train.py --config_path "$config" "${extra_args[@]}")
else
    cmd=(--module src.utils.launcher_entry "$action" --config_path "$config" "${extra_args[@]}")
fi

# Help/config inspection does not allocate logs, start workers or initialize CUDA.
for arg in "${extra_args[@]}"; do
    case "$arg" in
        -h|--help|--dry_run)
            if [[ "$action" == train ]]; then
                exec "${python_cmd[@]}" "${cmd[@]}"
            else
                exec "${python_cmd[@]}" -m src.utils.launcher_entry "$action" \
                    --config_path "$config" "${extra_args[@]}"
            fi ;;
    esac
done

torch_args=("--nproc_per_node=$NUM_GPUS")
if [[ "$DEBUG" == 1 ]]; then
    torch_args+=(--standalone)
else
    timeout=36000
    if (( WORLD_SIZE > 1 )); then timeout="$RDZV_TIMEOUT_SEC"; fi
    torch_args+=("--nnodes=$WORLD_SIZE" "--node_rank=$RANK" --rdzv_backend=c10d
                 "--rdzv-endpoint=$MASTER_ADDR:$MASTER_PORT"
                 "--rdzv_id=$(build_rdzv_id "$action" "$config")"
                 "--rdzv-conf=join_timeout=$timeout")
fi

if [[ "$action" == train ]]; then
    TRAIN_LOG_DIR="$(WORLD_SIZE=$(( WORLD_SIZE * NUM_GPUS )) RANK="$RANK" \
        "${python_cmd[@]}" -m src.utils.train_cli --config_path "$config" "${extra_args[@]}")"
    exec > >(tee -- "$TRAIN_LOG_DIR/launcher.log") 2>&1
    echo "Training launcher logs: $TRAIN_LOG_DIR"
    echo "Node rank: $RANK; nodes: $WORLD_SIZE; GPUs per node: $NUM_GPUS"
    exec "${python_cmd[@]}" -m torch.distributed.run "${torch_args[@]}" --redirects 3 --tee 3 \
        --log-dir "$TRAIN_LOG_DIR" "${cmd[@]}"
else
    if [[ "$DEBUG" != 1 ]]; then
        torch_args+=(--redirects 3 --tee 3 --log-dir .torchrun_logs)
    fi
    exec "${python_cmd[@]}" -m torch.distributed.run "${torch_args[@]}" "${cmd[@]}"
fi
