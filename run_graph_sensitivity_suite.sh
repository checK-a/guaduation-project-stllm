#!/usr/bin/env bash
set -euo pipefail

# Graph-structure sensitivity experiments for the revision.
# Mobility/commuting/air-travel graphs are intentionally excluded for now.
#
# Quick smoke example:
#   EPOCHS=5 MIN_EPOCHS=1 ES_PATIENCE=5 SEEDS=6666 HORIZONS=4 DATASETS=ili bash run_graph_sensitivity_suite.sh

DEVICE="${DEVICE:-cuda:0}"
EPOCHS="${EPOCHS:-400}"
MIN_EPOCHS="${MIN_EPOCHS:-400}"
ES_PATIENCE="${ES_PATIENCE:-400}"
BATCH_SIZE="${BATCH_SIZE:-32}"
LRATE="${LRATE:-0.0005}"
SEEDS="${SEEDS:-2024 2025 2026}"
HORIZONS="${HORIZONS:-4 8 12}"
DATASETS="${DATASETS:-ili nhsn}"
GRAPH_K="${GRAPH_K:-4}"
GRAPHS="${GRAPHS:-border identity distance_knn_k${GRAPH_K} correlation_topk_k${GRAPH_K} gravity_topk_k${GRAPH_K}}"
LOG_ROOT="${LOG_ROOT:-graph_sensitivity_logs}"
SAVE_ROOT="${SAVE_ROOT:-./logs/graph_sensitivity_}"

COMMON=(
  --device "${DEVICE}"
  --model epi_st_llm_plus
  --llm_fusion_mode direct
  --ablation_mode full
  --epi_encoder_type llm
  --epi_llm_init pretrained
  --epi_lora_mode lora
  --epi_freeze_gpt false
  --epi_param_generator temporal_cross_attn
  --epi_param_attn_heads 4
  --epochs "${EPOCHS}"
  --min_epochs "${MIN_EPOCHS}"
  --es_patience "${ES_PATIENCE}"
  --batch_size "${BATCH_SIZE}"
  --lrate "${LRATE}"
)

mkdir -p "${LOG_ROOT}"

python build_graph_sensitivity_variants.py --k "${GRAPH_K}"

run_graph() {
  local data_name="$1"
  local seed="$2"
  local graph="$3"
  local graph_path="dataset/${data_name}/${data_name}/graph_variants/adj_${graph}.pkl"
  local log_file="${LOG_ROOT}/${data_name}_seed${seed}_${graph}.log"
  local save_prefix="${SAVE_ROOT}${graph}_seed${seed}_"

  if [[ ! -f "${graph_path}" ]]; then
    echo "Missing graph file: ${graph_path}" >&2
    exit 1
  fi

  echo "===== ${data_name} seed=${seed} graph=${graph} ====="
  python -u train_plus.py "${COMMON[@]}" \
    --data "${data_name}" \
    --seed "${seed}" \
    --adj_override_path "${graph_path}" \
    --save "${save_prefix}" \
    2>&1 | tee "${log_file}"
}

for dataset_family in ${DATASETS}; do
  for H in ${HORIZONS}; do
    if [[ "${dataset_family}" == "ili" ]]; then
      DATA_NAME="ili_us_states_h${H}_leakfree"
    elif [[ "${dataset_family}" == "nhsn" ]]; then
      DATA_NAME="us_states_nhsn_flu_hosp_h${H}_leakfree"
    else
      echo "Unknown dataset family: ${dataset_family}" >&2
      exit 1
    fi

    for SEED in ${SEEDS}; do
      for GRAPH in ${GRAPHS}; do
        run_graph "${DATA_NAME}" "${SEED}" "${GRAPH}"
      done
    done
  done
done
