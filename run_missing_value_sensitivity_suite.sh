#!/usr/bin/env bash
set -euo pipefail

# Missing-value interpolation sensitivity experiments for the revision.
#
# Variants:
#   legacy_interpolate          full-panel linear interpolation package
#   leakfree_point_mask         causal-median input fill + point-wise y_mask
#   leakfree_drop_sample_metric same trained setting, but final test metrics drop samples with any missing target
#   leakfree_drop_node_metric   same trained setting, but final test metrics drop nodes with any missing target
#
# Quick smoke example:
#   EPOCHS=5 MIN_EPOCHS=1 ES_PATIENCE=5 SEEDS=6666 HORIZONS=4 DATASETS=ili VARIANTS=leakfree_point_mask bash run_missing_value_sensitivity_suite.sh

DEVICE="${DEVICE:-cuda:0}"
EPOCHS="${EPOCHS:-400}"
MIN_EPOCHS="${MIN_EPOCHS:-400}"
ES_PATIENCE="${ES_PATIENCE:-400}"
BATCH_SIZE="${BATCH_SIZE:-32}"
LRATE="${LRATE:-0.0005}"
SEEDS="${SEEDS:-2024 2025 2026}"
HORIZONS="${HORIZONS:-4 8 12}"
DATASETS="${DATASETS:-ili nhsn}"
VARIANTS="${VARIANTS:-legacy_interpolate leakfree_point_mask leakfree_drop_sample_metric leakfree_drop_node_metric}"
LOG_ROOT="${LOG_ROOT:-missing_value_sensitivity_logs}"
SAVE_ROOT="${SAVE_ROOT:-./logs/missing_value_sensitivity_}"
BUILD_DATASETS="${BUILD_DATASETS:-true}"

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

if [[ "${BUILD_DATASETS}" == "true" ]]; then
  python build_missing_value_sensitivity_datasets.py --horizons "$(echo "${HORIZONS}" | tr ' ' ',')"
fi

dataset_name_for_variant() {
  local family="$1"
  local horizon="$2"
  local variant="$3"
  local suffix="leakfree"
  if [[ "${variant}" == "legacy_interpolate" ]]; then
    suffix="legacy_interpolate"
  fi

  if [[ "${family}" == "ili" ]]; then
    echo "ili_us_states_h${horizon}_${suffix}"
  elif [[ "${family}" == "nhsn" ]]; then
    echo "us_states_nhsn_flu_hosp_h${horizon}_${suffix}"
  else
    echo "Unknown dataset family: ${family}" >&2
    exit 1
  fi
}

mask_policy_for_variant() {
  local variant="$1"
  if [[ "${variant}" == "leakfree_drop_sample_metric" ]]; then
    echo "drop_sample"
  elif [[ "${variant}" == "leakfree_drop_node_metric" ]]; then
    echo "drop_node"
  else
    echo "point"
  fi
}

run_variant() {
  local data_name="$1"
  local seed="$2"
  local variant="$3"
  local mask_policy="$4"

  local log_file="${LOG_ROOT}/${data_name}_seed${seed}_${variant}.log"
  local save_prefix="${SAVE_ROOT}${variant}_seed${seed}_"

  echo "===== ${data_name} seed=${seed} missing_value_variant=${variant} test_mask=${mask_policy} ====="
  python -u train_plus.py "${COMMON[@]}" \
    --data "${data_name}" \
    --seed "${seed}" \
    --test_y_mask_policy "${mask_policy}" \
    --save "${save_prefix}" \
    2>&1 | tee "${log_file}"
}

for dataset_family in ${DATASETS}; do
  for H in ${HORIZONS}; do
    for SEED in ${SEEDS}; do
      for VARIANT in ${VARIANTS}; do
        DATA_NAME="$(dataset_name_for_variant "${dataset_family}" "${H}" "${VARIANT}")"
        MASK_POLICY="$(mask_policy_for_variant "${VARIANT}")"
        run_variant "${DATA_NAME}" "${SEED}" "${VARIANT}" "${MASK_POLICY}"
      done
    done
  done
done

python collect_missing_value_sensitivity_results.py \
  --root logs \
  --out_csv review/missing_value_sensitivity_results.csv \
  --out_md review/missing_value_sensitivity_results.md
