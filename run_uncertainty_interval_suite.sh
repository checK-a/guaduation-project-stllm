#!/usr/bin/env bash
set -euo pipefail

# Full-model point uncertainty and split-conformal prediction interval suite.
# Calibrates intervals on validation residuals and evaluates PICP/MPIW/Winkler on test.
#
# Quick smoke example:
#   EPOCHS=5 MIN_EPOCHS=1 ES_PATIENCE=5 SEEDS=6666 HORIZONS=4 DATASETS=ili bash run_uncertainty_interval_suite.sh

DEVICE="${DEVICE:-cuda:0}"
EPOCHS="${EPOCHS:-400}"
MIN_EPOCHS="${MIN_EPOCHS:-400}"
ES_PATIENCE="${ES_PATIENCE:-400}"
BATCH_SIZE="${BATCH_SIZE:-32}"
LRATE="${LRATE:-0.0005}"
SEEDS="${SEEDS:-2024 2025 2026}"
HORIZONS="${HORIZONS:-4 8 12}"
DATASETS="${DATASETS:-ili nhsn}"
COVERAGES="${COVERAGES:-0.9,0.95}"
LOG_ROOT="${LOG_ROOT:-uncertainty_interval_logs}"
SAVE_ROOT="${SAVE_ROOT:-./logs/uncertainty_interval_}"

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
  --eval_conformal_intervals true
  --conformal_coverages "${COVERAGES}"
  --epochs "${EPOCHS}"
  --min_epochs "${MIN_EPOCHS}"
  --es_patience "${ES_PATIENCE}"
  --batch_size "${BATCH_SIZE}"
  --lrate "${LRATE}"
)

mkdir -p "${LOG_ROOT}"

run_dataset() {
  local data_name="$1"
  local seed="$2"
  local log_file="${LOG_ROOT}/${data_name}_seed${seed}.log"
  local save_prefix="${SAVE_ROOT}seed${seed}_"

  echo "===== ${data_name} seed=${seed} conformal intervals ====="
  python -u train_plus.py "${COMMON[@]}" \
    --data "${data_name}" \
    --seed "${seed}" \
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
      run_dataset "${DATA_NAME}" "${SEED}"
    done
  done
done

python collect_conformal_interval_results.py \
  --root logs \
  --out_csv review/conformal_interval_results.csv \
  --out_md review/conformal_interval_results.md
