#!/usr/bin/env bash
set -euo pipefail

# LLM / language-pretraining contribution ablations for the revision.
#
# Defaults run both CDC ILI and NHSN leak-free datasets across H=4/8/12 and
# three seeds. Override from the shell for quick smoke tests, e.g.:
#   EPOCHS=5 MIN_EPOCHS=1 ES_PATIENCE=5 SEEDS=6666 HORIZONS=4 DATASETS=ili bash run_llm_pretraining_ablation_suite.sh

DEVICE="${DEVICE:-cuda:0}"
EPOCHS="${EPOCHS:-400}"
MIN_EPOCHS="${MIN_EPOCHS:-400}"
ES_PATIENCE="${ES_PATIENCE:-400}"
BATCH_SIZE="${BATCH_SIZE:-32}"
LRATE="${LRATE:-0.0005}"
SEEDS="${SEEDS:-2024 2025 2026}"
HORIZONS="${HORIZONS:-4 8 12}"
DATASETS="${DATASETS:-ili nhsn}"
LOG_ROOT="${LOG_ROOT:-llm_pretraining_ablation_logs}"
SAVE_ROOT="${SAVE_ROOT:-./logs/llm_pretraining_ablation_}"

COMMON=(
  --device "${DEVICE}"
  --model epi_st_llm_plus
  --llm_fusion_mode direct
  --epi_param_generator temporal_cross_attn
  --epi_param_attn_heads 4
  --epochs "${EPOCHS}"
  --min_epochs "${MIN_EPOCHS}"
  --es_patience "${ES_PATIENCE}"
  --batch_size "${BATCH_SIZE}"
  --lrate "${LRATE}"
)

mkdir -p "${LOG_ROOT}"

run_variant() {
  local data_name="$1"
  local seed="$2"
  local variant="$3"
  shift 3

  local log_file="${LOG_ROOT}/${data_name}_seed${seed}_${variant}.log"
  local save_prefix="${SAVE_ROOT}${variant}_seed${seed}_"

  echo "===== ${data_name} seed=${seed} variant=${variant} ====="
  python -u train_plus.py "${COMMON[@]}" \
    --data "${data_name}" \
    --seed "${seed}" \
    --save "${save_prefix}" \
    "$@" 2>&1 | tee "${log_file}"
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
      run_variant "${DATA_NAME}" "${SEED}" "full_pretrained_lora" \
        --ablation_mode full \
        --epi_encoder_type llm \
        --epi_llm_init pretrained \
        --epi_lora_mode lora \
        --epi_freeze_gpt false

      run_variant "${DATA_NAME}" "${SEED}" "random_init_gpt2" \
        --ablation_mode full \
        --epi_encoder_type llm \
        --epi_llm_init random \
        --epi_lora_mode lora \
        --epi_freeze_gpt false

      run_variant "${DATA_NAME}" "${SEED}" "vanilla_transformer" \
        --ablation_mode full \
        --epi_encoder_type transformer

      run_variant "${DATA_NAME}" "${SEED}" "without_llm" \
        --ablation_mode no_llm \
        --epi_encoder_type llm \
        --epi_llm_init pretrained \
        --epi_lora_mode lora \
        --epi_freeze_gpt false

      run_variant "${DATA_NAME}" "${SEED}" "frozen_gpt2_no_lora" \
        --ablation_mode full \
        --epi_encoder_type llm \
        --epi_llm_init pretrained \
        --epi_lora_mode none \
        --epi_freeze_gpt true
    done
  done
done
