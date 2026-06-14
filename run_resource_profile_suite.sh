#!/usr/bin/env bash
set -euo pipefail

# Training-time, GPU-memory, and scalability profiling for the revision.
#
# Defaults profile CDC ILI and NHSN leak-free datasets across H=4/8/12 with
# one seed. Override from the shell for full repeated runs or quick smoke tests:
#   EPOCHS=5 MIN_EPOCHS=1 ES_PATIENCE=5 SEEDS=6666 HORIZONS=4 DATASETS=ili bash run_resource_profile_suite.sh
#   SEEDS="2024 2025 2026" VARIANTS="full without_llm vanilla_transformer" bash run_resource_profile_suite.sh

DEVICE="${DEVICE:-cuda:0}"
EPOCHS="${EPOCHS:-400}"
MIN_EPOCHS="${MIN_EPOCHS:-400}"
ES_PATIENCE="${ES_PATIENCE:-400}"
BATCH_SIZE="${BATCH_SIZE:-32}"
LRATE="${LRATE:-0.0005}"
SEEDS="${SEEDS:-6666}"
HORIZONS="${HORIZONS:-4 8 12}"
DATASETS="${DATASETS:-ili nhsn}"
VARIANTS="${VARIANTS:-full without_llm vanilla_transformer}"
LOG_ROOT="${LOG_ROOT:-resource_profile_logs}"
SAVE_ROOT="${SAVE_ROOT:-./logs/resource_profile_}"

COMMON=(
  --device "${DEVICE}"
  --model epi_st_llm_plus
  --llm_fusion_mode direct
  --epi_param_generator temporal_cross_attn
  --epi_param_attn_heads 4
  --profile_resources true
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

  echo "===== ${data_name} seed=${seed} resource_profile=${variant} ====="
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
      for VARIANT in ${VARIANTS}; do
        if [[ "${VARIANT}" == "full" ]]; then
          run_variant "${DATA_NAME}" "${SEED}" "${VARIANT}" \
            --ablation_mode full \
            --epi_encoder_type llm \
            --epi_llm_init pretrained \
            --epi_lora_mode lora \
            --epi_freeze_gpt false
        elif [[ "${VARIANT}" == "without_llm" ]]; then
          run_variant "${DATA_NAME}" "${SEED}" "${VARIANT}" \
            --ablation_mode no_llm \
            --epi_encoder_type llm \
            --epi_llm_init pretrained \
            --epi_lora_mode lora \
            --epi_freeze_gpt false
        elif [[ "${VARIANT}" == "vanilla_transformer" ]]; then
          run_variant "${DATA_NAME}" "${SEED}" "${VARIANT}" \
            --ablation_mode full \
            --epi_encoder_type transformer
        elif [[ "${VARIANT}" == "random_init_gpt2" ]]; then
          run_variant "${DATA_NAME}" "${SEED}" "${VARIANT}" \
            --ablation_mode full \
            --epi_encoder_type llm \
            --epi_llm_init random \
            --epi_lora_mode lora \
            --epi_freeze_gpt false
        else
          echo "Unknown variant: ${VARIANT}" >&2
          exit 1
        fi
      done
    done
  done
done

python collect_resource_profile_results.py \
  --root logs \
  --out_csv review/resource_profile_results.csv \
  --out_md review/resource_profile_results.md
