#!/usr/bin/env bash
set -euo pipefail

# Latent SIR mass-loss diagnosis and correction ablations for the revision.
#
# Variants:
#   lambda_mass_0p01       current setting: lambda_mass=0.01, lambda_param=0.01
#   lambda_mass_0          removes Eq.37 mass loss, keeps beta/gamma smoothness
#   no_mech_regularizers   removes both mass loss and parameter smoothness
#
# Quick smoke example:
#   EPOCHS=5 MIN_EPOCHS=1 ES_PATIENCE=5 SEEDS=6666 HORIZONS=4 DATASETS=ili VARIANTS=lambda_mass_0 bash run_sir_mass_ablation_suite.sh

DEVICE="${DEVICE:-cuda:0}"
EPOCHS="${EPOCHS:-400}"
MIN_EPOCHS="${MIN_EPOCHS:-400}"
ES_PATIENCE="${ES_PATIENCE:-400}"
BATCH_SIZE="${BATCH_SIZE:-32}"
LRATE="${LRATE:-0.0005}"
SEEDS="${SEEDS:-2024 2025 2026}"
HORIZONS="${HORIZONS:-4 8 12}"
DATASETS="${DATASETS:-ili nhsn}"
VARIANTS="${VARIANTS:-lambda_mass_0p01 lambda_mass_0 no_mech_regularizers}"
LOG_ROOT="${LOG_ROOT:-sir_mass_ablation_logs}"
SAVE_ROOT="${SAVE_ROOT:-./logs/sir_mass_ablation_}"

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
  --eval_sir_diagnostics true
  --sir_diagnostic_splits train,val,test
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

  echo "===== ${data_name} seed=${seed} sir_mass_variant=${variant} ====="
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
        if [[ "${VARIANT}" == "lambda_mass_0p01" ]]; then
          run_variant "${DATA_NAME}" "${SEED}" "${VARIANT}" \
            --lambda_mass 0.01 \
            --lambda_param 0.01
        elif [[ "${VARIANT}" == "lambda_mass_0" ]]; then
          run_variant "${DATA_NAME}" "${SEED}" "${VARIANT}" \
            --lambda_mass 0.0 \
            --lambda_param 0.01
        elif [[ "${VARIANT}" == "no_mech_regularizers" ]]; then
          run_variant "${DATA_NAME}" "${SEED}" "${VARIANT}" \
            --lambda_mass 0.0 \
            --lambda_param 0.0
        else
          echo "Unknown variant: ${VARIANT}" >&2
          exit 1
        fi
      done
    done
  done
done

python collect_sir_mass_ablation_results.py \
  --root logs \
  --out_csv review/sir_mass_ablation_results.csv \
  --out_md review/sir_mass_ablation_results.md
