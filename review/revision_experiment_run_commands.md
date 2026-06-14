# 返修六组实验运行命令与合并建议

本文档汇总 `review/revision_experiment_plan_and_leakage_check.md` 中六组补充实验的服务器运行命令，并说明哪些实验可以合并运行、哪些需要单独训练。

默认脚本均使用 leak-free 数据包，默认覆盖 CDC ILI / NHSN、H=4/8/12、3 seeds。正式运行前建议先做 smoke test。

## 0. 前置数据检查

确认 leak-free 数据包与 legacy interpolation 对照包存在：

```bash
python check_leakfree_datasets.py
python build_missing_value_sensitivity_datasets.py --horizons 4,8,12
```

输出：

- `review/leakfree_dataset_check_report.md`
- `dataset/*_legacy_interpolate/`

## 1. LLM / Language Pretraining 贡献消融

目的：回应 LLM 预训练到底是否有贡献。

Smoke test：

```bash
EPOCHS=5 MIN_EPOCHS=1 ES_PATIENCE=5 SEEDS=6666 HORIZONS=4 DATASETS=ili bash run_llm_pretraining_ablation_suite.sh
```

正式运行：

```bash
bash run_llm_pretraining_ablation_suite.sh
```

默认变体：

- `full_pretrained_lora`
- `random_init_gpt2`
- `vanilla_transformer`
- `without_llm`
- `frozen_gpt2_no_lora`

输出：

- `review/llm_pretraining_ablation_results.csv`
- `review/llm_pretraining_ablation_results.md`

## 2. 图结构敏感性实验

目的：回应模型是否依赖某一种手工图结构。暂不包括 Mobility / commuting / air-travel graph。

Smoke test：

```bash
EPOCHS=5 MIN_EPOCHS=1 ES_PATIENCE=5 SEEDS=6666 HORIZONS=4 DATASETS=ili bash run_graph_sensitivity_suite.sh
```

正式运行：

```bash
bash run_graph_sensitivity_suite.sh
```

默认图：

- `border`
- `identity`
- `distance_knn_k4`
- `correlation_topk_k4`
- `gravity_topk_k4`

输出：

- `review/graph_sensitivity_variant_report.md`
- `review/graph_sensitivity_results.csv`
- `review/graph_sensitivity_results.md`

## 3. 不确定性与预测区间

目的：补充 split-conformal prediction intervals。注意这里不是声称模型是 probabilistic forecasting，而是基于验证集 residuals 做 conformal calibration。

Smoke test：

```bash
EPOCHS=5 MIN_EPOCHS=1 ES_PATIENCE=5 SEEDS=6666 HORIZONS=4 DATASETS=ili bash run_uncertainty_interval_suite.sh
```

正式运行：

```bash
bash run_uncertainty_interval_suite.sh
```

输出：

- 每个 run 目录下的 `conformal_intervals.csv`
- `review/conformal_interval_results.csv`
- `review/conformal_interval_results.md`

## 4. 训练时间、显存和可扩展性

目的：回应部署成本、训练时间、显存占用和 horizon 扩展性。

Smoke test：

```bash
EPOCHS=5 MIN_EPOCHS=1 ES_PATIENCE=5 SEEDS=6666 HORIZONS=4 DATASETS=ili bash run_resource_profile_suite.sh
```

正式运行：

```bash
bash run_resource_profile_suite.sh
```

默认变体：

- `full`
- `without_llm`
- `vanilla_transformer`

输出：

- 每个 run 目录下的 `resource_report.csv`
- 每个 run 目录下的 `resource_report.json`
- `review/resource_profile_results.csv`
- `review/resource_profile_results.md`

## 5. Latent SIR Mass Loss 诊断与修正

目的：回应 Eq.37 是否恒为 0、是否有有效梯度的问题。

Smoke test：

```bash
EPOCHS=5 MIN_EPOCHS=1 ES_PATIENCE=5 SEEDS=6666 HORIZONS=4 DATASETS=ili VARIANTS=lambda_mass_0 bash run_sir_mass_ablation_suite.sh
```

正式运行：

```bash
bash run_sir_mass_ablation_suite.sh
```

默认变体：

- `lambda_mass_0p01`: 当前设置，`lambda_mass=0.01, lambda_param=0.01`
- `lambda_mass_0`: 删除 mass loss，只保留参数平滑
- `no_mech_regularizers`: 删除 mass loss 和参数平滑

输出：

- 每个 run 目录下的 `sir_diagnostics.csv`
- `review/sir_mass_ablation_results.csv`
- `review/sir_mass_ablation_results.md`

## 6. 缺失值插值敏感性实验

目的：回应 full-panel interpolation 是否泄漏，以及修正后结论是否稳定。

Smoke test：

```bash
EPOCHS=5 MIN_EPOCHS=1 ES_PATIENCE=5 SEEDS=6666 HORIZONS=4 DATASETS=ili VARIANTS=leakfree_point_mask bash run_missing_value_sensitivity_suite.sh
```

正式运行：

```bash
bash run_missing_value_sensitivity_suite.sh
```

默认变体：

- `legacy_interpolate`: 原始 full-panel linear interpolation 对照
- `leakfree_point_mask`: causal imputation + point-wise target mask
- `leakfree_drop_sample_metric`: 测试时丢弃含缺失 target 的样本
- `leakfree_drop_node_metric`: 测试时丢弃含缺失 target 的节点

输出：

- `review/missing_value_sensitivity_results.csv`
- `review/missing_value_sensitivity_results.md`

## 哪些实验可以合并跑

可以合并的部分：

| 可合并项 | 建议 |
|---|---|
| 第 3 组 conformal intervals | 可以和任意 full-model run 合并，只需加 `--eval_conformal_intervals true` |
| 第 4 组 resource profiling | 可以和任意训练合并，只需加 `--profile_resources true` |
| 第 5 组 baseline mass 诊断 | `lambda_mass=0.01` 这一项可以和 full-model run 合并，只需加 `--eval_sir_diagnostics true` |
| 第 1 组 `full_pretrained_lora` | 等价于主模型 full run，可复用主模型结果 |
| 第 2 组 `border` 图 | 等价于默认边界图 full run，可复用主模型结果 |
| 第 6 组 `leakfree_point_mask` | 等价于 leak-free 主模型 full run，可复用主模型结果 |

不建议合并、需要单独训练的部分：

| 实验 | 原因 |
|---|---|
| 第 1 组 random init / transformer / no LLM / frozen GPT | 模型结构或初始化改变，需要重新训练 |
| 第 2 组 identity / distance / correlation / gravity graph | 图进入训练过程，需要重新训练 |
| 第 5 组 `lambda_mass=0` / `lambda_mass=lambda_param=0` | loss 改变，需要重新训练 |
| 第 6 组 `legacy_interpolate` | 数据构造改变，需要重新训练 |

推荐的省时合并方式是：单独安排一次主模型 full run，同时打开三类只读诊断：

```bash
python -u train_plus.py \
  --device cuda:0 \
  --model epi_st_llm_plus \
  --data ili_us_states_h12_leakfree \
  --seed 2024 \
  --llm_fusion_mode direct \
  --ablation_mode full \
  --epi_encoder_type llm \
  --epi_llm_init pretrained \
  --epi_lora_mode lora \
  --epi_freeze_gpt false \
  --epi_param_generator temporal_cross_attn \
  --epi_param_attn_heads 4 \
  --eval_conformal_intervals true \
  --conformal_coverages 0.9,0.95 \
  --profile_resources true \
  --eval_sir_diagnostics true \
  --sir_diagnostic_splits train,val,test \
  --epochs 400 \
  --min_epochs 400 \
  --es_patience 400 \
  --batch_size 32 \
  --lrate 0.0005 \
  --save ./logs/merged_full_diagnostics_
```

但为了减少管理复杂度，目前各 suite 脚本是自包含的，会有少量重复训练。正式赶时间时，优先合并第 3/4/5 的只读诊断。

## 基线模型是否需要重跑

需要重跑。原因是主数据包已经从 legacy interpolation 改为 leak-free causal imputation + masked target evaluation，主表里所有和 EpiSTLLM 比较的 baseline 都应在同一套 `_leakfree` 数据和同一套 masked metric 下重新评估。

建议至少重跑论文主表中的强基线：

- `cola_gnn`
- `DCRNN`
- `CausalGNN`
- `SIR`
- `SEIR`
- 如果主表包含，也重跑 `GRU` / `LSTM` / `PatchTST` / `STGCN` / `AR` / `VAR`

不需要把所有 baseline 都放进六组 sensitivity。推荐分工是：

- 主性能表：EpiSTLLM + 所有 baseline，全部用 `_leakfree` 数据重跑。
- 第 1-6 组返修敏感性实验：主要跑 EpiSTLLM full 和对应变体。
- 第 6 组 legacy vs leak-free：用 EpiSTLLM full 即可说明原数据处理敏感性；除非审稿人特别要求，不必对所有 baseline 重做 legacy 对照。

## 建议执行顺序

1. 先跑 smoke tests，确认服务器依赖、GPT-2 权重、CUDA 都正常。
2. 跑 leak-free 主模型 full run，可同时打开 conformal/resource/SIR diagnostics。
3. 重跑主表 baseline，得到新的公平比较表。
4. 跑 LLM/pretraining 消融。
5. 跑 graph sensitivity。
6. 跑 mass-loss ablation 中两个非 baseline 变体。
7. 跑 missing-value sensitivity 的 legacy/drop-sample/drop-node 对照。
8. 汇总所有 `review/*_results.md`，再写 response letter 和论文修订。
