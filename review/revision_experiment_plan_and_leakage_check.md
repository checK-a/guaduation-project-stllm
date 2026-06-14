# 返修补实验计划与插值泄漏检查

本文档用于规划返修阶段需要补充的实验，并记录当前数据预处理是否存在插值泄漏风险。

## 一、总体实验设置

除特别说明外，所有补实验保持与论文主实验一致：

- 数据集：CDC FluView state-level ILI；NHSN state-level influenza hospitalization。
- 节点：50 states + DC，共 51 个区域。
- 输入长度：24 weeks。
- 预测长度：H = 4, 8, 12 weeks。
- 数据划分：chronological split，train/val/test = 7:1:2。
- 指标：MAE、RMSE、MAPE；新增 WMAPE 可作为附加指标。
- 随机种子：建议至少 3 个 seed，例如 2024、2025、2026；核心表格报告 mean ± std。
- 训练策略：所有模型使用相同 split、early stopping、validation metric、batch size、epoch/patience 上限。
- 汇报原则：不要再写 "consistently outperforms"。改为 "best or competitive in most settings"，并指出哪些指标或 horizon 不是最优。

## 二、必须补充的实验

### 1. LLM / language pretraining 贡献消融

目的：回应 Reviewer 1 对 “为什么没有文本数据还需要预训练语言模型” 的核心质疑。

实验变量：

| Variant | 设置 | 目的 |
|---|---|---|
| Full EpiSTLLM | 当前完整模型，GPT-2 pretrained + LoRA + graph + latent mechanism | 主模型 |
| Random-init GPT-2 | GPT-2 架构相同，但不加载语言预训练权重 | 分离 language pretraining 与 Transformer architecture |
| Vanilla Transformer | 使用同等层数/隐藏维度附近的 Transformer encoder/decoder 替代 GPT-2 | 验证是否只是普通 Transformer 效果 |
| w/o LLM | 移除 GPT-2 表征分支，保留 graph 与 mechanism | 论文已有，但建议多 seed 复跑 |
| Frozen GPT-2 no LoRA | 冻结 GPT-2，不训练 LoRA，仅训练输入/输出头 | 判断 LoRA adaptation 的作用 |

实验设置：

- 数据集：CDC + NHSN。
- Horizon：H = 4, 8, 12。
- Seeds：至少 3 个。
- 指标：MAE/RMSE/MAPE mean ± std。
- 额外记录：trainable parameters、total parameters。

预期写法：

- 如果 Full 不总是最佳，仍可接受；重点是诚实说明 language pretraining 在部分数据集/指标上有收益，但不是所有场景都主导。
- 如果 Random-init GPT-2 接近 Full，则应弱化 “LLM pretraining” 贡献，改写为 “pretrained Transformer backbone / LLM-style sequence representation”。

### 2. 图结构敏感性实验

目的：回应 Reviewer 2 对 state adjacency 不能代表真实传播路径的质疑。

实验变量：

| Graph | 构造方式 | 说明 |
|---|---|---|
| Border adjacency | 当前州接壤图 + self-loop | 主设置 |
| Identity graph | 仅 self-loop | 等价于不使用跨区域传播 |
| Distance-kNN graph | 基于州中心点距离，每州连接 k 个近邻，建议 k=4 或 5 | 地理距离替代接壤 |
| Correlation graph | 仅用训练集历史序列相关性构图，取 top-k 正相关边 | 数据驱动传播 proxy，避免 test 泄漏 |
| Gravity graph | 基于 population_i * population_j / distance^2 | 没有 mobility 数据时的低成本替代 |
| Mobility/commuting/air-travel graph | 若能获取公开数据则加入 | 最贴近审稿意见，但成本最高 |

实验设置：

- 固定模型其余超参数。
- 至少在 Full EpiSTLLM 上跑 CDC + NHSN 的 H = 4, 8, 12。
- 若时间紧，优先跑 H = 12，因为审稿意见关注长 horizon 和真实传播路径。
- 相关性图必须只使用 train split 构建，不能用 val/test。

需要输出：

- 一张 graph sensitivity 表：每种 graph 在 MAE/RMSE/MAPE 上的结果。
- 一段讨论：state adjacency 是一种稀疏、可解释、数据需求低的近似；mobility 信息可能改善传播边权，但也可能引入噪声、时变性和数据可得性问题。

### 3. 不确定性与预测区间

目的：回应 Reviewer 2 指出当前只有 point-estimate metrics。

最低成本方案：

- 对核心模型和强基线进行 3 seed 或 5 seed 重复实验。
- 报告 mean ± std 或 95% confidence interval。

推荐方案：

- 使用 validation residual 做 conformal prediction interval。
- 在 test 上报告：
  - PICP / empirical coverage。
  - MPIW / mean prediction interval width。
  - 可选：Winkler score 或 calibration curve。

实验设置：

- 点预测模型不需要改训练过程。
- 对每个 horizon 单独计算 residual quantile。
- 目标 coverage 建议 90% 或 95%。

需要输出：

- 新增一张 uncertainty 表。
- 方法部分补一句：intervals are post-hoc calibrated using validation residuals，不声称模型本身是 probabilistic forecasting。

### 4. 训练时间、显存和可扩展性

目的：回应 Reviewer 2 对实际公共卫生部署成本的质疑。

需要记录：

| Model | Total params | Trainable params | sec/epoch | Total train time | Peak GPU memory | Inference time |
|---|---:|---:|---:|---:|---:|---:|
| Full EpiSTLLM |  |  |  |  |  |  |
| w/o LLM |  |  |  |  |  |  |
| Vanilla Transformer |  |  |  |  |  |  |

实验设置：

- 硬件：single NVIDIA RTX 4090。
- 统一 batch size、H、dataset。
- 建议至少报告 H=12 的 CDC 和 NHSN；若时间允许，H=4/8/12 都报告。

实现提示：

- 代码中已有 train time / val time 记录。
- 需要补充 `torch.cuda.max_memory_allocated()` 或 `torch.cuda.max_memory_reserved()`。

### 5. Latent SIR mass loss 诊断与修正

目的：回应 Reviewer 1 对 Eq.37 的质疑。审稿人指出：如果 Eq.26 已经严格守恒，则 Eq.37 基本恒为 0，没有梯度。

当前检查结论：

- 代码中 `delta_inf` 和 `delta_rec` 使用 `torch.minimum` 约束，并用
  `S <- S - delta_inf`,
  `I <- I + delta_inf - delta_rec`,
  `R <- R + delta_rec`
  更新。
- 因此 `S + I + R` 按构造近似守恒，`mass_loss` 很可能接近 0。

建议补充实验：

| Variant | 设置 | 目的 |
|---|---|---|
| lambda_mass = 0.01 | 当前设置 | baseline |
| lambda_mass = 0 | 删除 mass loss，只保留参数平滑 | 判断 mass loss 是否有贡献 |
| no mechanism regularizers | lambda_mass = 0, lambda_param = 0 | 判断 regularizer 总体贡献 |

需要记录：

- test MAE/RMSE/MAPE。
- train/val `mass_loss` 的平均值。
- 如果 `mass_loss` 始终接近 0，应删除或弱化 Eq.37，不再把它作为有效 regularizer 声称。

论文修改建议：

- 如果保留质量守恒，应写成 “the rollout is constructed to conserve latent mass”。
- 不建议继续声称 Eq.37 提供重要梯度，除非改成真正非平凡的约束。

### 6. 缺失值插值敏感性实验

目的：回应 Reviewer 1 对 Algorithm 1 中 linear interpolation 是否在 split 前完成、是否泄漏的质疑。

当前检查结论见第三部分。CDC ILI 存在测试窗口插值值，必须处理。

建议实验：

| Dataset version | 设置 | 用途 |
|---|---|---|
| Original preprocessing | 当前版本，完整 panel 上 linear interpolation | 作为旧结果对照 |
| Causal imputation | 对每个样本的 input 只使用该 forecast origin 之前的信息，如 forward fill + training seasonal median | 无未来信息输入 |
| Masked target evaluation | 若 test target 缺失，不插值目标；metric 对缺失 target 位置 mask 掉 | 避免测试标签被未来信息构造 |
| Drop affected samples/states | 删除含缺失 target 的测试样本，或做 sensitivity check | 检查结论是否依赖少量插值点 |

推荐修复策略：

- 输入 `x`：validation/test 样本只能使用 forecast origin 之前的信息填补。
- 目标 `y`：validation/test 中原始缺失位置不要用插值值参与评价；用 mask 排除。
- scaler：继续只基于 train `x` 计算，目前代码已经是 train-only。
- 论文中明确写出缺失处理顺序：split first / train-only imputation statistics / masked evaluation。

需要重跑：

- CDC ILI 的 H=4/8/12 Full EpiSTLLM。
- 至少重跑最强 baselines，例如 Cola-GNN、DCRNN、SIR/SEIR、CausalGNN。
- 若改动后结果变化不大，可作为强回应：结论不依赖插值泄漏。

## 三、当前插值泄漏检查

### 1. 检查依据

检查了以下文件：

- `prepare_cdc_ili.py`
- `prepare_nhsn_flu_us_states.py`
- `dataset/ili_us_states_h4/processed/panel.csv`
- `dataset/ili_us_states_h8/processed/panel.csv`
- `dataset/ili_us_states_h12/processed/panel.csv`
- `dataset/us_states_nhsn_flu_hosp_h4/us_states_nhsn_flu_hosp_h4/meta.json`
- `dataset/us_states_nhsn_flu_hosp_h8/us_states_nhsn_flu_hosp_h8/meta.json`
- `dataset/us_states_nhsn_flu_hosp_h12/us_states_nhsn_flu_hosp_h12/meta.json`

代码层面发现：

- CDC ILI 在构建完整 panel 后执行：
  `series.interpolate(method="linear", limit_direction="both")`，然后再切分 train/val/test。
- NHSN 也在完整 requested matrix 上执行：
  `requested.interpolate(method="linear", limit_direction="both").ffill().bfill()`，然后再切分。
- scaler mean/std 来自 `split_data["train"]["x"][..., 0]`，因此 scaler 是 train-only，不是泄漏点。

### 2. CDC ILI 检查结果

CDC ILI 的 `panel.csv` 中存在 `is_imputed` 标记。统计结果：

- 总插值行数：576。
- 涉及州：
  - Florida：418 行。
  - Louisiana：157 行。
  - District of Columbia：1 行。
- 插值时间跨度：2013W40 到 2022W08。

按 horizon 和 split 统计：

| Dataset | Split | Split coverage | Imputed rows in coverage | In input range | In target range |
|---|---|---:|---:|---:|---:|
| ILI H=4 | train | 2013W40..2020W44 | 527 | 523 | 479 |
| ILI H=4 | val | 2020W21..2021W43 | 72 | 72 | 48 |
| ILI H=4 | test | 2021W20..2023W40 | 21 | 21 | 1 |
| ILI H=8 | train | 2013W40..2020W39 | 522 | 514 | 474 |
| ILI H=8 | val | 2020W16..2021W40 | 77 | 70 | 53 |
| ILI H=8 | test | 2021W17..2023W40 | 24 | 24 | 1 |
| ILI H=12 | train | 2013W40..2020W35 | 518 | 506 | 470 |
| ILI H=12 | val | 2020W12..2021W39 | 81 | 69 | 57 |
| ILI H=12 | test | 2021W16..2023W40 | 25 | 25 | 1 |

结论：

- CDC ILI 当前版本存在插值泄漏风险。
- 原因不是 scaler，而是 test input 和 test target 中包含由完整时间序列线性插值得到的值。
- 尤其 test target 中也有 1 个插值标签位置；即使数量很小，也会被审稿人抓住。
- 建议必须做 leak-free preprocessing sensitivity，并在论文中报告修正后结果。

### 3. NHSN 检查结果

NHSN 的 meta 显示：

- 总缺失值：36。
- 涉及州：MA 17、MN 14、WV 5。
- 缺失日期：2024-05-18 到 2024-10-05。
- 缺失策略：interpolate。

按 horizon 和 split 统计：

| Dataset | Split | Split coverage | Missing/interpolated values in coverage | In input range | In target range |
|---|---|---:|---:|---:|---:|
| NHSN H=4 | train | 2022-02-05..2025-02-01 | 36 | 36 | 36 |
| NHSN H=4 | val | 2024-08-24..2025-06-28 | 16 | 16 | 0 |
| NHSN H=4 | test | 2025-01-18..2026-04-18 | 0 | 0 | 0 |
| NHSN H=8 | train | 2022-02-05..2025-01-04 | 36 | 36 | 36 |
| NHSN H=8 | val | 2024-07-27..2025-06-21 | 24 | 24 | 0 |
| NHSN H=8 | test | 2025-01-11..2026-04-18 | 0 | 0 | 0 |
| NHSN H=12 | train | 2022-02-05..2024-12-07 | 36 | 28 | 36 |
| NHSN H=12 | val | 2024-06-29..2025-06-14 | 24 | 24 | 0 |
| NHSN H=12 | test | 2025-01-04..2026-04-18 | 0 | 0 | 0 |

结论：

- NHSN 当前版本没有直接 test leakage，因为 test input 和 test target 不包含这些缺失/插值点。
- 但 validation input 包含插值点，且插值是在完整 panel 上先做的；从严格审稿角度，仍建议改成 split-aware 或 causal imputation。
- NHSN 的风险等级低于 CDC ILI。

## 四、返修中的建议回应

可以在 response letter 中这样回应：

1. 承认原稿没有充分说明缺失值处理顺序。
2. 明确说明 scaler 只使用训练集 input 计算。
3. 承认原始 CDC ILI preprocessing 中 full-panel interpolation 可能造成 test-window imputed values。
4. 说明已改为 leak-free preprocessing：
   - validation/test input 仅使用 forecast origin 之前的信息填补；
   - validation/test target 的原始缺失位置在 metric 中 mask；
   - 所有 imputation statistics 仅从 train split 得到。
5. 补充修正前后敏感性结果，证明主要结论不依赖该问题。

## 五、推荐执行顺序

1. 先修复 CDC ILI / NHSN 的缺失处理脚本，生成 leak-free 数据包。
2. 复跑 Full EpiSTLLM 在 CDC ILI H=4/8/12 上的结果。
3. 复跑主要强基线，至少覆盖 Cola-GNN、DCRNN、SIR/SEIR、CausalGNN。
4. 跑 LLM/pretraining 消融。
5. 跑 graph sensitivity。
6. 跑 uncertainty / multi-seed。
7. 补 runtime/memory 表。
8. 最后统一改论文措辞、公式、图注、伦理限制和相关工作。

