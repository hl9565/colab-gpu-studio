# Agent Note: 纯采样冷基线估时器与严格截止时间收紧

Status: implemented

## Problem

原有 `rates.py` 通过将 `SAMPLING_SHARE = 0.35`（隐含 65% 固定启动权重）混乘进 `api/app/gpu/rates.py` 中的 `length_factor`（L67–L77）来归一化历史总耗时 `expected_minutes`。一旦历史样本混入冷启动恢复权重或参数倍率偏离默认档位，采样耗时被固定权重扭曲；同时缺失 `None` 安全收紧与 batch `following` 交接检查时，短租约截止时间可能被放宽或触发 `min()` 空序列异常。

## Decision

PR4 的复用准入与剩余时间估算独立使用 `api/app/gpu/estimator.py`，停用 `rates.length_factor` 与 `rates.expected_minutes` 参与调度准入决策：

- **可加性冷基线 + 纯采样分解**：总需要时长按固定阶段叠加计算 `COMFY_START_S + restore_s + sampling_s + TAIL_S + SAFETY_MARGIN_S`（常量 `COMFY_START_S = 30.0`、`TAIL_S = 60.0`、`SAFETY_MARGIN_S = 60.0`）；`api/app/gpu/estimator.py` 中的 `env_weights_restore_seconds`（L184–L190）在 verified warm（须通过 [暖机证明校验](./2026-11-16-warm-proof-5tuple-and-cas-fences.md)）时为 `0.0`，否则图像为 `180.0` 秒、视频或未知模态为 `330.0` 秒。
- **只紧不松截止时间收紧**：`api/app/gpu/estimator.py` 中的 `min_deadlines`（L169–L181）过滤 `None`、校验数值有限性并仅允许向更早截止时间收紧；全 `None` 返回 `None`，禁止空参数调用 `min()` 或将 `0` 视作无期限。
- **Batch `following` 交接审校**：batch handoff 严格考虑 `following` 队列后续任务需求，与 `check_job_reuse_guard` 一并决定保留或回收实例。

被否方案记录于 [放弃乐观暖机标记与混乘 length_factor](../../rejected/architecture/2026-11-16-reuse-rates-length-factor-and-optimistic-warm.md)。

## Alternatives considered

- **继续微调 `rates.py` 的 `SAMPLING_SHARE` / `LENGTH_EXPONENT` 参数并复用 `expected_minutes`** — 优势是无需维护单独的 `estimator.py` 且沿用既有费用估算接口；但它把一次性的环境/权重恢复惩罚固定按 `0.65` 比例揉进整条曲线，无法根据同代 `warm_proof` 状态分别归零或叠加 `180s/330s`，在冷/热机混跑下估算系统性失真，因此被否。
- **为所有任务套用保守固定超时常量，不做历史采样与模态区分** — 实现最简单；但会导致轻负载短任务过度占槽被误拒，或长分辨率/高帧数任务估算偏低而中途强杀，因此被否。

## Consequences

- **收益**：将固定启动、条件性权重恢复、纯采样时长与安全余量物理拆层，彻底隔离展示层 CU 估算（`rates.py`）与调度器准入/截止时间硬边界（`estimator.py`）。
- **代价与已知上限**：未完成基线校准的工作流参数（缺失必需 `workflow_key` 或非正有限参数）会明确抛出 `UncalibratedWorkloadError`，拒绝静默降级兜底；当前固定开销常量基于 Colab L4/T4/A100 观测，若更换底层存储恢复速度需显式校准上述常量与模态恢复表。

## Verification

在 `/Users/huanglin/code/github/colab-gpu-studio` 运行：

- `pytest -q tests/gpu/test_reuse_estimator.py tests/gpu/test_issue5_batch_handoff.py tests/gpu/test_pr4_signoff_regressions.py`
- `pytest -q tests/`（`803 passed`）
