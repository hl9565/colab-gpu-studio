# Agent Note: 拒绝混乘 rates.length_factor 与跳过完整 warm proof 校验

Status: rejected — 将固定启动权重混入 length_factor 或放宽 warm_proof 字段校验会直接破坏截止时间守卫与恢复隔离

## Problem

在设计 PR4 运行时复用准入与崩溃恢复期间，曾评估两条“复用既有代码/简化状态字段”的方案，以期减少新增 `api/app/gpu/estimator.py` 与完整 `warm_proof` 校验逻辑。

## Proposal

1. 在调度准入中直接调用 `api/app/gpu/rates.py` 中的 `expected_minutes`（L30–L51）及其 `length_factor`（L67–L77）计算剩余可执行窗口，而非新建可加性冷基线采样模型。
2. 允许基于布尔 `warm` 标记或仅匹配 `manifest` 放行暖机复用，免去 `worker_reported_execution_id`、`warm_epoch_id`、`runtime_meta.python_mm` 与代际 `generation` 一致性核对。

## Alternatives considered

- **独立可加性冷基线 + 严格 `warm_proof` 校验** — 最终采纳方案，见 [纯采样冷基线估时器](../../implemented/architecture/2026-11-16-cold-baseline-sampling-estimator.md) 与 [暖机复用证明校验与代际 CAS 隔离](../../implemented/architecture/2026-11-16-warm-proof-5tuple-and-cas-fences.md)。

## Risks

- `rates.length_factor` 将固定 `SAMPLING_SHARE = 0.35`（其余 65% 为固定权重）按比例揉入单次总时长，无法按本次实例是否持有有效 `warm_proof` 精确切换 `0s` / `180s` / `330s` 恢复惩罚，且会继续沿用展示层统计口径污染硬截止期决策。
- 放宽 `warm_proof` 字段校验后，Python 小版本变化、失败/取消中断的半成品环境、或上一代过期租约延迟回写都会伪装成 verified warm 逃过恢复惩罚与代际栅栏。
- **重提条件**：仅当 `rates.py` 彻底剥离费用展示口徑且按可加阶段重写、且运行时引入密码学签名的不可变镜像快照代替 `warm_proof` 字段核对时，才可重新评估合并两套机制。
