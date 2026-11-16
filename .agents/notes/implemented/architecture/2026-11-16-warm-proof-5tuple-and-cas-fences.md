# Agent Note: 暖机复用证明校验与代际 CAS 隔离

Status: implemented

## Problem

远端 Colab GPU 运行时的创建、复用到释放跨越进程重启、并发停止与异步对账。若仅凭 lease 的 `idle` 状态或运行中写入的“已预热”标记放行复用，崩溃重启、裸 handle 租约、`cleanup_pending`/`deleted` 状态误转回就绪池、或迟到 worker 的陈旧写操作会静默污染执行归属并越过容量与预算上限。

## Decision

复用资格校验与生命周期流转通过严格签名证明和单调代际 CAS 隔离执行：

- **完整 warm proof 校验（`must`）**：`api/app/gpu/repository.py` 中的 `_validate_warm_proof`（L2452–2511）仅在 `terminal_state` 为 `uploading` 或 `completed`、非空 `execution_id`、`warm_saved == True`、`worker_reported_execution_id == execution_id`、非空 `warm_epoch_id`、`runtime_meta["python_mm"]` 且 `warm_fingerprint` 精确匹配 `compute_manifest_warm_fingerprint(manifest, python_mm=...)`（或受限 mock 规则）及正整数 `generation >= 1` 时落盘 proof。`api/app/gpu/estimator.py` 中的 `_select_warm_proof`（L705）与 `is_warm_verified_for_job`（L787）在 instance 与 lease 同时存在时要求两者持有相同当前代 proof，并与任务的 `manifest`、`generation`、`epoch`、`execution`、`python_mm` 一一对齐。
- **代际绑定的 CAS 与执行隔离**：`api/app/gpu/repository.py` 中的 `_execution_matches_bindings_and_generation`（L858）在每条执行写入路径强制校验 `job`、`lease`、`instance` 与 `execution` 的非空归属和当前 `generation`，防止迟到上报覆盖新一轮租约。
- **创建/停止/恢复单调栅栏**：`api/app/gpu/repository.py` 中的 `_resolve_lease_provider_in_session`（L5096）解析无显式 `provider` 字段的裸 Colab handle 租约，避免误跳过隔离；创建前原子占额并写入 `create_intent_id` 与递增 `alloc_generation`，`cleanup_pending` 或 `deleted` 实例与租约绝不回退为 `retained_idle` 或 `idle`，停止删除在远端确认前保留容量占额。

 相关被否路线见 [放弃乐观暖机标记与混乘 length_factor](../../rejected/architecture/2026-11-16-reuse-rates-length-factor-and-optimistic-warm.md)。

## Alternatives considered

- **仅在 lease/instance 保存布尔 `warm` 标记或忽略 `python_mm` 与执行 ID 核对** — 该做法实现简单且不依赖执行末段签名元数据；但无法识别运行期已更换依赖环境、跨代次复用残留或中途失败的伪暖机状态，在多轮并发恢复下会出现冷环境误按热机免扣 `env_weights_restore_seconds` 的超时事故，因此被否。
- **允许 `cleanup_pending` 在本地对账前乐观回到 `idle` 池，或按 `provider` 字符串空值默认走非 Colab 分支** — 这样可短暂提高本地可见空闲率；但会在 `start × stop` 竞态或历史裸 handle 租约上丢失隔离校验并导致双花容量占额，因此必须在 session 内经 `_resolve_lease_provider_in_session` 补齐归属后再判定。

## Consequences

- **收益**：阻断未完成执行、跨代延迟写、裸 handle 租约与待清理实例进入复用池，保障重启恢复（R0–R3）与 `start × stop` 并发边界的单一事实源。
- **代价与已知上限**：每次生命周期迁移与执行写回必须携带匹配的 `generation` 与完整 `warm_proof` 字段，mock 测试也必须满足显式 `provider == "mock"` 与签名校验，无法省略字段快速造假数据；当后续引入非 Colab 的多运行时隔离机制时，需同步扩展 `_resolve_lease_provider_in_session` 与 `_validate_warm_proof`。

## Verification

在 `/Users/huanglin/code/github/colab-gpu-studio` 运行：

- `pytest -q tests/`（已在 commit `8d6e7df` 前全量通过：`803 passed`，覆盖 `tests/gpu/test_warm_proof_isolation.py`、`tests/gpu/test_recovery_r0_r3_and_fences.py`、`tests/gpu/test_pr4_signoff_regressions.py`）
- `/Users/huanglin/.cache/uv/archive-v0/5WpI6oa8el_QWH87/bin/ruff check api tests`（零告警/零错误）
