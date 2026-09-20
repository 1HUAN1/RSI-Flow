# RSI 三域训练流程：开发交付与运行说明

项目位置：`/root/data/RSI_iclr2027/rsiH`。新增流程继续使用修复后的 Task–Meta 主循环。
正式 Meta 为项目管理的开源 Codex + OpenRouter `deepseek/deepseek-v4-flash-0731`；当前开发聊天窗口不充当实验 Meta。
对比 baseline 明确为 [SIA](https://github.com/hexo-ai/sia) 和 [HarnessForge](https://github.com/mingju-c/HarnessForge)。

## 已实现的工程主线

- `sia/task_meta/pipeline.py`、`pipeline_execution.py`：三域共享一个 Qwen3-4B Task 状态；按固定域配额、不放回遍历训练池；每代另用同一个只读开发探针。预算到期可以结束已接受干预的配对评测，不能产生未评测 successor。
- `loop.py`、`durable.py`：保留 `T_in → 执行 → T_out → 单组件 M/H/A 干预 → 下一代执行 → 经验 → Meta 更新`。N 次执行对应 N−1 次干预/经验，末轮单独 consolidation。没有固定 H/M/A 顺序，没有性能回退和 best-of-K 干预选择。自然资产生成与 A 干预分别计账。
- `seed_harness/seed.json`、`sia/task_meta/seed.py`：明确可执行的初始 Harness；H updater 只改受控 JSON 叶子，在下一代重新装载。权重、数据、评分、安全边界与外部预算不能由 H 修改。
- `data.py`、`retrieval.py`、`environments.py`、`sandbox.py`：流式 SQLite manifest、冻结 FTS5 检索、真实 EnvScaler 工具与官方检查、完整 TACO 训练验证器。候选 Python 通过 Landlock/seccomp、独立 UID、无网络和资源上限运行；隐藏期望值留在可信评分进程。
- `meta_backends/`、`meta_harness/`：真正启动 Codex exec 的 MetaBackend，独立调用工作区、请求身份/schema、事件/transport 证据、外置密钥转发、G Bundle 版本提交与后续装载。G 当前可改 instructions/context/workflow；不宣称支持修改 Codex Rust 内核、构建源码或权限边界。
- `sft.py`、`train_task_meta_sft.py`、`serve_gpu.py`：从当前 checkpoint 做正样本 LoRA SFT、合并为新权重、服务重新绑定和逐请求验证。Code 必须完整通过训练 verifier，Tool 必须任务成功，Search 必须 EM 正确。每条真实模型调用拆为一条训练对话，仅监督该对话末尾 assistant，从而覆盖规划/动作/记忆决策；重写的历史和工具观察不计算 loss。超长样本拒绝并记账，不静默截断目标。旧 GPQA 保留独立兼容路径。
- `file_lock.py`、阶段 receipt：防止重复并发 resume；已提交阶段按内容校验重放；外部副作用状态不明时保留 pending，不重复训练。G 版本重命名后、active 指针发布前崩溃可恢复。
- `report_predictions.py`、`reporting.py`：冻结 best-on-dev 后才生成最终预测，训练/Meta 不可读最终评分；固定分母、一题一个最终提交、CSV 和 LaTeX 输出。Code 的 system-level pass@1 包含声明的内部调用预算；没有未登记的 Overall。

## 初始化和数据协议

Task 固定为 `/root/data/zh/huggingface/Qwen3-4B`。权重、tokenizer、chat template 与 seed 均有指纹。
共同 seed 参考 HarnessForge commit `05b3ecadb3c9a7a938f75129ea22b8f2b36cf289`，初始资产为空。
参考实现的规划/动作/记忆请求及停止行为已用相同模拟响应回放；这是契约验证，尚非真实 Qwen rollout。
适配差异见 `seed_harness/README.md`。后续 SIA、HarnessForge 的受控对比必须加载同一初始化导出，并单独注明原算法及适配差异；不把尚未执行的 baseline 宣称为已对齐验证。

| 域 | evolve_train | 内部 search_dev | 固定探针 |
|---|---:|---:|---:|
| Tool / EnvScaler | 2,100 | 450 | 16 |
| Code / TACO | 6,600 | 734 | 16 |
| SearchQA / NQ、Hotpot、2Wiki | 302,948 | 34,112 | 16 |

训练池共 311,648 题，内部开发集 35,296 题。9 条 Search 重复问题被显式登记；保留既有 Env 环境级划分及 TACO train/val 划分。原五项清单 346,219 包含 Env heldout、未包含额外 TACO val；不能直接当 evolve_train 数量。manifest 中有来源/hash、遗漏理由、固定 probe 身份。

Search 当前使用训练池公开 context 段落构成的统一开放检索库，没有把问题/答案/supporting facts 当检索文档。它不是完整 Wikipedia；NQ 和最终 QA 的证据覆盖尚未测量。换检索库必须登记新协议，不能混合为同一实验。

覆盖量与模型更新量分开：遍历全部任务表示 rollout 覆盖，不表示每题都被用于 SFT；Meta 可能选择 H/A。每次 M 都保存筛选、去重、实际采样、监督 token 和 checkpoint 证据。GRPO 未实现为本轮训练结果。

## 当前验证范围和明确限制

实际执行的是回归测试、模拟模型的完整三域闭环/恢复测试，以及真实数据和官方评分器的 CPU fixture。所有 fixture 均标记 `mock` 或 `test_override`，不进入训练池或正式成绩。

最终服务器回归为 **374 passed、2 skipped、19 subtests passed**；另有一条既有 Web 依赖弃用警告。本次修改的 Python 文件通过 Ruff 检查。HumanEval+ 全部 124,253 个增强测试输入和 MBPP+ 全部 41,015 个输入通过安全传输审计，两者首题官方参考程序通过完整 base/plus 判定；这不是对 Task 模型运行全量 benchmark。真实 EnvScaler 工具/反馈、TACO 完整题目判定和冻结检索的 CPU fixture 也已执行。

Codex 固定 `rust-v0.153.4` / commit `3d2ee51ca2d5db578f328aa75e20aa22c0197c9a`，官方同 tag 二进制 SHA-256 为 `56ef98ab4032d317ab26e9b5e5a1756507177351edb16ed9cde0cb6d1734d62da`。这是官方 release 产物，不是本机 Rust 编译。原生配置 schema、公开模型 metadata 与 native catalog 已核对；真实 API 工具兼容性仍为 `IMPLEMENTED_NOT_API_VALIDATED`。

本机 Task 的 Landlock/seccomp 可用；正式 Meta 所需的 bubblewrap/user/PID/network namespaces 不可用，状态为 `BLOCKED_META_SANDBOX`。当前会话未提供实验 OpenRouter 环境密钥，也未启用付费 smoke。必须先在支持这些边界的运行环境部署 Meta runner；不能通过取消隔离来启动。

最终评测已登记五个可执行适配路径：LiveCodeBench release_v6、HumanEval+ v0.1.10、MBPP+ v0.2.0、HotpotQA-dev、2Wiki-dev。本机原标记为 EvalPlus 的 HF 文件缺 `base_input/plus_input`；已另取官方 release 的完整增强测试，保持旧文件原样。2Wiki 源数据不含 answer_id/evidences_id，匹配官方原版 v1 评分器；不伪称运行 alias-aware v1.1。

BFCL-v3 和 ACEBench 的官方多轮执行、输出格式/子集适配仍为 `BLOCKED_REPORT_ADAPTER`。七项报告保留这两列 N/A，不能宣称七项最终评测全部完成。真实 GPU smoke、真实 M 更新、真实 API G 更新及 pilot/full 均尚未执行。当前代码接口、真实 API smoke、完整研究实验是三个不同交付状态。

## 开始、恢复和最终评测

以下命令在服务器项目目录执行。默认开发检查不会启动付费模型或 GPU 训练。

```bash
cd /root/data/RSI_iclr2027/rsiH
export PATH=/root/data/conda/envs/sia/bin:$PATH
python scripts/run_multidomain.py prepare --config configs/multidomain-dev.json
python scripts/export_initialization.py --output local_baseline/shared_initialization_v2
python scripts/run_multidomain.py preflight --config configs/multidomain-dev.json
python -m pytest tests -q
python scripts/prepare_report_specs.py
```

本次已生成 `local_baseline/shared_initialization_v2/`，包含共同的 `seed.json`、空 `artifacts/` 和模型/数据/预算身份清单。初始化 hash 为 `9dea9eaf8f3f7de1d622f20b56552a623e9b74121e21f58b44e445f19e7dfce8`。SIA、HarnessForge 后续适配时应验证此清单，不能仅凭参数名称相同就认定初始状态一致。

支持 Meta 隔离、配置密钥且显式启用后，独立 API 兼容关卡才可执行。密钥只通过运行进程环境提供，不写入配置/仓库/日志。

```bash
python scripts/run_multidomain.py api-smoke --config configs/multidomain-api_smoke.json --run-dir local_baseline/api_compatibility
```

API smoke 必须验证真实 Codex 工具、patch/test、事件流、schema 与 G 重载。只通过裸 completion 不够。
推理服务需另一个服务器终端；选择有空闲资源的 GPU，启动器不会停止其他进程。推理需至少 20 GB 空闲，训练需至少 40 GB；GPU 分配和预算在运行前写入配置并随协议冻结。

```bash
python scripts/start_multidomain_service.py --config configs/multidomain-pilot.json --enable-gpu
python scripts/run_multidomain.py run --config configs/multidomain-pilot.json --run-dir runs/multidomain_pilot_001
python scripts/run_multidomain.py audit --config configs/multidomain-pilot.json --run-dir runs/multidomain_pilot_001
python scripts/run_multidomain.py resume --config configs/multidomain-pilot.json --run-dir runs/multidomain_pilot_001
```

pilot 验证后显式 full，使用新目录。`full` 配置的上限会裁到实际数据窗口数量；7 天墙钟预算包含离线中断时间。停止时查看 coverage，不能因配置名称为 full 就称全量完成。

```bash
python scripts/run_multidomain.py run --config configs/multidomain-full.json --run-dir runs/multidomain_full_001
python scripts/run_multidomain.py resume --config configs/multidomain-full.json --run-dir runs/multidomain_full_001
python scripts/run_multidomain.py audit --config configs/multidomain-full.json --run-dir runs/multidomain_full_001
python scripts/run_multidomain.py freeze --config configs/multidomain-full.json --run-dir runs/multidomain_full_001
python scripts/run_multidomain.py report-predict --config configs/multidomain-full.json --run-dir runs/multidomain_full_001 --evaluator-specs configs/report-evaluators.json --output-dir runs/multidomain_full_001/report_predictions --enable-inference
python scripts/run_multidomain.py report-eval --config configs/multidomain-full.json --run-dir runs/multidomain_full_001 --evaluator-specs configs/report-evaluators.json --predictions-dir runs/multidomain_full_001/report_predictions --output-dir runs/multidomain_full_001/report_eval
```

完成运行自动保存 best-on-dev 与 last-evaluated 独立身份。恢复要求原协议、代码、数据、模型和初始 seed 不变；保护逻辑的工程修补要开新实验。`intervention_receipt.json` 为 started 而非 committed 时先根据训练日志/checkpoint 对账，不能删 receipt 重试。

## 证据和历史保护

分支：`codex/multidomain-pipeline-v2-20260908`。原工作树备份、既有差异和本次逐文件部署清单位于 `local_baseline/pipeline_v2_20260908/`。未覆盖旧运行和 `run_4001`，未提交其他用户修改。
该目录保存测试日志、CPU 适配器验收、native evaluator 输入审计、Codex 配置验证和 tokenizer mask 结果；最终验证汇总另存 `delivery_status.json`。
具体入口为 `pytest_final.log`、`ruff_final.log`、`adapter_integration2/integration_results.json`、`native_report_verification2/verification.json`、`native_config_probe/verification.json`、`qwen3_tokenizer_mask.json`。`delivery_report/results.csv` 和 `table_row.tex` 是七项均 N/A 的未执行清单，不是正式成绩。正式逐题预测与 evaluator 原始输出只有真实 report_eval 完成后才会产生。
数据/模型准备身份位于 `data/pipeline_v2/`；正式 report 数据独立位于 `data/report_eval/`。
