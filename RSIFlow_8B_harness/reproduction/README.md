# 复现输入与启动

本版是在 skill 维护基础上让 Meta 更新完整三文件 Python 包，并在下一轮实际调用。
Task 仍为 Qwen3-4B，Meta 使用 DeepSeek-V4.1-Flash + 常驻 Codex。
默认五轮、每轮三领域各 60 条共 180 条、A0 独立基线开启、ACE 跳过。
当前默认八卡、八个推理服务端口和 32 个 rollout worker，MODEL 为八卡 DDP；保留四卡模式兼容。启动前按机器硬件设置配置，不要把八卡默认配置直接用于四卡机器。

## 已包含什么

- 完整 Task / Meta 源码、初始 Task Harness、固定 HarnessForge 源码及生产模板。
- `meta_harness/G000/{workflow.py,planning.py,memory.py}`、版本加载器和真实工具接线。
- 配置、启动／监控／聊天脚本、`eval/` 三个独立入口、测试和 Debug 记录。
- `frozen_manifests/B1` 到 `B5` 的原始任务清单，共 900 个不同任务 ID；每轮三领域各 60。
- `frozen_manifests/validation/manifest.json`：原报告用清单，共 300 条；运行跳过 ACE 的 50 条。
- 原数据准备审计、原协议、检索库版本清单及 `environment-observed.json` 的关键依赖观察值。

这些清单保留原路径、角色和任务身份，没有改写清单以伪装为自包含数据集。
清单中的 record_file、记录偏移和检索引用仍需要真实文件；复制清单本身不能恢复题目。
新启动使用代码中的最新 configs/train_180_a0_v1.json，而不是直接把历史数据 protocol.json 当启动配置。

## 还需要准备什么

| 外部依赖 | 配置或仓库中的依据 |
|---|---|
| Qwen3-4B 权重 | runtime/configs/base.json 的 task_checkpoint；原始 Task 种子不变 |
| GPU 驱动、CUDA、推理和 SFT Python 环境 | runtime/pyproject.toml、upstream/HarnessForge_4B/requirements.txt、environment-observed.json |
| 同一批冻结原始任务记录、Task 索引和公共检索库 | frozen_manifests、configs/train_180_a0_v1.json 的 frozen_data_dir / retrieval_index |
| EnvScaler 环境定义与对应 utils | runtime/configs/base.json 的 envscaler_root / envscaler_utils / envscaler_commit |
| 官方评测器与 benchmark 数据、对应 Python 环境 | configs/validation.json、仓库 runs/output_submission_20260916/runtime/configs/report-evaluators.json |
| 支持当前 app-server / dynamic tools 的 Codex 可执行文件 | 仓库 third_party/codex-provenance.json；RSIFLOW_CODEX_EXECUTABLE 可覆盖路径 |
| DeepSeek API 认证 | 自行设置 AUTODL_API_KEY；API 服务与模型配置已包含，不上传真实凭据 |

仓库已有 HotpotQA / 2Wiki 评测源码，BFCL、EvalPlus、LiveCodeBench、EnvScaler 等外部目录和
所需数据仍要按记录版本准备。第三方源码保留各自许可证，环境和模型不随 Git 安装。

当前服务器可只读复用已配置的模型、冻结数据、索引和评测器。
在另一台机器，需要设置下面的路径；仅 git clone 不能满足这些外部依赖。

## 在新机器上的顺序

1. clone 本仓库，进入 RSIFlow_8B_harness，准备与记录版本兼容的 Python/CUDA/Codex 环境。
2. 准备模型、官方 benchmark 及同一批任务的原始记录和索引；不得静默换成新随机任务。
   任务清单包含原 record_file/偏移，路径迁移后需要重建相应索引并保留任务 ID。
3. 调整 configs/train_180_a0_v1.json 中的 dataset_root、frozen_data_dir、retrieval_index、
   validation_vault、output_root 和全新的 run_name；服务端口数量应与目标 GPU 数量一致。
4. 调整 runtime/configs/base.json 中的模型、EnvScaler 和 trainer_python 路径，
   以及 configs/validation.json 和官方 specs 中的解释器、数据、评测器路径。
   只设置 RSIFLOW_PYTHON 不会替代 JSON 里的其他解释器路径。
5. 清理自己配置中不适用的 wait_for_processes 项；它是本机等待条件，不是实验组件选择。
6. 在终端设置 AUTODL_API_KEY、RSIFLOW_PYTHON 和 RSIFLOW_CODEX_EXECUTABLE，先离线检查再启动。

```bash
git clone https://github.com/1HUAN1/RSI-Flow.git
cd RSI-Flow/RSIFlow_8B_harness

# 先完成上述依赖及 JSON 路径设置；认证只设置在当前环境，不写入 Git。
export RSIFLOW_PYTHON="$(command -v python)"
export RSIFLOW_CODEX_EXECUTABLE="$(command -v codex)"

"$RSIFLOW_PYTHON" launch_meta.py --check
"$RSIFLOW_PYTHON" -m pytest -q tests

# 使用新目录；同一目录再次启动是续接，不是重新初始化。
bash start_meta.sh --wait-for-gpus --run-dir /你的数据盘/Rollout_logs/runs/新实验名
bash watch_meta.sh --run-dir /你的数据盘/Rollout_logs/runs/新实验名
bash chat_meta.sh --run-dir /你的数据盘/Rollout_logs/runs/新实验名
```

`launch_meta.py --check` 不调用 API/GPU，也不保证全部外部 benchmark 环境已安装。
源码发布的 168 项离线回归通过；仍需真实运行来验证新机器的 API、GPU、数据及评分接线。

## 对照入口

- `eval/task_version.py`：指定 Task 模型／Harness／Artifacts 组合。
- `eval/meta_pair.py`：固定 Task、证据、skill、上下文，比较原／新 Meta 包的首个候选。
- `eval/loop_summary.py`：汇总固定 Meta 与可更新 Meta 的多轮结果。

前两个入口默认只准备；明确使用 --execute 才运行。独立报告不参与接受、SFT 或经验学习。
