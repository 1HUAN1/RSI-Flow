# 当前五轮源码快照与外部依赖

本目录包含当前实验的主程序、Task runtime、初始 Harness manifest、固定的 HarnessForge 源码、
Meta 技能维护实现及测试。目录名 8B 不代表模型更换：Task 仍为 Qwen3-4B。
默认配置现在是五轮，每轮三个领域各 60 条，共 180 条；A0 独立基线评测开启。
旧运行目录名称中仍有 3round，实际轮数以配置和 workflow_policy.json 为准，不重命名活跃日志。

## 源码不等于完整服务器环境

Git 不包含 API/GitHub 密钥、Codex 二进制、conda/venv、模型权重、冻结任务、检索 SQLite、
官方 benchmark 数据、评测器安装目录或 Rollout_logs/Meta_logs。仅 clone 不能直接复现服务器环境。
启动代码不依赖 RSIFlow_4B 的 Python 源码；当前五轮任务的前三批仍引用其冻结数据目录，
密钥读取也保留旧 API_key.md 的可选回退。新机器可以用 AUTODL_API_KEY，避免依赖旧密钥文件。

需要准备：

| 依赖 | 当前配置入口 |
|---|---|
| Task 模型及四卡环境 | runtime/configs/base.json 的 task_checkpoint、envscaler_root、envscaler_utils |
| 五轮冻结任务、原始源数据和检索语料 | configs/train_180_a0_v1.json 的 frozen_data_dir、dataset_root、retrieval_index |
| 独立评测任务清单 | 同配置的 validation_vault |
| BFCL/EvalPlus/LiveCodeBench/HotpotQA/2Wiki 官方评测器、对应 Python 环境及数据 | configs/validation.json、其 official_specs 指向的外部配置 |
| Codex 可执行文件 | persistent_meta.py 默认的服务器 third_party/codex-bin；按本机安装路径准备 |
| Meta 模型/API 接线 | configs/codex_host.toml、configs/codex-deepseek-v4.1-flash.json、AUTODL_API_KEY |

配置中的 /root/data/... 是服务器现有路径，迁移机器时需要对应修改，不能视为可下载的依赖。
当前官方评测脚本及版本列表由 configs/validation.json 的 official_specs 指向外部 JSON；
该 JSON 和它指向的评测器必须另行准备。ACE 默认跳过。
runtime/pyproject.toml、upstream/HarnessForge_4B/requirements.txt 是依赖参考，
实际 GPU、Codex、官方评测器和模型服务还需要匹配各自环境。

## 启动、续接与监控

先进入准备好的 Python/conda 环境，并在本机设置 AUTODL_API_KEY，不要写入 Git。
在当前服务器，start_meta.sh 默认使用 /root/data/conda/envs/sia/bin/python；
其他机器可设置 RSIFLOW_PYTHON 为自己的解释器。

```bash
cd RSIFlow_8B
python launch_meta.py --check

# 新实验必须用新的运行目录；同目录再次启动为续接。
bash start_meta.sh --wait-for-gpus --run-dir /你的数据盘/Rollout_logs/runs/新实验名
bash watch_meta.sh --run-dir /你的数据盘/Rollout_logs/runs/新实验名
bash chat_meta.sh --run-dir /你的数据盘/Rollout_logs/runs/新实验名
```

extend_rounds.py 可以从已有三轮冻结清单补充后两轮，前提是对应真实源数据与评测配置已准备。
它不重新采样前三轮、只生成 B4/B5，并建立共享检索索引及新任务索引。
EnvScaler 环境定义可以复用，但五轮 task_id 与规范化题目内容不重复；
Code/SearchQA 保留跨轮家族隔离。这不是五轮环境全部不同的实验。

## 验证与已知边界

离线回归不等于五轮真实实验完成，更不保证每次候选涨分。
当前候选拒绝后会在同批任务上重新选择，尝试次数不固定，总时长不固定。
B3 曾遇到 worker 中断和未完成调用恢复问题；当前源码包含 Meta 当轮落实的恢复和单题超时改动。
阶段账本若缺少完整回执，可能落后于实际任务执行，应结合 selection.json、performance.json 和作业结果检查。
经验维护目前以新增、补证和关联修订等记忆操作为主，不等于 Meta 模型参数或 Codex 底座自我训练。

```bash
PYTHONPATH="$PWD/runtime:$PWD" python -m pytest -q tests
```

详细修改记录见 DEBUG_LOG.md，技能契约及工具用法见 META_SKILLS.md。

