# RSI 联合进化与逐轮验证

面向现有四 GPU Linux 训练服务器的一键入口。默认 **3 轮联合进化**，也支持 `--rounds 5`。每轮提交 Task/Meta 状态后，先运行全部 7 个外部验证集，再进入下一轮。每次实际 SFT 更新训练 **1 epoch**。

这是新实验脚本；不恢复、修改或拼接此前中断实验。当前交付不包含真实训练成绩。CPU 模拟检查也不计为实验成绩。

## 启动

服务器工作目录是 `/root/data/RSI_iclr2027/rsiH`。代码保存在该目录的 `RSI_joint_training_20260918/` 子目录，根目录入口为 `start_joint_346219.sh`。首次启动前，在该终端配置既有 Meta 服务的认证环境变量与健康的隔离 Codex worker；变量名沿用服务器 `runs/output_submission_20260916/runtime/configs/envscaler-256-direct-five-global-until-20260917-235900.json` 中的 `meta.api_key_env` 和 `meta.remote_worker_token_env`。这些凭证不写入脚本、配置或日志。

ACE 的多轮交互还需要官方用户模拟器：配置 `ACE_USER_API_KEY`、`ACE_USER_BASE_URL`。本次使用已核实支持该版本的 OpenRouter；官方模型名为 `gpt-4o-2024-08-06`，供应商请求名为 `openai/gpt-4o-2024-08-06`，由 `user_provider_model` 显式绑定。这是评测器的用户角色，不是 Task 或 Meta 模型。每个验证轮次最多 20,000 次用户模拟请求、每次最多 2,048 输出 token，不自动重试或充值。

直接从实验工作目录执行：

```bash
# 默认 3 轮；后台运行，关闭 SSH 后继续
cd /root/data/RSI_iclr2027/rsiH
bash start_joint_346219.sh --detach

# 如果本次要 5 轮，用这一条代替上面一条
bash start_joint_346219.sh --detach --rounds 5
```

先检查、不训练：`bash start_joint_346219.sh --check`。前台运行：`bash start_joint_346219.sh`。不要同时启动 3 轮和 5 轮；目录锁会阻止并发控制器。训练使用原 conda 环境；BFCL/ACE 使用代码目录内的 `.venv`，其官方依赖不进入训练进程，也不修改原 conda 环境。默认训练 Python 为 `/root/data/conda/envs/sia/bin/python`，可通过 `RSI_PYTHON` 指定。若移动整套目录，需把附带的 `start_joint_346219.sh` 复制到代码目录的父目录。

服务器需保留既有 RSI 源码、Qwen3-4B 模型、数据文件和官方评测器。位置均在 `configs/train.json`、`configs/validation.json`，可在首次启动前修改。脚本在服务器本地复制必要运行源码到本目录 `runtime/`，应用经检查的扩展；不复制旧运行结果和认证文件。源码版本不兼容、依赖缺失、数据数量不符或训练/验证重叠时，在训练前明确报错。

## 训练数据与轮次

|数据集|条数|
|---|---:|
|EnvScaler-RL|2,550|
|DeepCoder TACO|6,600|
|NQ|79,168|
|HotpotQA train|90,447|
|2Wiki train|167,454|
|合计|346,219|

每轮使用全部训练场景，每题一个 rollout，不把小窗口循环冒充全量训练。EnvScaler 使用现有 `train` 的 2,100 条加 `heldout` 的 450 条；这 450 条在新协议中明确属于训练数据，不能再称为外部验证集。

每轮流程：全训练集 Task rollout → Meta 自主选择 Harness/Model 候选 → 同父状态评估与部署判定 → Meta 原生自更新提交 → 保存该轮 Task/Meta 快照 → 7 个外部验证集 → 下一轮。沿用单组件、严格正收益才部署的规则；未提升则保留父 Task，仍做真实 Meta 更新和本轮外部验证。Meta 可以选择 Harness，所以 3 轮不保证产生 3 次新模型权重。

部署判定使用从训练集中固定抽取的 48 题（每域 16 题），明确标记为 **训练内监测集**，不会扣减 346,219 条训练数据。外部验证分数仅用于报告，不提供给 Meta、不进入 SFT、不参与候选选择。详细 Meta 轨迹证据按领域和成功状态固定抽取最多 96 条并限制提示长度；完整训练统计和原始回执仍保留。

SFT 使用本轮成功 Task 轨迹中的合法完整监督样本，`num_train_epochs=1`、`max_steps=-1`；不会用固定 32 steps 冒充一轮 epoch。超过 16,384 token 的整条调用排除并记录哈希，不截断；格式/模板错误会中止。1 epoch 指对最终合法 SFT 样本集完整训练一遍，**不等于把失败轨迹或所有原始题目的标准答案直接监督训练**。真实采样覆盖与完成 epoch 写入训练回执。

## 验证与输出

|领域|每轮验证集|报告内容|
|---|---|---|
|Tool Use|BFCL-v3、ACE Bench（英文全类别）|官方各类别分数、完整分母，以及明确命名的宏平均/按样本加权汇总|
|Code|LiveCodeBench、HumanEval+、MBPP+|固定版本官方评分器结果、单次提交通过率|
|SearchQA|HotpotQA-dev、2Wiki-dev|官方 EM/F1|

默认 Code/Search 版本与题目 ID 清单沿用服务器 `report-evaluators.json` 的固定记录。所有外部数据、配置及 Tool 评测源码在启动时冻结哈希，各轮核验；不会悄悄更换测试集或漏掉失败题目。

BFCL/ACE 使用官方外层工具/多轮执行器，每次助手回复通过保存的完整 Task harness 生成；接入协议记录为 `official_outer_saved_harness_inner_v1`。这是明确的 harness 包装评测条件，不宣称与排行榜上直接调用裸模型的条件相同。原始类别分数、预测和官方输出文件全部保留。

Search 使用训练集公开 context 段落构建的固定 SQLite 检索库，问题、答案和 supporting facts 不索引。当前未接入完整 Wikipedia，NQ 检索覆盖未测量；QA 成绩须标注这一检索条件，不宣称是完整 Wikipedia 检索成绩。

默认 3 轮输出：

```text
runtime/runs/joint_346219_r3/round_0/complete.json  # 原生联合轮次回执
runtime/runs/joint_346219_r3/round_1/complete.json
runtime/runs/joint_346219_r3/round_2/complete.json
validation/joint_346219_r3/round_01/round_snapshot.json  # 模型、Harness、Meta 绑定
validation/joint_346219_r3/round_01/frozen_task.json
validation/joint_346219_r3/round_01/scores/              # 各基准真实结果
validation/joint_346219_r3/round_01/complete.json        # 7 个基准全部完成才生成
validation/joint_346219_r3/all_rounds.csv                # 所有轮次各指标
validation/joint_346219_r3/RESULTS.md                    # 易读成绩表
logs/launcher.log
logs/joint_346219_r3.log
```

5 轮目录后缀为 `_r5`。模型权重与 harness 文件位置由每轮快照绑定；不反复复制巨大的相同模型，原文件保留在本实验运行目录或初始模型目录。

## 查看进度与恢复

本次启动另有服务器常驻 `monitor.py`，每分钟更新 `monitor.json` 并追加 `logs/monitor.jsonl`，记录进程身份、GPU、样本回执数和已完成轮次。它不自动重发未决模型请求；故障记录为 `requires_attention`，核对回执并修复后接续。

```bash
RSI_joint_training_20260918/.venv/bin/python RSI_joint_training_20260918/status.py
tail -f RSI_joint_training_20260918/logs/launcher.log RSI_joint_training_20260918/logs/joint_346219_r3.log
```

完成回执分别统计“联合轮次已提交”和“7 基准验证已完成”。运行失败时保留所有回执和费用账目；修复提示的具体问题后重跑同一条启动命令。已完成的轮次、训练和验证结果按哈希复用。未决模型请求或不明确的训练副作用需要核对回执，脚本不会盲目重发；不能保证上游服务永不中断。

新实验总时限默认 30 天，单次 SFT 7 天，可在首次运行前调整；Meta 单操作限制继承基础配置。达到上限会保存现状并停止，不自动扩额。大规模运行的实际耗时和硬盘需求取决于每题调用次数，当前没有完成全量实测，不能承诺某个小时内跑完。

## 文件职责

- `start_joint_346219.sh`：放在服务器工作目录的启动入口。
- `start.sh` / `setup_environment.py` / `launch.py`：独立依赖环境、前置检查、唯一控制器、一键启动；首次安装后记录实际依赖版本，后续启动核验。
- `configs/train.json` / `prepare_data.py`：训练数据、固定数量、索引与泄漏检查。
- `train.py` / `install_runtime.py` / `runtime_extensions.py`：原生 RSI、多 GPU 分批调度、落盘轨迹、1 epoch 扩展。
- `configs/validation.json` / `round_validation.py` / `validate.py`：独立逐轮验证及成绩表。
- `validation_sources.py`：跨轮验证版本冻结。
- `tool_validation.py` / `official_tool_entry.py` / `harness_api.py` / `tool_sandbox.py`：官方 Tool 评测适配及隔离。
- `tests/test_contracts.py`：CPU 合约与恢复测试，使用明确的模拟模型。
- `tests/server_cpu_checks.py`：服务器原生评测器导入、RPC 和隔离检查，无模型调用。

评测接口参考：[BFCL 官方仓库](https://github.com/ShishirPatil/gorilla/tree/main/berkeley-function-call-leaderboard)、[ACEBench 官方仓库](https://github.com/Agent-Suite/AgentSuite/tree/main/ACEBench)。实际运行以服务器冻结版本为准。
