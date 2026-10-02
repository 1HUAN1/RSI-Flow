# RSIFlow_8B：Codex 主导的实验编排

`8B` 是新版代码目录名，Task 模型仍是本机 Qwen3-4B。本目录保留 Task 执行引擎，但不调用旧版 `pipeline.run()` / `run_sequential_task_meta()` 固定决策循环。

**当前新实验默认：五轮 B1—B5，每轮 600 条，共 3000 条不同任务；每轮保存快照，仅第五轮结束后评测完整 Task，验证集固定 744 条，不包含 LiveCodeBench。** Meta 仍为 skill-only，不启用 Meta Harness 自修改。历史 180 条配置和实验记录保持不变，下文旧协议说明不覆盖新配置。外部依赖见 [SOURCE_RELEASE.md](SOURCE_RELEASE.md)。

## 600×5 / 744 新实验

配置为 `configs/train_600_5round_744.json`，评测配置为 `configs/validation_744.json`。
训练沿用 shared3000 的全部既有任务，不重新抽样：每轮 EnvScaler 200、Code 200（stdio 120 / function 80）、HotpotQA 100、2Wiki 100。
任务 ID 与问题内容跨轮不重复；EnvScaler 可以复用环境定义，不宣称环境跨轮隔离。
验证为 BFCL-v3 152、ACEBench 50、HumanEval+ 164、MBPP+ 178、HotpotQA-dev 100、2Wiki-dev 100。
ACE 沿用这份已有评测的 DeepSeek-V4.1-Flash 用户模拟器口径，启动评测前需提供 `ACE_USER_API_KEY` / `ACE_USER_BASE_URL`；不跳过 ACE，不复用旧模型分数。

```bash
cd /root/data/RSI_iclr2027/rsiH/RSIFlow_8B
# 仅准备数据、检查路径，不调用 API/GPU
/root/data/conda/envs/sia/bin/python prepare_600_744.py
/root/data/conda/envs/sia/bin/python launch_meta.py --check
# 明确要启动时执行；四卡忙时等待，不终止其他作业
bash start_5round_600.sh --wait-for-gpus
```

新数据在 `runtime/data/rounds_3000_5x600/B1..B5` 与 `runtime/data/validation_744`；准备脚本可重复运行，不覆盖旧数据。
A0 只保存初始化快照，不评测；B1—B4 仍完成修改、配对复测、skill 维护和快照，但不做独立评测。
B5 保存选定 Task 后，以快照中的 checkpoint + Harness + Artifacts 调用独立评测；完成全部 744 条评分后才能结束。
配对接受规则、拒绝后重选、MODEL 的成功轨迹单 epoch LoRA SFT、48 条路由片段规则不变。
此轮变更不迁移其他项目的推理引擎，继续使用本项目现有 Task runtime。

## 职责边界

- `persistent_meta.py` 保持一个常驻 Codex App Server 进程和同一会话；Codex 通过原生工具调用负责组件选择、候选修改、复测判读、部署、经验和继续执行。`meta_loop.py` 中旧的逐命令 exec/resume 循环不再用于启动实验。
- `controller_tools.py` 每次请求启动新进程，只执行 Codex 指定的操作并返回事实。`compare_scores` 不接受候选；`activate_task` 只落实 Codex 的明确选择。
- `task_adapter.py` 提供单阶段 Task rollout、HarnessForge 候选验证、成功轨迹 LoRA SFT、Artifacts 提交修改和独立评估。
- `launch_meta.py` 准备 DeepSeek-V4.1-Flash 与实验提示；会话、回执和轨迹写入 `Rollout_logs`。

研究口径：A0 独立评估；三轮，每轮三个域各 60 条、共 180 条；四卡各运行一个模型副本、最多 16 个 rollout worker；参考全部成绩及按 16/域、8 成功/8 失败筛出的 48 条代表片段。每次尝试由 Meta 选择 HARNESS/MODEL/ARTIFACTS 之一、生成一个候选并原任务配对复测。严格正增益才部署；若拒绝，追加失败经验、保留父代并在本轮同一批任务上重新选择组件，不直接独立评测或跳到下一轮，不重复未变父代的 rollout。暂不设重选次数上限，持续无增益时本轮可能持续较久。接受后保存 Task/Meta 快照、做报告用独立评估。`skip_acebench=true` 时 ACE 是未评分，不算完成成绩。

每次尝试单独保存在 `round_N/attempts/attempt_K/`：决策、定位报告、方向报告、完整候选、检查结果、复测及选择记录。所有尝试追加经验、不覆盖旧证据。账本按 Meta 写入的选择记录提示重选，不自己依据分数选择组件或接受候选。已有实验可用 `meta_session/workflow_policy.json` 的 `retry_rejected_from_round` 标记新规则的生效轮次，默认新实验从 B1 生效。

HARNESS 必须由 Meta 按上游三个模板依次执行：`01_module_localization.yaml` 读取父代源码、成绩/成本、成功与失败轨迹并形成定位报告 → `02_improvement_directions.yaml` 结合定位与历史 Harness/经验，确定保留、借鉴、避免和修改内容 → `03_harness_generation.yaml` 基于前两份报告生成新目录中的完整包 → 最多 3 轮可执行性检查（首次检查加最多两次修复后的复查；一旦通过就不重复检查）。`materialize_harness` 返回模板与两份报告的落盘路径；`harnessforge` 接收这两份报告并负责最后的检查及 Task 状态落位，不是前三阶段的替代品。完整包可以保留未改文件，并非必须改动全部模块。由同一个 Meta 阅读模板、分析和编程，不额外启动另一个生产模型。

报告缺失返回 `production_incomplete`；候选结构/导入/构建失败返回诊断，Meta 在同一会话修复。检查次数用候选旁的 `*_production/validation_attempts.json` 记录，不因更换 output_dir 重置。超限保留父代、总结失败，再由 Meta 按已有重选规则继续；可执行性通过不等于涨分接受。没有新增组件选择或候选内容白名单。

## 轻量版本快照与交叉组合

`snapshot_task_meta` 保存的是组件版本及组合引用，不再逐轮复制整个 Meta 源码、配置、聊天和 Task 目录。每次尝试传入 `parent_task_state`、已生成的 `candidate_task_state`、最终采用的 `active_task_state`、`skills_path`，统一指定 `versions_root=<run>/versions`，使用不同快照目录。激活前另存父代 state，避免 active 文件被覆盖后丢失父代组合。

```text
<run>/versions/index.json          编号及内容哈希索引
             model_001/reference.json   初始模型 checkpoint 引用
             model_002/reference.json   SFT 模型 checkpoint 引用
             harness_001/contents.json  完整初始 Harness manifest（含源码）
             harness_002/contents.json  完整修改版 Harness manifest
             meta_001/contents.jsonl    当版累计经验库
             meta_002/contents.jsonl    下一版累计经验库
<snapshot>/versions.json           parent / candidate / selected 的版本组合
           task/task_state_snapshot.json  仍可交给独立评估的 Task 状态
```

未变化的组件复用同一编号；Artifacts 存在时也只按内容变化保存一次。Meta 的交接上下文继续原处维护，快照保存路径与哈希，不复制整轮聊天。模型权重不额外复制：初始权重和每次 SFT 的独立 checkpoint 目录必须保留，不能删除后仅依赖 reference.json 恢复。旧实验快照和原始成绩不会自动迁移或改写。

可通过 `controller_tools.py` 的 `compose_task` 工具生成控制变量评测状态，例如：

```json
{
  "operation": "compose_task",
  "versions_root": "/path/to/run/versions",
  "template_state_path": "/path/to/snapshot/task/task_state_snapshot.json",
  "model_id": "model_001",
  "harness_id": "harness_002",
  "destination": "/path/to/run/ablations/model_001_harness_002.json"
}
```

模板提供其余 Task 字段（包括 Artifacts），仅替换指定模型和 Harness。返回的 `state_path` 可传给 `prepare_validation_snapshot`，再交给 `evaluate`；不会自动激活这个组合。编号属于各自实验的版本库，不是轮次号。

## 训练接口与评分修复（9.25）

- 工具联合类型在 HF Tool.inputs 投影成一个支持的类型，原 JSON Schema 保留供实际参数校验，因此 nullable/多类型字段不再在模型调用前被错误拒绝。
- TACO call-based 评分按[官方 execute_cb_code](https://github.com/FlagOpen/TACO/blob/main/metrics/testing_util.py)支持完整返回值或外层包装内返回值比较；不递归展平、不改变 STDIO 评分。训练 verifier 标识改为 `v2`。
- 新版父代、候选都应使用修复后的同一执行器和评分器重新测量；不要把新成绩与旧版错误标签直接配对，或把工程修复带来的增分记作 Meta 的进化收益。独立 benchmark 官方评分器不变。

## Meta 条件经验、反例与追加修订（9.26）

`meta/skills.jsonl` 仍是累积经验账本，增加 `case`（干预案例）、`rule`（条件原则）、
`incident`（已定位的接口/评分故障）、`rule_update`（追加修订）；旧 `skill/principle`
兼容读取，不覆写历史。`skills_index.json` 是可重建目录，快照继续保存账本版本。

`prepare_meta_evidence` 增加观察性故障指纹；路由前用 `retrieve_skills` 获取相关支持、
反例与规则，必要时用 `read_skill` 及轨迹引用读取完整证据。默认每组件每类最多取 1 条，
单次检索回执预算 24,000 字符；不限制经验库总条数，也不改变原 48 条片段的筛选。
候选生成前 `record_skill_use` 保存引用 ID 与事前预测，复测后 `compare_task_differences`
按任务 ID 列出新成功、新退步、错误/成本变化和原始证据引用，帮助 Meta 追加有边界的经验。
工具不替 Meta 决定组件、候选、接受或规则可信度，没有新增闭环完成门槛。

新增 `maintain_skills`：Meta 根据新证据选择新增、补证、关联版本修订、归并或停用。
具体技能为 procedure，维护为追加事件；旧记录和旧成绩不覆盖，默认检索只提供当前
active 版本，历史仍可按 ID 或 include_inactive 查阅。补证不新建近似技能，修订不把
旧版实测结果当新版支持；归并保留反例和来源。工具不自动判定语义相同或替 Meta 路由。

历史复核种子位于 `meta_skills/reviewed_seed.jsonl`，可通过 `--skill-seed` 显式用于新实验；
默认初始经验仍为空，已有实验恢复时不重放种子。配置为 `meta_skill_context_chars`、
`meta_skill_per_category`、`meta_skill_seed_path`。完整工具用法及证据边界见 [META_SKILLS.md](META_SKILLS.md)。

## 启动与续接

Meta 记录统一入口为 `../Meta_logs/<实验名称>/`：`meta_session/` 保存 Codex 交互事件、逐次请求/回执、聊天消息、状态和本会话 Codex HOME；`meta/` 保存经验库与交接上下文。`controller_receipts/`、`controller_transport/` 链接到运行目录中的工具记录与作业日志。

新实验的 `meta_session/` 和 `meta/` 直接写入 Meta_logs，原 `Rollout_logs/runs/<实验名称>/` 下保留兼容链接；已在运行的旧实验只增加反向入口链接，不移动活跃日志、不复制内容、不重启进程。可通过 `meta_logs_root` 配置覆盖默认根目录。Task 轨迹、模型和评测数据仍在原位置。Meta 原始交互可能包含任务信息，不应未经检查直接公开上传。

```bash
cd /root/data/RSI_iclr2027/rsiH/RSIFlow_8B
/root/data/conda/envs/sia/bin/python launch_meta.py --check
bash start_meta.sh --run-dir /root/data/RSI_iclr2027/rsiH/Rollout_logs/runs/rsiflow_8b_codex_180_a0_v1
```

同一命令和运行目录用于断点续接。默认配置为 `configs/train_180_a0_v1.json`；Task 数据从现有冻结数据集只读，模型权重与原始轨迹不复制到代码仓库。密钥优先读 `AUTODL_API_KEY`，否则从本机 `RSIFlow_4B/API_key.md` 读取；不会写入 Codex 配置或回执。

四卡暂时被其他实验占用时，加 `--wait-for-gpus`：后台每 60 秒检查 GPU 0–3，连续两次
每卡显存占用不超过 1024 MiB、利用率不超过 5% 后启动。不停止其他 GPU 作业；等待
状态在 `meta_session/gpu_wait.json`，日志在 launcher.log。自有旧推理服务仍占显存时需先
释放，否则等待器会持续等待。新实验使用独立 --run-dir，默认 Meta 初始经验为空；
需要显式继承时另加 --skill-seed。

`experiment` 是 Codex 的原生工具，而不是模型输出一段命令 JSON 后退出。长任务返回 `job_id`，Codex 通过 `wait_job` / `job_status` 等待并检查结果，进程保持运行；回执错误返回同一会话，不自动终止实验。`finish_experiment` 只在必要产物齐全时结束。

Meta 的工作目录为本项目的上级 `rsiH`，原生 shell 和 apply_patch 已启用。Meta 可检索证据、编写调试脚本、修复或添加工具实现、恢复任务、修正快照；`experiment` 等封装工具是可选的便利接口，不是唯一能力。桥接层不再拦截新增 operation；每次调用会启动新工具进程，读取当前工具代码。`materialize_harness` 仍复制当前父代的完整 Harness，不恢复成初始版、不限制为七个文件。

本机不能使用 bubblewrap 挂载，改用已实测的 Codex Landlock 兼容模式及 `workspace-write`，不设置整机无限制模式。临时文件在运行目录，原始证据与评分方法不因工程修复而改写。`persistent_meta.py` 为新建及恢复的线程显式设置原生环境，实际参数记录于 `meta_session/app_server/thread_parameters.json`。兼容模式针对当前安装的 Codex 版本测试，升级二进制后需重测。

Task rollout/SFT/评估仍由宿主机工具子进程使用 GPU 执行，并不变成 Codex 内部沙箱任务。常驻进程解决会话执行方式，不等于建立了 OS 级安全隔离，也不能保证候选永不出错。错误与成绩仍供 Meta 判断。

切换旧实验时会一次性创建带原生工具的新线程，保留旧线程 ID、工具回执和全部证据，并接管已运行任务，不重新执行。后续阶段共享该原生线程及进程；若进程意外退出，重启入口会恢复该线程并连接已有任务。监控文件：`meta_session/state.json`、`RUN_STATUS.md`、`app_server/events.jsonl`、`launcher.log`。

## 实时观察 Meta（不启动新实验）

```bash
cd /root/data/RSI_iclr2027/rsiH/RSIFlow_8B
bash watch_meta.sh
```

默认跟随当前配置的实验：显示最近 12 条可见回复/工具回执，随后流式显示 Meta 的回复、工具调用与结果（包括原生命令与文件修改）；每 30 秒展示 Codex PID、主线阶段及后台任务状态。`Ctrl+C` 只关闭监控，不停止实验。

- `bash watch_meta.sh --once`：只查看一次状态。
- `bash watch_meta.sh --history 30`：查看更多近期记录（从日志末尾最多 4 MiB 提取）。
- `bash watch_meta.sh --history 0`：只看新输出。
- `bash watch_meta.sh --run-dir /绝对路径/另一个实验`：指定实验。

监控只读取 `state.json`、`workflow_state.json`、`jobs/*/job.json` 和 `app_server/events.jsonl`；不调用 API、不恢复会话、不更新账本。它显示可见回复和工具事实，不显示内部推理。`running` 表示任务尚未结束，不是成功；`tool_error` 可能只是读取了尚未生成的文件，需要结合后续回执判断。

API 接线：`configs/codex_host.toml` 中的 `model=DeepSeek-V4.1-Flash`、`model_provider=autodl`、`base_url=https://www.autodl.art/api/v1`、`wire_api=responses`。Codex 用 `AUTODL_API_KEY` 调用该服务商兼容接口，把会话和工具结果送给模型；模型返回回复或工具调用，Codex 调用本地注册工具、接收结果后继续请求模型。Task 的 Qwen3-4B 推理使用本地四卡服务，与此 Meta API 分开。

## 边看输出，边向同一个 Meta 发消息

```bash
bash /root/data/RSI_iclr2027/rsiH/RSIFlow_8B/chat_meta.sh
```

输入文字后回车发送；`/status` 查看当前会话和最近消息投递状态；`/quit` 或 Ctrl+C 只退出聊天界面，不停止 Meta 或 GPU 任务。支持 `--run-dir` 和 `--history 30`。只发送一条消息可用：

```bash
bash /root/data/RSI_iclr2027/rsiH/RSIFlow_8B/chat_meta.sh --send '请解释本轮修改的原因，回复后继续当前实验。'
```

消息写入运行目录下 `meta_session/chat/`，现有 Meta 进程通过官方 `turn/steer` 接口投递到当前线程和当前 turn；聊天界面不会启动第二个 Codex，不会重新运行实验。界面同时跟随原有事件日志。工具正在执行时回复可能延迟；`accepted` 只表示 Codex 接收了消息，不表示要求已执行。尚未投递的消息断线后保留，确认前断线的消息标记 `delivery_unknown`，不会盲目重复发送。实验结束或 Meta 未运行时消息不会凭空启动新任务。

首次升级此入口需要重载一次 Meta 通信进程并恢复原线程，已启动的独立 GPU 作业不重启。之后每次发消息都不需要重启。`watch_meta.sh` 继续保持纯只读。

## 验证边界

```bash
PYTHONPATH="$PWD/runtime:$PWD" /root/data/conda/envs/sia/bin/python -m pytest -q tests
```

已做离线编排测试、Qwen3-4B A0 初始化冒烟和 DeepSeek/Codex 原生工具实测：同一进程跨两次 turn 连续四次工具调用，其中一次故意报错后继续成功。离线及小型 API 结果不等于三轮真实 GPU rollout、SFT 与官方评测已经跑通；启动前应确认另一四卡实验已结束。
