# Debug 记录：180 条混合任务实验恢复

日期：2026-09-21。此记录区分代码检查、实际执行与尚未完成的实验结果，不代表三轮实验已全部成功。

## 当前实验口径

- 三轮，每轮三个领域各 60 条，共 180 条；轮间任务不重叠，轮内父子代使用同一批任务。
- 每轮只选择一个组件、最多一个候选；完整可比且成功率严格上涨才接受。
- MODEL 只用符合条件的父代成功轨迹做一轮四卡 LoRA SFT。
- 路由仍使用全部任务统计与 48 条分层筛选的代表轨迹片段。
- 每轮保存 Task/Meta 状态，随后做 300 条独立、仅报告用途的验证。

## 1. Task 上下文失败不再中断整轮

问题：Task 自身超出固定上下文预算，被当成整场实验的基础设施错误，评分汇总无法继续。

处理：已识别的上下文预算拒绝记为 `context_budget_exhausted`，任务计失败，保留请求和此前轨迹，仍进入分母和 Meta 证据。不增加模型上下文上限；未知服务、传输和评分器错误不一概当作正常失败。

另对已复现的 EnvScaler 特定任务检查器类型错误，按任务标识、原始检查代码哈希和异常条件精确识别，保留 `task_state_type_mismatch` 证据。不扩大为忽略任意评测异常。详见 [180 条实验说明](EXPERIMENT_180.md)。

## 2. Meta 证据容量与可变工作区预算混在一起

问题：第一轮已完成候选训练和配对评估，但 Meta 经验入库前，重复嵌套轨迹的比较文件约 32.85 MB，触发原有 16 MB 文件/工作区检查，报 `nonregular_or_oversized_file`。当时尚未发出该阶段的模型请求。

处理：

- 原始轨迹保留在 `Rollout_logs`；证据用内容哈希、任务标识和路径引用，不重复塞进 skill。
- Meta 按需读取完整训练轨迹；摘要和内联经验有界，历史详情不因此删除。
- 只读证据、经验输入、控制文件、可写输出分别统计并限制。
- 默认只读证据总预算 2 GiB，单证据文件 128 MiB；经验输入 256 MiB；控制文件 16 MiB。可写输出仍使用原有独立预算。
- skill 文件容量和条目数独立校验；已有经验追加保留，不靠静默截断腾空间。
- 同一份预算规则用于准备、沙箱输入检查、运行期检查和结果回收。

验证：旧比较文件压缩为约 1.69 MB 的摘要与引用；离线检查确认展开后的证据一致。实际恢复调用准备了约 918 MB 只读证据，不等于把 918 MB 一次性发送到模型上下文。真实沙箱检查确认大证据可读、不可修改，输出另行限额。

实现入口：`meta_backends/input_budget.py`、`meta_harness/evidence_delivery.py`、`meta_harness/read_evidence.py`。详见 [Meta 经验与证据](META_MEMORY.md)。

## 3. 系统盘容量与隔离文件系统兼容

问题：系统盘接近满载，主要是旧 `/tmp/rsi-native-rootfs`（约 20 GiB）及多份闲置 Meta 隔离环境，不是当前模型检查点放错位置。

处理：检查进程占用后，将旧环境迁到 `/root/data/RSI_iclr2027/.runtime`；普通实验临时文件、缓存、日志和持久结果放数据盘。迁移保留文件，不删除实验结果。

兼容性发现：数据盘使用 Lustre，隔离 UID 访问时出现 `Operation not supported`；同时 `/root` 的 0700 权限不能为迁移而放宽。因此当前 Task/Meta 的必要隔离执行目录保留在 `/tmp`，不更换已经验证过的隔离机制。Meta relay socket 必须和沙箱位于同一文件系统，才能进行硬链接。

启动前检查系统盘已用空间加 4 GB 预留不超过 20 GB（十进制）；Meta 调用前和执行期间也检查。此机制不是能约束其他程序的整机磁盘配额。实测迁移后启动新 Meta 时约 4.7 GiB。

真实隔离检查通过：Codex 可执行、relay 可达、能力清除、外网及宿主根目录访问被拒绝、禁止修改运行时、工作区允许写入。Task 的 16 worker 沙箱及官方评测兼容测试也通过。详见 [存储说明](STORAGE.md)。

## 4. 从已部署 Task 的边界恢复

已完成并保留的第一轮结果：父代 18/180，子代 27/180，成功率 10% → 15%，绝对增益 5 个百分点；接受 MODEL 更新。该涨分发生在恢复之前，不能记为本次工程修改带来的新涨分。

恢复入口 `resume_experiment.py` 的范围：

1. 校验已接受 checkpoint 的实际权重哈希、原始协议和前后证据。
2. 将旧协议备份至运行目录的 `recovery/protocol_before.json`；记录授权的代码版本和明确的协议修订，不静默覆盖历史来源。
3. 仅允许本次声明的工程恢复与默认 Meta 输入预算迁移，不改数据、模型、三轮配置或接受规则。
4. 跳过已完成的父代 rollout、候选生成、SFT 和配对复测，补 Meta1 经验提交。
5. 提交第一轮快照，完成独立验证，再继续第二、三轮。

`recovery/authorization.json`、`protocol_revision.json` 和 `effective_config.json` 只保存在对应运行目录，不作为源码提交。原始权重、轨迹和结果不进入 Git。

恢复当前服务器上的这个断点：

```bash
cd /root/data/RSI_iclr2027/rsiH/RSIFlow_4B
RSIFLOW_CONFIG="$PWD/configs/train_180.json" \
RSIFLOW_RESUME_DEPLOYED_RUN=/root/data/RSI_iclr2027/rsiH/Rollout_logs/runs/rsiflow_4b_ds41_180_fresh_v1_r3 \
bash start_3round_training.sh
```

只在没有正在运行的同一实验进程时使用；入口和原生控制器会获取对应锁。Git clone 不包含该历史断点，不能在空运行目录上使用这个恢复命令。新实验按 `EXPERIMENT_180.md` 准备外部数据后启动。

## 5. 发布前新发现：Meta 经验引用校验失败（14:58:27 CST）

恢复的 Meta 首次生成和一次修复均已执行，但最后没有通过经验候选校验，控制器退出，`active_run.json` 变为 `failed`。Task1 权重及 18/180 → 27/180 的配对结果未被回滚或覆盖，Meta1 尚未正式入库，独立验证也未开始。

确切触发点：`recursive_feedback.py` 的 `validate_recursive_memory()` 要求每条新增通用原则的 `principle_operations[].rationale` 包含本轮新增组件 skill 的完整 ID。候选已生成 `skill.MODEL.sft_parent_successes_positive_transfer`，但通用原则的 rationale 仅写了“derived from the new skill evidence”，没有写完整 ID；修复阶段修正了原则 ID 的 `principle.` 前缀，却仍未补足这条精确引用。

最终错误：`General principle rationale must derive from a newly appended component skill`；同时出现的 `Native candidate delivery has not passed; retained repair input is not an accepted candidate` 表示留存的修复输入没有被标记为通过，不是新的 API 或磁盘错误。

这次故障说明：基础设施已能完成真实模型调用，但经验生成/修复与确定性引用契约仍未闭环。待处理方向是让生成和修复明确携带新增 skill ID，并补充回归检查；不应直接绕过验证器或伪造经验已入库。此次发布请求附带“正常才上传”条件，因此发现故障后暂停 commit/push，未将此版本作为正常运行版发布。

## 6. 经验 Prompt 契约遗漏修复

根因：`RoundProtocol.enrich_effect()` 先设置基础 `instructions`，随后单候选实验分支整段替换该字段，遗漏了通用原则必须在 operation.rationale 中引用本轮新增 skill 完整 ID 的要求。校验器仍执行原规则，因此不是 API 失败或 Task 新的推理问题。

修复：在公共代码中定义 `MEMORY_ID_CONTRACT`，各分支结束后统一追加；原生 Meta 首次总结和修复调用均将大写必填要求放在有界证据 JSON 之外。组件经验使用 `skill.<COMPONENT>.<unique_id>`，通用原则使用 `principle.<unique_id>`，引用需要说明依据。成功和失败经验均只追加，已有记录保持不变；不绕过校验，不硬编码当前实验 ID，不修改 Task 候选数或接受规则。

验证：`test_memory_prompt_contract.py`、`test_recursive_feedback.py`、`test_append_memory.py`、`test_meta_round_context.py` 合计 **20 passed，16 subtests passed**。覆盖两种提示分支和后续轮次、真实原生调用路径中的首次总结/修复（模型调用使用测试替身）、内联数据被压缩时仍保留要求、成功/失败追加和历史记录不变。未调用真实 API，不代表 Meta1 已入库。

当前没有重启实验或替换旧经验文件。当前断点的恢复授权绑定旧控制器代码哈希；修改后的代码恢复前需记录新的授权修订，不能直接覆盖原授权或将旧候选标为通过。第 4 节命令不能在未完成该步骤时当作已验证的直接恢复命令。全新实验使用更新后的公共代码，不依赖这个历史断点。

## 7. 追加经验校验与提前 rollout（同日后续修订）

15:35 左右的真实修复仍未入库：首次总结将新 skill 的 evidence_state 写成 supported，通用原则 ID 也缺 principle. 前缀；原生校验的第一个异常跳过了独立的组件经验校验，修复只看到初始状态错误，补好后才暴露前缀错误。失败候选与调用证据均保留，未伪造提交。

修复：原生校验与实验追加校验独立执行，前者失败不再遮蔽后者；修复一次收到两个检查层已发现的错误。错误指出具体操作、ID 和正确格式。公共提示补充新增记录 revision=1、active=true、evidence_state=tentative，即使本轮实测涨分也不能直接把可复用原则标为普遍成立；强调只追加机制、条件、结果与证据引用，不复制整轮轨迹。现有历史记录仍不可覆盖。

依用户要求新增提前 rollout：Task 接受/保留决定确定后，独立进程在下一轮已划定训练任务上生成原生断点记录，与本轮 Meta 学习并行。父代 rollout 的行为身份不再依赖尚未更新完的 Meta 经验；模型、Task Harness、Artifacts、任务清单、seed、协议和实现仍绑定。Meta 路由仍在上一轮经验提交后执行，使用新 Meta；本轮 Meta 看不到下一轮证据。控制器在独立验证或后续训练前等待提前 rollout 完成，避免 GPU 阶段重叠。暂停边界和最后一轮不额外启动下一批。

实现：`early_rollout.py`、`sequential_loop.py`、`deployed_recovery.py`、`round_evolution.py`、`pipeline.py`。保持三轮、180 条、48 条路由样本、一个候选、严格涨分和独立验证。提前 rollout 使用本轮已确定的 Task，下一轮主流程复用同目录的 execution_receipt，不重复执行。

验证：9 个相关测试文件合计 **60 passed，16 subtests passed**，包含证据无损引用、预算分离、分页读取、原始训练档案/验证隔离、经验追加与跨轮上下文、同时反馈两层错误、提前 rollout 的身份一致与 GPU 等待顺序。模型调用为测试替身，不等于真实 Meta1 入库。

恢复修订保存在运行目录 `recovery/revisions/append_checks_early_rollout_20260921/`，归档前次协议与授权，不覆盖 Task1 成绩、检查点和已接受经验。新控制器 PID 195039 已启动；真实入库与提前 rollout 完成结果仍需看运行记录。

## 8. 独立评估的生成/评分接口修复（2026-09-21）

后续真实状态更新：Meta1 已正式入库 v001_043bc63ff8d4，保留旧 5 条经验、追加 MODEL skill 和通用原则各 1 条，共 7 条；第二轮提前 rollout 已完成 180 条。随后第一轮独立验证在 BFCL 发生 RPC 错误并退出，后六项未运行。以上为最新状态，前文的早期故障状态仅作历史记录。

根因：OfficialTurn 没有 evaluate 方法；Task 已产生答案，但 Harness 终止时调用该方法产生 AttributeError，两层包装仅保留 RuntimeError。追加本地原始错误信息，并以独立 official_evaluation.py 返回 pending_official、reward=None、success=None；官方外层负责实际评分。待评分回答不进入结果型 Memory 归纳。

七项接线复核发现 Code/Search 分支仍调用训练终局评分：公开 Search 输入刻意不包含 gold，因而产生 invalid_reference_answers；此外 Harness 的 _evaluation 是 AdapterResult，不能直接 JSON 序列化。report_environment.py 保留公开工具、延迟评分并显式序列化回执。训练环境评分未改变。

评估已有独立 validate.py 进程入口和 round_validation.py 轮次调用接口；补充 start_evaluation.sh 与 EVALUATION.md，使同一评估也能单独调用，不启动 Meta 或训练。保持固定 300 条清单、七个官方评分器、报告独立性与完整分母，不跳过异常任务冒充评分完成。

新增测试使用真实 Harness 执行（仅模型为离线替身），覆盖原先缺失的终止接口、待评分标记、Memory 不误写、五类 Code/Search 公开环境及 JSON 保存、原始错误保留。另复核官方比较器、固定分母、四卡分片、经验追加与提前 rollout 回归。详细检查结果见 EVALUATION_AUDIT.md。

未重启实验、未覆盖旧结果、未修改已提交经验。旧失败请求仍需审计恢复；修复源码不代表旧进程已热更新，也不代表 300 条实际评测已成功。

## 验证与未完成事项（此前发布检查记录）

- 发布副本中本次选定的 9 个测试文件：40 passed，11 subtests passed。测试从 `RSIFlow_4B` 目录启动，并设置 `PYTHONPATH="$PWD/runtime:$PWD:$PWD/tests"`。从仓库根目录直接调用会优先导入保留的旧 `sia` 包，导致收集阶段的 `ModuleNotFoundError`；不能把错误导入路径的测试结果当作当前 runtime 的验证结果。
- 恢复、快照、启动器及经验增长组合测试：20 passed，11 subtests passed。
- 存储、启动器、Meta 输入及经验组合测试：25 passed，11 subtests passed。
- Task 沙箱并发、官方评测兼容、Harness 候选验证：6 passed。
- 真实大证据沙箱读写检查和本机隔离自检通过，检查本身不调用模型 API。
- 恢复后，首个真实 Codex 阶段正常退出（退出码 0），13 次 API 请求均 HTTP 200，经验候选已生成；随后进入后续 Meta 处理。

以上测试组合有重叠，不应相加视为不同测试数量。单元测试通过不代表真实 Meta 的自由生成输出一定满足全部契约；当前已因第 5 节的新故障停止。不能把经验候选生成等同于入库完成，也不能宣称第一轮独立验证或三轮自动迭代已经完成。

查看实时进度：

```bash
tail -f /root/data/RSI_iclr2027/rsiH/Rollout_logs/logs/rsiflow_4b_ds41_180_fresh_v1_r3_resume.log
```

本次发布不修改运行中的模型、轨迹和源码执行文件；Debug 文档为独立记录。API 密钥和 GitHub token 始终排除在源码提交之外。

## 9.22 新实验启动时隔离目录权限修复

9.22 当前版本的问题是 Meta 隔离根目录在控制器较严格的 `umask` 下生成了仅 root 可遍历的静态子目录；Codex 进入非 root 的 chroot 后可能在读取运行文件时遇到权限错误。这是目录权限继承问题，不是 DeepSeek 决策或模型输出错误。

已在隔离根目录构建后规范静态目录的遍历权限，保留工作区等可变目录原有权限；增加本地执行回归测试。该修复只影响 Meta 隔离环境的文件可达性，不修改 Task、评分、接受规则或经验内容。离线测试通过；新三轮实验的真实闭环仍以运行记录为准。
