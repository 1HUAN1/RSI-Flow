# Task–Meta Co-Evolution MVP 第二版修复说明

日期：2026-09-06。项目：`/root/data/RSI_iclr2027/rsiH`。

本文说明根据上一版 Pipeline 审阅意见所做的工程修复，供研究方案审阅与实验复核使用。目标是让状态、资产、单组件修改和反馈时序对应清楚。本次不评价 Meta 自更新的质量，也不以成功率上涨作为闭环完成条件。

**完成状态：五轮真实自主运行 `run_4001` 已正常结束，证据审计通过。正确数为 0/16、0/16、3/16、12/16、13/16；四次自主动作是 HARNESS、HARNESS、MODEL、MODEL。第一次 MODEL 已完成训练、真实下一代推理、评分、经验生成、真实 Meta 学习和新版 Meta 后续路由的完整链路。**

运行结果为 `status=completed`、`stop_reason=max_generations`，5 次执行、4 次实际修改、4 条经验、4 次常规 Meta 更新、1 次最终总结；最终 Task 为 T4，Meta Harness 为 v5，没有 T5。主循环耗时约 31 分 47 秒。结果仅用于工程闭环验证。

## 1. 核查确认的旧实现问题

| 问题 | 旧行为及影响 | 本版处理 |
| --- | --- | --- |
| 分数和状态混写 | 执行后将新资产写回同一个 Task 状态，使 `state_before` 难以区分产生分数的输入与 Meta 实际修改的起点。 | 显式保存执行输入、执行输出和干预后下一代输入；经验中分别引用。 |
| 同名资产覆盖 | 同一运行目录中的 `generated_strategy.md` 被不同题目后写覆盖；轮末选最后一个 rollout 的资产目录。 | 按题目和 rollout 隔离产物，登记所有当前轮产物；不以最后一次输出代表整轮。 |
| 旧资产混入输出 | 输入资产与输出资产共用工作目录，旧文件可能因没有重新生成而继续存活。 | 输入冻结，输出独立；下一代默认只使用当前轮真正产生的输出资产或其显式 A 修改版本。 |
| 决策目标过于模糊 | 自然语言决策可能同时要求修改机制与恢复知识，而实际 H 补丁只改了两个函数。 | 请求显式列出组件、操作和目标；未实现、额外实现或跨组件请求触发工程约束反馈。 |
| Harness 实现范围过窄 | 将可修改 Harness 约束在 `format_question`、`parse_answer`，不足以表达资产使用与重试策略。 | 最小扩展为五个机制钩子，公开具体可执行范围；复杂外部工具与检索仍未实现。 |
| MODEL 未完成主循环接线 | 独立训练成功不能证明训练后的模型进入下一代、产生经验并被 Meta 学习。另有生成但不执行的大段训练代码阻塞固定后端。 | 主循环直接调用固定正奖励 SFT 后端，记录当前 checkpoint 起点、新权重、实际服务绑定与下一代请求。真实全链路是否完成以本次运行证据为准。 |
| 新 Meta Harness 未覆盖所有阶段 | 新版本参与后续 routing，但 H/A 具体补丁生成缺少同版本 Meta 工作规则。 | 所有 Meta 调用通过同一包装层接入所属版本，并记录版本、hash 和决策/经验 ID。 |
| Meta 观察不完整 | 顺序截取少量成功/失败例子，资产只显示文件名与 hash；截断、解析错误与答错容易混淆。 | 优先提供完整必要轨迹与实际资产内容；压缩按题目/结果分层，记录覆盖情况；提供独立执行指标。 |

旧 `run_3002`、`model_branch_3002` 及其 checkpoint 保留为历史记录；它们不作为第二版修复后结果。旧报告中的 `0/16、6/16、16/16、0/16、0/16` 是旧五轮中每轮 16 次作答的正确数，不是五次训练的指标，也不能据此确认或否定某一个组件的独立因果贡献。

## 2. 系统定义与五轮时序

Task-Agent 为 `T = (Model, Harness, Artifacts)`；Meta-Agent 为 `M = (固定 Model, 可修改 Harness)`。

Task 使用本地 `/root/data/zh/huggingface/Qwen2.5-3B-Instruct`。Meta 保持现有 OpenRouter `z-ai/glm-5.2` 配置。凭证由环境变量提供，不写进代码、实验报告或 Prompt。

对于第 t 次执行，显式区分：

```text
T_t_in = (theta_t, H_t, A_t_in)

Execute(T_t_in, 同一固定任务批次)
    -> trajectories_t, p_t, A_t_out

T_t_out = (theta_t, H_t, A_t_out)

Meta 只选择一个 action ∈ {HARNESS, MODEL, ARTIFACTS}
T_(t+1)_in = U_action(T_t_out)
```

`p_t` 关联 `T_t_in`。单组件变化检查针对 `T_t_out → T_(t+1)_in`。不要求 `T_t_in → T_(t+1)_in` 只变化一个组件，因为正常执行本身会产生或失去资产。

五轮包括初始执行，具体为：

| Task 执行 | 执行时已有的 Meta 版本 | 本轮结果反馈与 Meta 学习 | 后续动作 |
| --- | --- | --- | --- |
| T0 | M0 | 得到 p0，无可计算的前次修改经验 | M0 自主选择并实施一个修改，生成 T1 输入 |
| T1 | M0 | 得到 p1，形成 e0；M0 学习为 M1 | M1 自主选择并实施一个修改，生成 T2 输入 |
| T2 | M1 | 得到 p2，形成 e1；M1 学习为 M2 | M2 自主选择并实施一个修改，生成 T3 输入 |
| T3 | M2 | 得到 p3，形成 e2；M2 学习为 M3 | M3 自主选择并实施一个修改，生成 T4 输入 |
| T4 | M3 | 得到 p4，形成 e3；M3 学习为 M4 | 一次最终总结，将 M4 更新为 M5，结束 |

正常完成时必须是：5 次 Task 执行、4 次实际 Task 修改、4 条经验、4 次常规 Meta 更新、1 次最终总结。不生成 T5。Meta Harness 最终版本 v5 不代表额外执行了 Task 第六轮。

预算在代际边界检查；已经完成的执行先落盘，并形成其对应经验。最终总结是否完成由 `final_consolidation_status` 如实记录。失败不能被写成正常完成。

观测分差仍为 `observed_performance_delta = p_(t+1) - p_t`。它可能同时受到资产自然变化、任务采样与所选修改的影响。没有新增成对重跑或三分支比较来分离这些因素。

## 3. Artifacts 的输入、生成、登记与交接

初始 `A0_in` 为空。每轮在 `artifacts_input` 保存冻结快照，所有题目和所有 rollout 从同一份输入出发。一条 rollout 的新笔记不会进入本轮另一条 rollout 的输入。

新产物写入各 rollout 工作目录的独立树：

```text
artifacts_generated/<安全 task_id>/rollout_<rollout_id>/strategy.md
artifacts_generated/<安全 task_id>/rollout_<rollout_id>/strategy_attempt_<n>.md
```

文件路径含题目和 rollout 归属。重试产生的笔记也有独立文件名。登记过程校验文件与轨迹声明一致，并避免同名覆盖。

轮末按确定性规则登记所有当前产物，形成 `A_t_out`。不让 LLM 合并，不按终局奖励挑选，不把最后一次 rollout 作为整轮资产，也不自动继承没有重新产出的旧文件。资产按规则过期会记录在生命周期差异中，不能被描述成意外覆盖。

下一代输入为当前输出资产，或者 ARTIFACTS updater 对当前输出执行新增、修改、删除后的版本。若 Meta 希望恢复已过期内容，必须明确选择 ARTIFACTS；HARNESS 只能改变如何读取、使用或生成资产。

每份产物记录来源题目、rollout、hash、生成方式与对应终局奖励。`knowledge_verified=false`：正奖励证明这次作答成功，不证明笔记的每项知识正确。历史来源无法可靠匹配当前 hash 时标记未知，不猜测来源或奖励。

轨迹、日志、外部评分输出、评分答案文件和运行控制文件不属于资产。旧快照保留用于追溯，不自动成为长期知识库。

最终结果同时保存：

- `last_evaluated_task_input`：实际取得最后分数的输入状态。
- `last_output_artifacts` / `last_task_output_state`：最后执行产生的输出资产和状态；这些资产可能还没有被下一次执行使用。

## 4. Meta 决策与实际补丁的对应关系

自主运行中每次 routing 都由 Meta 分析当轮轨迹、结果、资产变化、历史经验与当前能力后选择动作。没有 H→W→A 固定顺序，没有先跑三条分支后选最优，也没有满分停止或性能回滚。

决策含 `action`、`target_components`、`requested_changes` 和 `expected_effect`。每个 requested change 必须包含独立 ID、组件、操作、精确目标和修改说明。`target_components` 必须恰好是所选组件。

| 分支 | 本版支持的实际操作 | 受保护边界 |
| --- | --- | --- |
| HARNESS | `replace_hook`：修改 `format_question`、`parse_answer`、`harness_config`、`retry_prompt`、`select_artifact_context` 中明确请求的函数体 | 不修改当前资产内容、checkpoint 绑定、数据集、外部评分器、凭证、日志与资产隔离 I/O |
| ARTIFACTS | 对明确列出的相对路径执行 `write_asset` 或 `delete_asset` | 不修改 Harness、模型、评估器、运行控制文件；不允许额外未声明路径 |
| MODEL | `sft`，目标 `current_checkpoint`；调用固定正奖励 SFT 后端 | 不改 Harness 或当前资产；不得虚构样本、训练空数据或重置为初始权重 |

`harness_config` 当前可表达是否使用/生成资产、资产输入字符上限、是否对解析/API 失败重试以及最多尝试次数。它没有开放任意文件、工具或网络执行。Task 与 Meta 的 Harness 使用同一概念定义；具体可修改实现和权限范围不同。

在提交前检查：组件一致、操作与目标有效、声明目标是否真正变化、是否修改额外目标、路径是否合法、受保护代码是否保持，以及 checkpoint 接口是否满足约定。每次 routing 最多经过三次工程约束尝试，未提交的失败不会算作 Task 更新。

例如，HARNESS 决策若要求直接恢复知识文件，应当修正为可执行的单组件请求；不能把恢复要求悄悄丢弃。对自然语言预期效果无法机械证明的部分记录 `semantic_status=unverified`。生成补丁不等于所有语义目标已被验证。

`task_update.json` 分别保存 `requested_changes`、`applied_changes`、`unapplied_changes`、修改文件与前后 hash、工程检查结果。结构合法的 Meta 自更新直接接受，不再评估新 Meta 是否更好。

## 5. MODEL 接入同一主循环的实现与完成判据

MODEL 的实现路径为：

```text
Meta 选择 MODEL
 -> 当前 rollout 中可用的正奖励文本样本
 -> training_request.json
 -> 固定 GPU LoRA SFT 后端
 -> 保存合并后的新 checkpoint
 -> 注册并加载到本地推理服务
 -> 下一代实际请求新 model_ref
 -> 同批任务重新执行并评分
 -> MODEL 对应 ImprovementExperience
 -> 真实 Meta 学习更新 Harness
 -> 新版 Meta 继续 routing
```

固定后端不再要求 Meta 生成一份不执行的训练程序。训练起点来自当前 Task checkpoint；连续 MODEL 更新应从上一次训练产物继续，而非重新读取最初的 Qwen 权重。

无可用正奖励完整对话、未配置训练器或当前本地 checkpoint 不可用时，routing 输入明确显示 MODEL 不可执行及原因。不会伪造正样本，也不会把准备好的请求称为成功训练。

当前 SFT 使用 Hugging Face Trainer 与 PEFT LoRA。loss 只监督最后一条已记录的 assistant 响应；此前对话 token 掩码。保存可用正样本数、实际采样样本及去重数量、监督 token 数、optimizer steps、训练参数与权重变化证据。每题 8 次 rollout 不代表 GRPO；8 个 optimizer step 不代表完整遍历全部正样本。

验证模型使用不能只看模型列表。需同时对应：训练输出路径与权重 hash、服务加载绑定、下一代 `model_ref_requested` / `model_ref_response`、每次响应的 checkpoint 绑定、下一代分数与经验中的版本引用。

独立 SFT loss 下降或权重变化，只能说明训练执行过。只有上述推理、评分、经验、Meta 学习和后续 routing 连起来，才可称为 MODEL 工程闭环完成。

若五轮自主运行没有选择 MODEL，使用独立 integration test，在同一个主循环注入一次测试决策并执行真实训练和推理。该决策保存 `decision_source=test_override`，后续恢复真实 Meta routing。它只能证明 MODEL 接线，不是 Meta 自主选择 MODEL 的证据，也不会改变默认自主运行策略。

## 6. Meta 实际看到的信息与调用审计

Meta 可看到本轮执行输入与输出状态、输入/输出资产内容或明确截断的内容、来源与 hash、自然资产变化、上一轮实际补丁、每题及整体结果、执行预算、可用动作和历史经验。

当前 16 条 rollout 在证据预算允许时全部提供；超预算后按题目与观测结果类型确定性分层。`observation_coverage` 记录包含/遗漏的轨迹 ID、截断字段及原长度/hash。不会把“只给了少量样例”写成“分析了所有完整轨迹”。

执行指标区分成功率、有效答案率、解析失败、输出截断、模型/API 失败，以及可获得的 token 和耗时。有效但答错与没有生成有效答案不是同一类现象。Meta 对机制原因的解释是诊断假设，不是已验证错误标签。

本次 GPU smoke profile 将 Task 每次请求输出上限设为 512 token；temperature=0.7、seed=42、固定两题、每题 8 次 rollout。基础评估设置在同一 run 内固定落盘。Harness 可以在声明范围内修改任务内重试策略，该变化本身需作为 H 修改记录。

所有 routing、H/A 补丁、Meta 学习与最终总结，通过同一 Meta 调用包装层明确使用当前版本。每次记录 `operation`、`meta_harness_version`、`meta_harness_hash`、`decision_id` / `experience_id`。不可变工程规则优先于可修改 Meta 工作规则。截断的结构化输出不能被接受为有效决策或有效 Harness。

Task rollout、Task 修改、Meta 调用分别有成本记录。API 金额、GPU 小时无法可靠取得时仍为 null。Task 修改耗时可能包含其中的 Meta 补丁调用等待时间，因此这些层级的耗时不能直接相加成总耗时。

## 7. 代码、测试与真实结果

主要变更涉及：

- `sia/task_meta/types.py`、`loop.py`、`storage.py`：状态、决策、经验契约、代际时序及快照。
- `sia/task_meta/execution.py`、`gpqa_target.py`、`harness.py`：隔离资产、全部产物登记、执行指标、机制钩子。
- `sia/task_meta/meta.py`、`prompts.py`、新增 `observations.py`：当前 Meta Harness 注入、不可变规则、覆盖与内容证据。
- `sia/task_meta/updaters.py`：单组件精确目标验证与固定 MODEL 后端。
- `sia/task_meta/entry.py`、`serve_gpu.py` 与 `configs/task-meta-gpu.json`：运行参数、GPU 约束及 checkpoint 实际绑定。
- `scripts/train_task_meta_sft.py`、`test_task_meta_model_integration.py`、`audit_task_meta_run.py` 及对应测试：真实训练、测试专用注入、结果证据检查。

旧 SIA 模式继续保留，新增行为通过 `--evolution-mode task-meta` 显式启用。最终文件列表以交付时工作区差异为准。

**自动化验证：完整仓库测试 237 passed、1 skipped、7 subtests passed；Ruff 检查通过。** 跳过的是 `test_openrouter_prefix_preserves_cache_control_on_the_wire`，因为本环境没有安装可选 OpenHands；当前 Task–Meta 使用 pydantic-ai。另有一条 Starlette/AnyIO 弃用提示，不是测试失败。日志位于 `local_baseline/verification_v2/pytest_full.log` 和 `ruff.log`。

已要求覆盖的工程契约包括：初始空资产、冻结输入、题目/rollout 隔离、无隐式历史继承、分数关联输入、基于 intervention base 的单组件检查、拒绝静默部分执行、新 Meta 版本进入 routing 与补丁、MODEL 后新 checkpoint 请求，以及五执行/四修改/四经验/一次总结且无多余 successor。

### 五轮自主运行：run_4001

本次实际在 A800 上执行了本地 Qwen 推理与真实 LoRA SFT。路由、H 补丁、经验学习和总结均调用现有 OpenRouter Meta 模型；所有最终提交决策的 `decision_source=model`。

| Task | 正确数 / 16 | 有效答案率 | 解析失败 / 截断 / API 失败 | 实际 checkpoint | 学习后 Meta 版本 | 下一次自主动作 |
| --- | --- | --- | --- | --- | --- | --- |
| T0 | 0 / 16 | 16 / 16（100%） | 0 / 1 / 0 | B | M0 | HARNESS：format_question |
| T1 | 0 / 16 | 13 / 16（81.25%） | 3 / 3 / 0 | B | M1 | HARNESS：harness_config、format_question |
| T2 | 3 / 16 | 16 / 16（100%） | 0 / 1 / 0 | B | M2 | MODEL：第一次真实 SFT |
| T3 | 12 / 16 | 16 / 16（100%） | 0 / 0 / 0 | P1 | M3 | MODEL：从 P1 继续真实 SFT |
| T4 | 13 / 16 | 16 / 16（100%） | 0 / 0 / 0 | P2 | M4，最终总结后 M5 | 无 |

checkpoint 引用：

- B：`/root/data/zh/huggingface/Qwen2.5-3B-Instruct`。
- P1：`/root/data/RSI_iclr2027/rsiH/runs/run_4001/gen_2/model_update/checkpoint`。
- P2：`/root/data/RSI_iclr2027/rsiH/runs/run_4001/gen_3/model_update/checkpoint`。

上表“解析失败 / 截断”统计的是每条 rollout 最后一次调用。完整调用统计如下，包含后来经重试恢复的事件：

| Task | 轨迹数 | 实际模型调用数 | 全部解析失败事件 | 全部输出截断事件 | API 错误事件 |
| --- | --- | --- | --- | --- | --- |
| T0 | 16 | 16 | 0 | 1 | 0 |
| T1 | 16 | 16 | 3 | 3 | 0 |
| T2 | 16 | 23 | 7 | 8 | 0 |
| T3 | 16 | 23 | 7 | 6 | 0 |
| T4 | 16 | 22 | 6 | 5 | 0 |

因此 T3/T4 最终没有解析失败，不代表中间没有发生失败；重试属于 H 中的任务内执行机制，仍计在同一条任务轨迹中。分数是 16 条轨迹的最终正确比例，不是 pass@8。

资产清单数量依次为：T0 输入/输出 0/11、T1 为 11/11、T2 为 11/0、T3 为 0/0、T4 为 0/0。Meta 在 T1 后的 H 修改中将 `use_artifacts` 和 `generate_artifacts` 设为 false，同时将 `max_attempts` 由 1 改为 2。因此 T2 仍有 11 份冻结输入资产，但执行机制未使用它们且未生成新资产，之后 A 为空。这是已声明的机制修改及当前轮资产生命周期结果；没有覆盖或删除旧追溯快照。

四条经验完整保存在 `meta/experiences.jsonl`，同时分别保存于其结果所在的 `gen_1` 至 `gen_4/improvement_experience.json`：

| 经验 | 实际动作 | 观测分差 | 接收该结果的 Meta 更新 |
| --- | --- | --- | --- |
| experience_0_1 | HARNESS | 0 | v0 → v1 |
| experience_1_2 | HARNESS | +0.1875 | v1 → v2 |
| experience_2_3 | MODEL | +0.5625 | v2 → v3 |
| experience_3_4 | MODEL | +0.0625 | v3 → v4 |

最后一次总结 v4 → v5 成功，`final_consolidation_status=completed`。只读审计结果 `runs/run_4001/evidence_audit.json` 为 `evidence_consistent`：状态和经验引用一致、实际请求与训练 checkpoint 一致、Meta 调用的 Harness hash 一致、没有额外未执行的 successor。

Task Harness 的 SHA-256 前 12 位：T0 为 `24e8240ca004`，T1 为 `4537d190cbf6`，T2/T3/T4 均为 `e55fdcff7a69`。后两次 MODEL 更新没有改变 H。

Meta Harness 文件位于 `runs/run_4001/meta/harness_vN.md`。v0 至 v5 的 hash 前 12 位依次为 `07e1466425d4`、`9829658302e3`、`221008b113f4`、`067169493b86`、`5a89c0ecb1ac`、`cc12a7cde931`。完整 hash 在调用记录和证据文件中。T0/T1/T2/T3/T4 执行时已有的 Meta 版本分别为 v0/v0/v1/v2/v3；后续路由分别使用 v0/v1/v2/v3。

共记录 15 次 Meta 调用：6 次 routing、4 次 H 补丁生成、4 次经验学习、1 次最终总结。多出的 routing/补丁来自两次未提交的工程约束重试：T0 后第一份 H 补丁修改了未声明的三个 hook；T1 后第一份 H 补丁存在语法错误。它们都被拒绝，`actual_task_modification_executed=false`；修正后才各提交一次 H 修改，不能把 6 次 routing 算成 6 次 Task 更新。

T0 已报告 11 份当前输出资产，均有来源记录。路由观察在 72,000 字符轨迹预算下提供 12/16 条完整记录，明确列出 4 条遗漏，字段截断数为 0；全量统计仍覆盖 16 条。不能把这次观察称为“Meta 读取了全部 16 条完整轨迹”。由于 T0 没有正奖励样本，其当时可用动作中 MODEL/SFT 为不可执行。

路由实际提供/遗漏/截断字段数分别为：T0 `12/4/0`、T1 `9/7/9`、T2 `8/8/6`、T3 `7/9/0`。每轮全量结果统计始终覆盖 16 条。T4 没有下一次 routing；其前后轨迹证据通过末轮 Meta 学习调用提供，覆盖记录保存在该调用的 Prompt 载荷中。这里采用有限字符预算后的分层观察，不宣称 Meta 读取了每轮全部完整轨迹。

有效答案率 100% 与一次输出截断可以同时出现：被截断的响应可能已经输出可解析的 A/B/C/D 答案。T0 的 0/16 表示这 16 次可解析答案均未得正奖励；不能由此直接认定失败一定来自某种推理瓶颈。

### MODEL 全链路

第一次 MODEL 已由真实 Meta 自主选择，并接通训练、下一代推理、评分、经验、真实 Meta 学习及新版本后续路由：`generation_2_decision_0 → gen_2/model_update → T3 → experience_2_3 → Meta v3 → generation_3_decision_0`。决策来源均为 `model`，没有测试注入。第二次 MODEL 同样由 Meta 自主选择，接续 P1 训练得到 P2，T4 评估后形成 `experience_3_4`，真实 Meta 学习为 v4，最后总结为 v5。第二次 MODEL 后没有再路由，是因为已达到五轮预算；不额外创建 T5。

| 训练记录 | 第一次 MODEL（T2 → T3） | 第二次 MODEL（T3 → T4） |
| --- | --- | --- |
| 当前训练起点 | B | P1，确实接续前次权重 |
| 输出 checkpoint | P1 | P2 |
| 可用正奖励轨迹 | 3 | 12 |
| 实际采样次数 / 不同轨迹数 | 8 / 3 | 8 / 8 |
| 数据集监督 token / 实际采样监督 token | 817 / 2,138 | 2,602 / 1,671 |
| optimizer steps | 8 | 8 |
| training loss | 0.36965244 | 0.18897519 |
| 发生变化的 LoRA 张量 | 144 | 144 |
| 合并权重探针中变化的元素 | 3,262,385 | 3,013,239 |

第一轮重复采样了 3 条可用正轨迹；第二轮只使用了 12 条中的 8 条不同轨迹，不能把它写成遍历了全部数据。两次训练均为单卡 A800 的 LoRA SFT，8 步、batch size=1、学习率 0.001、rank=8、q_proj/v_proj，之后合并为独立完整 checkpoint。第一次峰值 CUDA 分配约 7.61 GiB。

T3 的实际请求与每次模型响应都绑定 P1；T4 实际请求与每次响应均绑定 P2。相应训练请求、训练指标、`checkpoint_verified.json`、每代 `service_binding.json`、逐请求响应绑定与经验引用互相对应，并已通过只读审计。服务模型列表检查不是这项结论的唯一依据。

独立的 `test_task_meta_model_integration.py` 没有运行，因为自主实验已覆盖所要求的 MODEL 全链路；没有必要再注入决策或重复训练。ARTIFACTS updater 在本次自主运行中未被选中，因此本次没有它的真实 Meta 资产补丁调用；其文件增改删、边界和同循环接入由契约测试覆盖，不能声称本次自然运行覆盖了三种动作。

未新增因果成对重跑、三分支比较、完整 benchmark 或 Meta 质量评估。旧 `run_3002` / `model_branch_3002` 的记录与权重保留。修改前代码备份为 `/root/data/RSI_iclr2027/rsiH_revision_backups/pre-mvp-repair-20260906-012748.tgz`。

## 8. 复现入口与证据位置

在有权访问项目数据盘的服务器中，先激活现有环境，并通过环境变量提供凭证：

```bash
cd /root/data/RSI_iclr2027/rsiH
conda activate /root/data/conda/envs/sia
read -r -s -p 'OpenRouter API key: ' OPENROUTER_API_KEY
export OPENROUTER_API_KEY
export LOCAL_QWEN_API_KEY=local
bash scripts/start_task_meta_qwen_gpu.sh
bash scripts/run_task_meta.sh NEW_RUN_ID 5 configs/task-meta-gpu.json
```

将 `NEW_RUN_ID` 替换为未使用的整数；不要覆盖现有 run。项目当前默认以 GPU 0 提供推理、GPU 1 执行被选择的 SFT。服务器四张 A800 均可供使用，但这不等于本次流程自动占满四卡或启用了四卡并行训练。

只有需要额外证明 MODEL 分支接线时，另选 run ID 执行独立测试：

```bash
python scripts/test_task_meta_model_integration.py --run-id NEW_MODEL_TEST_RUN_ID --config configs/task-meta-gpu.json
```

该入口执行四次 Task，最多注入一次可执行的 MODEL 决策，然后恢复真实 Meta；默认自主入口不使用测试 override。运行完成后可执行只读证据审计：

```bash
python scripts/audit_task_meta_run.py runs/run_NEW_RUN_ID --output runs/run_NEW_RUN_ID/evidence_audit.json
```

复核本次已有结果，不重跑模型：

```bash
python scripts/audit_task_meta_run.py runs/run_4001 --output runs/run_4001/evidence_audit.json
python -m pytest -q --tb=short
```

主要证据位于 `runs/run_<ID>/`：

| 文件或目录 | 可核查内容 |
| --- | --- |
| `task_meta_config.json`、`profiles.json`、`implementation/` | 同 run 设置、模型配置、实际源代码和环境快照 |
| `gen_t/evaluated_state.json`、`task_state_after_rollout.json` | 分数对应输入、Meta 干预起点 |
| `gen_t/artifacts_input/`、`artifacts_generated/`、`artifact_provenance.json` | 冻结输入、当前输出、产物来源 |
| `gen_t/rollout_artifact_diff.json`、`intervention_diff.json` | 自然生命周期变化与 Meta 实际干预变化 |
| `gen_t/agent_execution.json`、`results.json`、`service_binding.json` | 真实请求、终局奖励、指标、实际模型绑定 |
| `gen_t/meta_observation.json`、`meta_decision*.json`、`task_update.json` | Meta 观察内容、尝试与最终决策、实际补丁 |
| `gen_t/model_update/` | 当前 checkpoint 训练请求、真实 SFT 指标、新权重和注册验证 |
| `gen_t/improvement_experience.json`、`meta/experiences.jsonl` | Task 修改结果进入 Meta 的经验 |
| `meta/calls/`、`meta/harness_v*.md` | 每次 Meta 调用、版本、hash、学习与最终总结 |
| `final_state.json`、`evidence_audit.json` | 最后已评估输入、最后输出资产、完成状态与证据一致性检查 |

## 9. 本版仍未声称完成的能力

当前是 GPQA 两题上的工程 smoke test。重复同批任务符合本次 MVP 设定，不能作为泛化能力评估。

目前没有 GRPO、MetaEval、候选 Meta 比较、Task/Meta 性能回滚、永久知识库、复杂资产质量筛选，也没有接入完整 Code、Tool Use、SearchQA 等任务。Harness 的完整概念范围大于这五个实现钩子；复杂检索、外部工具与通用工作流能力仍未实现。

结构和接口通过不等于语义效果达成；正奖励不等于资产知识全部正确；训练 loss 下降、权重改变或单轮满分不等于 RSI 有效。此次完成标准是：真实单组件修改进入下一代执行，状态与资产可追溯，结果反馈到下一版 Meta Harness，并且该版本确实参与后续改进。
