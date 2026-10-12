# Meta 条件经验与反例机制

经验仍以每个实验的 `meta/skills.jsonl` 为事实源，原始轨迹仍保存在 Rollout_logs。
`skills_index.json` 是可重建目录，不包含轨迹副本。经验版本快照保存 JSONL，恢复后
可以重建相同的有效规则。组件选择、候选内容和接受决定继续由 Meta 做出。

## 数据与更新

- `case`：一次真实干预。沿用 `skill.HARNESS.<id>` / `skill.MODEL.<id>` / `skill.ARTIFACTS.<id>`，
  保存 context、diagnosis、mechanism、事前 prediction、measured_outcome、used_skill_ids、
  skill_use_path、round/attempt、适用边界、未知项与证据引用。
- `rule`：有条件的经验。沿用 `principle.<id>`，包含 when、recommend、avoid_when、
  expected_signal、support、counterevidence、status、confidence。没有新原则时不强制编造。
- `incident`：已定位的运行接口或评分故障。记录异常签名、归属、是否已修复及证据。
  零模型调用本身不能证明环境故障；所有原任务仍在完整配对分母中。
- `rule_update`：追加修订事件。target 指向旧原则，operation 可为 weaken、strengthen、
  contradict、supersede；保存 reason、evidence_refs、support/counterevidence 和新 status。
  不覆写原行。案例事实不能通过规则修订改写。
- `procedure`：可复用的具体改法，保存 when、steps、preserve、证据及版本关系。
- `skill_maintenance`：补证、修订、归并和停用事件。更新有效调用视图，不删除历史。

状态建议使用 hypothesis / single_paired_support / repeated_support / contradicted / superseded。
可信度由 Meta 根据证据声明，工具不会因为引用次数增加就自动升级。
旧 kind=skill/principle 兼容读取；未声明的历史原则默认是 hypothesis，confidence=legacy_observation。
记录中的事实与作者当时的解释都保留，不把历史解释自动变成已证实结论。

同一 ID 的完全相同追加请求会返回 reused_ids，避免恢复后重复入库；同 ID 不同内容返回
警告并保留原条目，Meta 可另写新 ID 或 rule_update。坏历史行及孤立修订会报告警告，
其余经验仍可检索。单次或总库均无 8 条上限。

## 技能维护操作

通过 `maintain_skills(path=skills_path, operations=[...])` 调用。语义是否相同、是否需要
新建或归并由 Meta 判断；工具不通过相似度阈值替 Meta 决定，也不自动归并近似技能。

- **新增 `add`**：现有技能未覆盖的具体改法。record 建立 procedure，关联来源和初始
  适用范围；建议填写 component、when、steps、preserve、evidence_refs。
- **补证 `supplement`**：相同条件与改法获得新的响应。target 指向已有技能，追加
  support/counterevidence/evidence_refs；不复制近似技能，不更改步骤或版本，不自动升级可信度。
- **修订 `revise`**：条件、步骤或保持要求变化。target 指向旧版，record 提供新版字段，
  自动记录 previous_version 与递增 version；旧版不再默认检索，但原内容和成绩仍可读取。
  旧版支持与反例放在 prior_version_support/prior_version_counterevidence，不能当新版的实测支持。
- **归并 `merge`**：targets 指向重复技能，record 提供统一版本，或 canonical_id 指向
  已有当前版本。支持、反例和来源合并去重；原版本保留并指向统一版本。不自动扩大适用范围。
- **停用 `retire`**：target 指向过时技能，可注明 replaced_by。仅从默认调用视图移除，
  不删除事实或轨迹。缺目标返回警告，其他有效操作继续执行，不导致整场实验退出。

每项操作可携带 reason 和 evidence_refs。ID 未提供时工具生成；同一请求重试不会重复
追加。一次请求可以包含多项操作，前一项落盘后后一项可直接引用它。

```json
{
  "operation": "maintain_skills",
  "path": "/path/to/run/meta/skills.jsonl",
  "operations": [
    {
      "action": "add",
      "record": {
        "id": "skill.HARNESS.memory.v1",
        "component": "HARNESS",
        "when": ["Memory把没有观察支持的推测写成事实"],
        "steps": ["检查事实来源", "修改事实状态的保存和读取接口"],
        "preserve": ["原本正确的短答提交形式"]
      },
      "reason": "已有技能未覆盖该接口改法",
      "evidence_refs": ["/path/to/attempt/task_differences/summary.json"]
    },
    {
      "action": "supplement",
      "target": "skill.HARNESS.memory.v1",
      "support": ["skill.HARNESS.case_b2"],
      "counterevidence": ["skill.HARNESS.case_b3"],
      "reason": "同一改法的新响应"
    }
  ]
}
```

修订例：`{"action":"revise","target":"skill.HARNESS.memory.v1","record":{"id":"skill.HARNESS.memory.v2","when":["新适用条件"],"steps":["更新后的操作"],"preserve":["明确的保持要求"]},"reason":"原技能遗漏必要步骤"}`。

归并例：`{"action":"merge","targets":["skill.HARNESS.duplicate"],"canonical_id":"skill.HARNESS.memory.v2","reason":"条件与改法相同"}`。
停用例：`{"action":"retire","target":"skill.HARNESS.obsolete","replaced_by":"skill.HARNESS.memory.v2","reason":"旧方法已替代"}`。

`retrieve_skills` 默认只提供 active 记录，procedure 位于每组件的 skills 桶。
`include_inactive=true` 可检索历史版本；`read_skill` 按 ID 始终可以读取旧记录。
索引记录 active、version、previous_version、merged_from、replaced_by 和维护来源。
新规则及技能维护并未增加 finish 门槛，仍使用原经验阶段；快照保存 JSONL 即可重建视图。

## 一次尝试怎样调用

1. `prepare_meta_evidence` 继续生成原有统计、48 条分层片段与完整轨迹索引，并增加
   `failure_fingerprint_path`。指纹是观察到的领域、错误类型、零模型调用、官方成功
   任务数和评分版本。成功任务数不等于实际符合 SFT 筛选的训练样本数。
2. `retrieve_skills` 使用指纹，返回每个组件的支持案例、失败/反例、条件原则、未知案例，
   以及外部 incident。默认每类最多 1 条、正文回执预算 24,000 字符，可配置。
   这是按条件词匹配的检索，没有学习到的组件收益模型；相关度不等于预期增益。
3. Meta 查看条件、反例和未知项；需要正文时用 `read_skill` 分页，再沿证据引用读取
   完整原始轨迹。无匹配或空库都是有效结果。检索不要求三个组件必须有经验。
4. 候选构建前写 decision.json，并用 `record_skill_use` 保存 skill_use.json：实际引用的
   ID、选择理由、目标错误、预期改善领域、可能损害的成功行为和检索回执路径。
   工具保存第一次记录；预测缺失或引用未知 ID 会如实记录，不阻断实验。
5. 复测后由 `compare_scores` 提供完整配对成绩，由 `compare_task_differences` 生成
   新成功、新退步、保持成功/失败、未知状态、错误变化、成本差及原始轨迹行号/哈希。
   它仅保留小摘要与引用，不复制 messages/model_calls。官方成功布尔值缺失时记未知，
   不把正数部分奖励直接当作任务成功。
6. Meta 对照被冻结的预测与真实差异，追加 case；按上面的维护操作新增、补证、修订、
   归并或停用具体改法；有支持时追加 rule，原 rule_update 继续兼容。
   同一 before/after 干预的多份总结不是多次独立支持。
   measured_outcome 可保存 new_success/new_regression；prediction_assessment.met 是 Meta
   有依据时作出的判断，缺失时统计保持未知。
7. `snapshot_task_meta` 继续为累计 JSONL 保存版本，下一次检索读取最新账本。
   `skill_usage_report` 汇总唯一干预、第一候选严格正增益比例、已声明的预测命中率、
   净新增成功和被引用经验的对应结果；引用与理解、相关与因果都不能混同。

例：

```json
{"operation":"retrieve_skills","path":"/path/to/run/meta/skills.jsonl","fingerprint_path":"/path/to/attempt/meta_evidence/failure_fingerprint.json","fingerprint":{"dominant_errors":["unverified_memory"]},"per_category":1,"max_chars":24000}
```

返回 retrieval_path 保存本次指纹和有界结果。record_skill_use 可引用它；它提供的是本次
检索快照，后来的索引更新不覆盖它。

```json
{"operation":"append_skills","path":"/path/to/run/meta/skills.jsonl","entries":[{"id":"update.principle.example.b2","kind":"rule_update","target":"principle.example","operation":"weaken","status":"hypothesis","reason":"原结论仅有跨批观察，缺少同任务配对支持","counterevidence":["skill.HARNESS.b2"],"evidence_refs":["/path/to/task_differences/summary.json"]}]}
```

tool_error 回执仍由同一 Meta 会话处理；这些工具没有增加 finish 的新门槛。
组件能力分析应区分平台错误，但不删除失败任务或修改成绩。独立验证只用于报告，
不送入本轮经验学习。领域净成功数不变不能证明没有逐题退步。

## 当前历史经验与启动

`meta_skills/reviewed_seed.jsonl` 是可选、人工复核的启动经验：B1 输出契约退步、B2
Memory/观察接口、B3 SFT 数据覆盖，以及两个外部故障。包含 3 个案例、3 个条件规则、
2 个 incident。它明确保留组合干预、历史评分 bug 和泛化未知等边界，不含独立评测成绩。
原实验的 10 行账本没有改写。

原始同任务配对复核：B1 新成功 1 / 新退步 9；B2 为 6 / 3；B3 为 11 / 6。
可选种子同时保存这两类计数及领域分布，而不是把净涨分等同于没有回归。

默认初始经验为空：configs/train_180_a0_v1.json 的 meta_skill_seed_path=null。显式继承历史：

```bash
bash start_meta.sh --skill-seed meta_skills/reviewed_seed.jsonl --run-dir /root/data/RSI_iclr2027/rsiH/Rollout_logs/runs/NEW_NAME
```

也可在配置中设置 meta_skill_seed_path。种子只初始化新账本；恢复已有账本时不重放。
`skills_initialization.json` 标记 empty/seeded；对比实验必须报告初始 Meta 是否有历史经验。
`--check` 可以检查种子路径，且不调用 API 或写实验结果。

预算设置：meta_skill_context_chars 默认 24000，meta_skill_per_category 默认 1。
预算限制单次取用，不限制经验库的总条数或磁盘大小；大条目正文用 read_skill 分页。
这是字符预算，不是模型 token 预算。索引和经验都不复制整轮上下文。

## 验证范围

离线测试覆盖追加修订、旧记录兼容、成功与反例同时检索、有界正文、坏行恢复、
种子初始化失败后重试、同 ID 幂等追加、三轮继承/快照恢复、逐任务净零退步、
部分奖励与完整成功区分、冻结预测和使用统计。
真实能力增益需要新任务上的受控比较：固定 Task、API、预算和任务批次，比较空库、
原线性库与此条件检索机制。影子分析不改变主实验的单候选接受规则。
