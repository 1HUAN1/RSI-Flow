# Meta 三文件包：从 t 轮保存到 t+1

## 包接口

初始源码：meta_harness/G000/{workflow.py,planning.py,memory.py}。
启动时复制到新运行的 meta_harness/G000，保留独立固定起点。

| 文件 | 实际执行入口 | 作用 |
|---|---|---|
| workflow.py | prepare(context)、review(context) | 阶段证据指导、反馈整理和元层漏洞调查 |
| planning.py | prepare(context) | 组件比较、候选设计和自身包修订指导 |
| memory.py | retrieve(path,fingerprint,options)、append(path,entries)、maintain(path,operations) | 实际检索、写入、维护外部 JSONL 经验 |

context 含 round、phase、阶段目标、运行目录与证据路径；review 可接收 selection_paths、
decision/tool/skill-use 文件引用。返回可序列化字典，由常驻桥接作为工具结果提供给 Codex。
方法可以改变，但不得改变 Task 接受口径、评分、MODEL 训练数据来源或独立评测的用途。
Python 包函数的返回指导不是强制路由：Meta 仍负责组件选择、候选与接受。

三文件支持包内相对导入，例如 `from .planning import prepare`；在入口函数内延迟导入也可使用。
加载器在真实函数调用期间保留包，调用结束后清理模块。相同候选目录修改后会重新读取源码。
阶段程序的完整输出保存在 phase_context.json；大输出只给路径，Meta 可按需分页读取，不拒绝候选。

每个 major milestone 更新时才注入当前包输出；分页读取和 wait_job 不反复塞全套方法。
初始 Prompt 保留研究协议、主线与工具参数。活跃包输出明确标记当前版本，
不会依赖 AGENTS.md 修改后自动生效，更不重新安装或修改 Codex Rust 底座。

## 本轮反馈后的调用

```json
{
  "operation": "prepare_meta_review",
  "round_number": 1,
  "selection_paths": ["/数据盘/新run/round_1/attempts/attempt_1/selection.json"],
  "evidence_refs": ["/数据盘/新run/round_1/attempts/attempt_1/skill_use.json"]
}
```

结合 Task 的新成功／新退步、错误和成本变化，查看 Meta 当时的判断、读取记录、预测和候选差异。
实际打开文件不能证明理解正确；仅训练增益不能证明泛化或单个编辑的因果贡献。

Meta 自行判断：相关经验已提供但必要行为缺失、同类流程错误重复、
或者正确行为需要程序支持。缺少知识先维护 skill；外部工程故障另记 incident，
不自动把零收益或失败次数转成“必须改 Meta”。

有程序缺口：
1. materialize_meta_harness(destination,round_number) 复制当前完整三文件到候选目录。
2. Meta 修改必要模块，未修改模块原样保留；不得只交付一个文件或编辑当前已选版本。
3. check_meta_harness(candidate_dir,round_number) 在子进程编译、导入并调用真实接口。
4. update_meta_harness(decision=replace,candidate_dir,round_number,reason,gaps,evidence_refs)
   再做可执行性检查，保存完整 G001 包并排队到 t+1。
   无缺口则 decision=keep，保留当前包。
5. 写交接、保存 snapshot_task_meta，然后完成报告用独立验证；下一轮首次加载时采用排队版本。

检查没有 Meta 成绩门槛、身份/编号匹配接受门槛或“必须修改三个文件”的要求。
接口不存在、返回无法通过通信、语法/执行错误才返回 validation_failed；
结果送回同一 Meta，由它修复或保留原包。运行期钩子出错会记录 warning，
回用父版本的对应接口；若父接口也出错，仍返回可诊断工具/阶段错误而非伪造成功。

## 持久文件关系

```text
<run>/meta_harness/
  G000/{workflow.py,planning.py,memory.py}
  G001/{workflow.py,planning.py,memory.py}
  state.json          版本目录、各轮绑定、下一轮待选版本
  calls.jsonl         round / module / hook / requested_version / executed_version / warning
  phase_context.json  最近一次阶段输入和实际函数输出
  reviews/round_1.json 修改判断、原因、证据、检查和下一版引用
  checks/*.json       子进程可执行检查结果
<run>/meta/skills.jsonl             追加账本
<run>/meta/skills_index.json        派生目录，可重建
<run>/meta/context.json             正在维护的交接
<run>/snapshots/<name>/versions.json
  references.meta_program          本轮实际包及选定下一轮包
  references.context_reference     不可变 meta/context.json 副本
  meta                             当版完整 skill 快照引用
```

程序包与 skill 都保存历史，没有每轮复制整套 Codex。模型权重仅引用，
因此原始模型／SFT checkpoint 目录必须保留。不同 Task 组合可通过 compose_task
或 eval/task_version.py 测评，不修改 active_task.json。
新包只验证接线；是否帮助 Meta 改好 Task，由后续 Task 表现和报告用对照评估观察。

默认五轮 B1–B5。续接会同步配置中的目标轮数，并依据既有回执重建账本；
已经完成三轮的同一会话扩展为五轮时从 B4 继续，不重跑 A0 或 B1–B3。

## 实验对照

- 固定 Meta：同一程序种子 + 固定 skill 种子。
- skill-only：固定程序 + 动态维护 skill。
- 完整方法：程序可更新 + 动态维护 skill。

三臂保持 Task 起点、批次、预算、工具与评分一致。必须公平地应用工程修复；
不能把 schema/评分修复记作 Meta 能力收益。首候选表现、后续尝试数与代价应分别报告。
主实验接受策略不因对比而变；meta_pair 单独测试首个候选，避免“多试直到成功”掩盖路由能力。
