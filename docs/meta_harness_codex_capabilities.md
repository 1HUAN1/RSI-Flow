# 固定 Codex 能力与 Meta 后端接入证据

核查日期：2026-09-09。目标固定版本为 `rust-v0.153.4`，commit `3d2ee51ca2d5db578f328aa75e20aa22c0197c9a`。服务器协调者本轮只读检查报告 HEAD 匹配、二进制输出 `codex-cli 0.153.4`。本文件依据从该固定源码取得的本地归档检查，未依据最新在线文档推断能力；没有修改、编译或升级 Codex。

本地原始材料位于 `../references/codex-pinned-inspection/third_party/codex/codex-rs`（相对于本地 work 镜像）；两个归档为 `../codex_inspection.tgz`、`../codex_instructions_events.tgz`。文件指纹见相邻 JSON。目录是独立的只读参考副本，不属于 G 可变文件。

## 原生接口与缓存事实

| 固定版本事实 | 读到的源码 | 对实现的约束 |
|---|---|---|
| 无 resume/fork 的 exec 创建新 thread | `exec/src/lib.rs:917-922` 调用 `start_thread` | 每个未缓存的 G 模型阶段使用独立 `codex exec`，不恢复旧会话 |
| thread/start 接收 model/provider/cwd/ephemeral | `exec/src/lib.rs:1188-1199` | 继续由可信配置绑定身份与工作区，使用 `--ephemeral` |
| exec CLI 支持 strict-config、ephemeral、output-schema | `exec/src/cli.rs:22,36-37,49`；登记的 `codex_exec_help.txt` | 没有添加猜测的 Codex 参数 |
| CODEX_HOME 全局指令通过文件读取获得 | `codex-home/src/instructions/mod.rs:24-59`；第 71-72 行调用 loader | 每阶段使用空的新 CODEX_HOME，仅写入受管配置，避免开发会话全局指令混入 |
| 项目 AGENTS 从项目根到 cwd 发现，读取后记录 source_path | `core/src/agents_md.rs:133-170,185-218` | 本次 workspace 的 AGENTS 是只读挂载，来源 hash 单独登记 |
| 项目标记为 untrusted 会跳过项目 AGENTS；内容还受 project_doc_max_bytes 截断 | `core/src/agents_md.rs:61-65,152-160` | 不能仅因文件存在就声称完整原生加载。完整 G instructions 另行内联到阶段 prompt；不把 native consumption 标为已验证 |
| AGENTS 缓存以环境选择和项目 trust 为键，未比较文件 mtime/hash | `core/src/agents_md_manager.rs:12-30,46-64` | 原会话只改磁盘文件可能仍使用缓存。新进程的新 manager 从空缓存开始 |
| AGENTS 指令片段以 user role 渲染 | `core/src/context/user_instructions.rs:15-24` | 不把这一路径描述为独立不可覆盖的 developer/system 安全边界；安全约束由可信控制器执行 |
| 原生 JSONL 明确区分完成、失败、工具条目 | `exec/src/exec_events.rs:10-35,98-130` | `turn.completed`/失败事件用于收集检查；command_execution/file_change 分别归属 shell/apply_patch，MCP/collab/web 工具在受限后端拒绝 |

关键缓存片段：

```rust
if cache.selections.as_ref() == Some(&selections)
    && cache.active_project_trust_level == active_project_trust_level
{
    return Ok(());
}
```

本项目没有尝试热重载旧 Codex 会话。后端 `prepare` 创建新的调用目录和 CODEX_HOME，`run` 创建新的进程；恢复回执命中时则重放已经完成的结果，不伪称启动了新进程。

## 五类机制的能力分层

| 机制 | 可修改入口与实际效果 | 实现层次 |
|---|---|---|
| 证据与上下文 | evolution.json 的筛选、条件权重、分组、片段规则，结合 context.json 的边界；运行时产生带来源 ID/遗漏记录的派生包 | rsiH 可执行配置；最终指令输入使用 Codex 原生 CLI |
| 诊断与选择 | diagnosis 指令/条件规则、操作 workflow 中的分析与检查顺序；结果仍受单组件/可执行动作契约约束 | 指令层 + rsiH 可执行配置；模型阶段为真实 Codex 接口 |
| 修改编排 | workflow 的依赖读取、目标检查、分析、单候选生成、有界修复；H/A/MODEL 请求必须绑定对应路由的 G 与状态 | rsiH 可执行配置 + 原生 Codex 模型/工具运行；固定 updater 负责最后提交 |
| 经验使用 | experience 的条件、相关性、分组、字段摘要与适用/失效规则；缓存身份包含来源与 G 策略 | rsiH 可执行配置，不改原经验账本 |
| 自更新过程 | self_update 指令/条件规则及 meta_self_update/final_consolidation workflow；新 G 能改变下一次分析阶段和规则 | 指令层 + rsiH 可执行配置；不可递归启动无限学习或跳过 Task 评测 |

本轮没有开放任意 Python/Rust 插件或改变 Codex 内核。JSON 规则只通过固定、有限的原语解释；其条件、组合、顺序和有限修复次数可由 G 版本化修改。这一范围无需新增 Codex 源码补丁。

## 后端的实际调用、预算与恢复

v2 的 routing/harness_patch/artifact_patch/model_request/meta_self_update/final_consolidation 统一进入 `meta_harness.runtime.execute`。每个模型阶段回调现有 prepare/run/collect；原生 API 请求仍由 Codex 生成，可信 transport 只做转发、身份及预算检查，没有新建裸 API Agent。

v1 Bundle 的 `supports_evolution=False`，保留旧完整 prompt 路径；迁移由显式 Bundle 接口完成。v2 的调用方仅提供固定结果目的和可信 envelope，诊断/自更新策略来自当前 G。派生证据进入 `bundle.render(..., preselected=True)`，不会再对整个阶段提示做隐藏字符截断。过大的输入明确拒绝。

`operation_budget.py` 是固定可信层：单个操作的所有内部阶段共用请求数、总输出 token、墙钟时间、累计流量和阶段文件预算。transport 在请求发出前持久化 reservation；阶段之间不会重新获得全额额度。失败且实际调用状态不明时保留 pending，禁止自动重发；已经 collect 提交而阶段回执尚未发布的崩溃可从现有 collected.json 对账恢复。

每个 stage 记录 request_id、父 workflow_operation_id、stage_id、Task/G 身份、input hash、实际指令来源、模型/工具事件与收集结果。只有完成的 routing 回执能授权同 decision_id 的对应 H/A/MODEL 请求，G/hash/action/Task 状态不一致即拒绝。

`tool_events.json` 来自固定 exec JSONL 的工具条目，不把模型说“已经检查”当作工具事实。`bundle_load.json.instruction_sources` 明确 `native_agents_consumption_verified=false`；指令内联 hash 和文件 hash 可复核。在线 runtime_verified 也不能解释为语义质量通过。

## 验证边界

本轮后端定向测试覆盖离线阶段执行、可执行 runtime 接入、同路由版本绑定、可信 provenance 写入、真实项目规则进入后续 self_update、预算累计、无效候选修复、过期版本、中断不重发、收集后崩溃对账、原生事件映射与越权工具拒绝。

测试注入的模型/事件均为 `test_override`，相应 runtime_verified 为 false。修改 G 的行为测试是工程师编写的受控 fixture，不是 Meta 自主产生 G 的实验结果。测试中存在“运行真实项目解释器”与“真实 API”两种不同概念；前者已经离线验证，后者没有执行。

真实 Codex/OpenRouter、原生工具读写和在线 G 自更新仍需隔离环境、受管凭证及显式启用后的 API smoke。当前开发没有调用真实 API、执行 GPU rollout 或训练。既有受限 namespace 不可用时的 fail-closed 条件保持，不用取消隔离绕过。
