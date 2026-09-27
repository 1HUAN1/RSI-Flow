# RSIFlow_8B_harness：Task／Meta 协同进化

本目录是 RSIFlow_8B 的独立源码副本；Task 仍是 Qwen3-4B，Meta 仍是
DeepSeek-V4.1-Flash + 常驻 Codex。本次改动不修改、重启或停止 RSIFlow_8B 的在跑实验。

默认配置：A0 初始独立评估开启，五轮，每轮三个领域各 60 条共 180 条。
本次发布默认八卡各运行一个模型副本、32 个 rollout worker；MODEL 使用成功父代轨迹的一轮八卡 LoRA SFT。仍支持四卡配置；迁移机器时按硬件设置服务端口数量。
ACEBench 跳过。任务、评分和模型/API 配置不因 Meta 自更新而改变。

## 本轮主线

父代新批任务 → 读取全部统计、48 条代表片段、累计 skill 和交接上下文
→ Meta 自选 HARNESS／MODEL／ARTIFACTS → 一个候选 → 同批复测
→ Meta 按完整可比、成功率严格提高决定接受
→ 追加／维护经验 → 判断自身程序性缺口 → 检查并选择下一轮 Meta 包
→ Task／Meta 快照 → 报告用独立评测 → 下一轮。

拒绝后保持父代、记录失败经验，回到同一批任务的组件选择；不直接独立评测或跳轮，
没有固定重选次数上限。每次尝试使用独立的 round_N/attempts/attempt_K。
完整原始证据保留在 Rollout_logs，不复制到 skill；独立评测不参与接受、SFT 或 Meta 学习。

HARNESS 仍按固定的 HarnessForge 三阶段模板执行：
故障定位报告 → 改进方向报告 → 新目录的完整 Harness 包 → 最多三次可执行性检查／有限修复。
只有验证器调用不能代替前三阶段；未改文件可保留，修改不限于 Prompt。
ARTIFACTS 仍是直接改提交／重评分，不在本次变成可复用资产学习。

## Meta 的真实可演化入口

固定：Codex 二进制/API 接线、Task 评分器、研究主线、加载和工具传输机制。
可演化：一个完整包中的 workflow.py、planning.py、memory.py。
初始包在 [meta_harness/G000](meta_harness/G000)，接口和加载详见 [META_HARNESS.md](META_HARNESS.md)。

- Workflow：在阶段边界执行 prepare，反馈后通过 prepare_meta_review 执行 review。
- Planning：同一阶段执行 prepare，返回供 Meta 使用的路由、方案或自检指导，不写死组件。
- Memory：retrieve_skills、append_skills、maintain_skills 实际调用所选版本的 Python 函数。
- 每轮绑定版本；本轮检查通过的新包只排队到下一轮，不热换当前尝试，也不要求 Meta 先涨分。
- 可执行检查失败或运行钩子失败返回诊断；运行钩子可回用父包并记录实际执行版本。
  这不是能力提升证明，也不保证任意候选从不出错。

## 启动、续接和观察

当前默认新运行名为 rsiflow_8b_harness_180_5round_8gpu_20260927，推理服务使用独立端口 8371–8378。
同一运行目录再次启动是续接；另开实验须使用新运行目录。

```bash
cd /root/data/RSI_iclr2027/rsiH/RSIFlow_8B_harness
export RSIFLOW_PYTHON="$(command -v python)"
"$RSIFLOW_PYTHON" launch_meta.py --check
bash start_meta.sh --wait-for-gpus
bash watch_meta.sh --run-dir /root/data1/RSI_iclr2027/rsiH/Rollout_logs/RSIFlow_8B_harness_8gpu/runs/rsiflow_8b_harness_180_5round_8gpu_20260927
```

start_meta.sh 在后台启动；--wait-for-gpus 等待配置中的全部 GPU 连续两次空闲，并等待 wait_for_processes 中指定的本机进程结束，不抢占其他作业。迁移机器时须移除不适用的等待进程配置。
本次开发没有自动执行上述实验启动命令。密钥读取方式保持原样；不要把密钥提交到 Git。
聊天入口仍为 chat_meta.sh，使用同一 --run-dir。

Meta_logs/<run>/ 下保存 meta_session、meta、meta_harness；运行目录中保留兼容链接。
Task 轨迹、checkpoint 和评测输出在 Rollout_logs。阶段账本只是持久事实提醒，
不替 Meta 决策；finish 仍只检查约定轮次的必要产物，不增加元层涨分门槛。

## 版本与对照评估

每次快照保存 Task 的 model／Harness／Artifacts 编号组合，模型权重只保留 checkpoint 引用。
Meta 保存完整小型程序版本、当版 skill 内容和不可变的结构化交接副本，不复制 Codex 或聊天。
程序版本 Gxxx 和经验版本 meta_xxx 是两个不同维度，不强求一轮对应一个新程序版本。

[eval/README.md](eval/README.md) 提供三个独立入口：

1. task_version.py：指定 Task 状态或 model／Harness／Artifacts 组合，复用按配置并行的官方评测。
2. meta_pair.py：固定父代 Task、同一批轨迹、skill 和上下文，比原／新 Meta 包产生的首个干预。
3. loop_summary.py：并列汇总固定 Meta 与可更新 Meta 的多轮尝试和独立指标。

评估默认只准备；有 --execute 才请求模型或使用 GPU。对比结果不回写生产技能库。
固定 Meta 实验可设置 meta_harness_mode=fixed、meta_fixed_skills=true；
skill-only 对照设置 fixed、false，并用相同初始 Task、任务批次与 skill 种子。

## 验证与外部依赖

离线回归不等于完整 GPU/API 闭环已经跑完，更不保证每轮涨分。
```bash
/root/data/conda/envs/sia/bin/python -m pytest tests -q
```

模型权重、冻结数据、检索索引、官方评测器及 Codex 安装仍是外部依赖；
当前 frozen_data_dir 只读复用 RSIFlow_8B 的冻结 900 条任务，不调用其运行中控制器源码。
详细修改记录见 [DEBUG_LOG.md](DEBUG_LOG.md)，经验操作见 [META_SKILLS.md](META_SKILLS.md)。
