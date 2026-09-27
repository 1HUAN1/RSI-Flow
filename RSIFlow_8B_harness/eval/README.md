# 独立报告评估

所有输出放在数据盘的独立评估目录；入口默认准备计划，只有 --execute 才请求 API／GPU。
执行沿用父工程的四卡 Task 模型服务、成功轨迹 SFT 和官方 benchmark 评分器，
不新写一套评分规则，也不向生产 skill 库写评测结果。ACE 沿用配置跳过。

## 1. Task 指定版本／交叉组合

```bash
python eval/task_version.py --state /path/to/snapshot/task/task_state_snapshot.json \
  --output /root/data/RSI_iclr2027/rsiH/Rollout_logs/eval/task_combo --execute
```

可加 --versions-root /path/to/run/versions --model-id model_001 --harness-id harness_002
--artifacts-id artifacts_001。未指定的组件继承模板。不同实验的编号不能直接混用；
需要对应版本库和仍存在的 checkpoint 权重。生成 Task 状态只是评估用，不激活。

## 2. 原／新 Meta 配对干预

```bash
python eval/meta_pair.py --task-state /path/to/parent_state.json \
  --parent-rollout /path/to/round_1/parent \
  --old-package /path/to/meta_harness/G000 --new-package /path/to/meta_harness/G001 \
  --skills /path/to/skills_snapshot.jsonl --context /path/to/context_snapshot.json \
  --round-number 1 --output /root/data/RSI_iclr2027/rsiH/Rollout_logs/eval/meta_pair
```

先检查 pair_plan.json；确认资源空闲后用相同参数加 --execute。
分别启动常驻 Codex，顺序执行两臂；不并发抢四卡。两臂使用相同父代 Task、
冻结原轨迹引用、skill 与交接副本；各自的 Python 包提供路由／修改方法。
组件自由选择，一臂只测首个候选；记录完整配对分数及 new_minus_old_task_gain。
候选无法执行是 unmeasured、delta=null，不算零分或自动重选。
重复命令复用冻结计划并恢复线程／作业，不覆盖已保存输入；新比较要用新 output。

这是报告实验，不运行生产 finish 的五轮验收、不部署、不总结进入生产经验库。
若想对比 skill 差异而非仅程序差异，应分别准备匹配的受控实验；
这个入口故意保持相同 skill/上下文，隔离程序包差异。

## 3. 多轮闭环汇总

```bash
python eval/loop_summary.py --fixed-run /path/to/fixed_run \
  --evolving-run /path/to/evolving_run \
  --output /root/data/RSI_iclr2027/rsiH/Rollout_logs/eval/loop_comparison.json
```

保留每轮所有尝试、selection、decision、Meta 修订记录和实际 task_metrics.json。
不存在的指标保持缺失，不按零分补齐；只有相同基线、任务批次和评测集合才可作能力对照。
不会把训练增益或某一轮验证下降直接解释成 Meta 方法优劣或确定的因果关系。
