# 独立评估接口

评估与进化分开。训练只通过 `runtime/sia/task_meta/round_validation.py`
在轮次提交后调用独立进程 `validate.py`；评估不实例化 Meta、不执行 SFT，
不把验证成绩反馈给组件接受或经验学习。`start_evaluation.sh` 是同一入口的
独立 Bash 封装，不启动训练或 Meta worker。

## 输入与输出

- 输入：冻结的 `round_snapshot.json`、运行时配置、`configs/validation.json`、
  固定验证任务 manifest。快照引用 checkpoint、Harness 和 Artifacts，不复制模型。
- 输出：快照所在验证目录下的生成记录、官方原始评分、逐任务表、汇总表。
- 仅七项评测的固定分母全部完成评分后，才写 `complete.json`。
- 生成记录的 `completed` 仅表示回答生成完成；`evaluation_status=pending_official`
  不是成功/失败标签，分数也不是零。
- BFCL/ACE：`harness_api.py` 生成下一条回答；`official_evaluation.py`
  提供未评分的终止回执；`tool_validation.py` 调用官方生成/评分入口。
- Code/Search：`report_environment.py` 保留公开工具，屏蔽训练环境的终局评分；
  `report_predictions.py` 保存可序列化回答；`reporting.py` 调用官方评分器。
- 待官方评分的单轮回答不触发结果型 Memory 归纳。普通训练评分保持原逻辑。

## 单独调用

在已准备相同环境、数据、服务配置且没有并发训练占用/切换四卡模型时：

```bash
conda activate sia
bash /root/data/RSI_iclr2027/rsiH/RSIFlow_4B/start_evaluation.sh \
  --snapshot /path/to/validation/round_01/round_snapshot.json \
  --pipeline-config /path/to/run/effective_config.json \
  --config /root/data/RSI_iclr2027/rsiH/RSIFlow_4B/configs/validation.json \
  --role independent_validation \
  --manifest /path/to/validation/manifest.json \
  --execute
```

省略 `--execute` 只检查/展示清单预算，不发模型请求。可用 `RSIFLOW_PYTHON`
指定 Python。入口会核对/准备配置中的模型推理服务，因此不要与训练同时手工启动。
ACEBench 的用户模拟器还需要私密环境变量 `ACE_USER_API_KEY` 和
`ACE_USER_BASE_URL`；脚本不会读取、打印或提交密钥。

## 当前固定任务数与检查范围

| Benchmark | 完整任务数 | 最终评分来源 |
| --- | ---: | --- |
| BFCL-v3 | 50 | 固定上游的原生分类评分 |
| ACEBench | 50 | 原生评分；多轮任务保留完整会话 |
| LiveCodeBench | 50 | 固定上游比较逻辑，候选代码在隔离 worker 执行 |
| HumanEval+ | 25 | EvalPlus base 与 plus 均通过 |
| MBPP+ | 25 | EvalPlus base 与 plus 均通过 |
| HotpotQA-dev | 50 | 官方答案 EM/F1；不声称完整 supporting-fact 指标 |
| 2Wiki-dev | 50 | 官方答案 EM/F1；不声称完整 evidence 指标 |

总计 300。这里是固定子集，不是七个 benchmark 全量 leaderboard 成绩。
版本/文件哈希通过 `validation_sources.py` 与 `OfficialEvaluatorSpec` 核对。
离线测试与源文件检查不能替代七项真实模型评测。

## 2026-09-21 修复与现有断点

1. 修复 `OfficialTurn` 缺少 `evaluate()`，生成成功后被误报 RPC 失败。
2. Search 公共题面没有标准答案，不能调用训练 `SearchQAAdapter.evaluate()`；
   Code 也不应把公开题面的不完整测试当终局验证。使用未评分环境包装。
3. 将 Harness 返回的 `AdapterResult` 转成 JSON 回执，修复 Code/Search
   保存轨迹时的 dataclass 序列化缺口。
4. 本地错误回执保留具体异常与底层 Harness 失败原因。

旧运行 `rsiflow_4b_ds41_180_fresh_v1_r3` 在 BFCL 生成阶段结束后报错退出，
后六项未执行。本次修改不覆盖原始错误回执，不重训、不改已提交的 Meta1 经验。
错误请求标记 `requires_audit`，不会自动盲目重发；恢复旧断点前还需归档失败
验证尝试、记录实现修订并核对恢复绑定。不要把上述模板直接当作旧失败目录
已可无条件重跑的命令，也不要删除错误回执来绕过校验。
