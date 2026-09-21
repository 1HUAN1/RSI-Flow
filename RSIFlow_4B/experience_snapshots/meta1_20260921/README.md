# 已接受的 Meta1 累计经验快照

来源：`rsiflow_4b_ds41_180_fresh_v1_r3`，版本 `v001_043bc63ff8d4`。

- `principles.json`：完整累计库，保留原 5 条，追加 1 条 MODEL 经验和 1 条通用原则，共 7 条；同时保留库内修订记录。
- `memory_append_receipt.json`：原始入库回执，注明第二轮生效。
- `snapshot.json`：来源和文件 SHA-256，便于核对副本完整性。

新增 ID：`skill.MODEL.sft_parent_successes_gen1`、
`principle.positive_sft_signal_coverage_conditional`。

这是已经正式入库的经验，不是失败候选或重写的总结。导出不修改正在使用的
版本，不将它自动设为全新实验的初始经验，也不改变 Meta 加载/追加规则。

经验中的原始轨迹引用仍指向服务器 `Rollout_logs`；原始轨迹、权重、API 密钥
和完整模型会话不上传。克隆源码后可以阅读经验正文，但不意味着那些绝对路径
引用的证据也已随仓库下载。需要复查完整证据时，请使用服务器保留的原始档案。
