# RSIFlow skill evolution

This directory is the shared implementation for Qwen3-4B and Qwen3-8B Task agents.
The algorithm, Task components, Meta controller, skill memory, training protocol,
and evaluation are identical. Only the Task backbone changes.

See the [repository guide](../README.md) for installation and both launch commands.
Use `configs/train_4b.json` or `configs/train_8b.json`; the corresponding runtime
base config selects the checkpoint. Meta updates skill memory and does not rewrite
its executable program in this release.

- [Skill-memory interface](META_SKILLS.md)
- [External inputs and migration](SOURCE_RELEASE.md)
- [Third-party notices](../THIRD_PARTY_NOTICES.md)
