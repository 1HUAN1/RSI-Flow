# Source release and external inputs

The shared skill implementation supports Qwen3-4B and Qwen3-8B. Select a profile
with `launcher.py --backbone 4b` or `--backbone 8b`; both invoke `launch_meta.py`.
The Task backbone is selected in `runtime/configs/base_4b.json` or `base_8b.json`.

Before launching, configure the dataset root, source training/validation manifests,
frozen-data destinations, retrieval index, output directories, model checkpoint,
benchmark installations and interpreters, and the Codex executable/catalog paths.
The inherited absolute server paths are examples and must be replaced locally.
The frozen cohort must contain 3,000 training tasks and 744 independent evaluation
tasks. Keep both profiles' protocol and non-model settings identical.

For an existing working server, retain its model weights, datasets, evaluator
installations, retrieval indexes and Python environments. Point the new profiles
to those assets; `/path/to/models/Qwen3-*` is a placeholder, not a downloaded model.
In particular, `configs/validation_744.json` still requires your external
`official_specs` file and benchmark-specific `python_executable` paths. A path
containing an old experiment directory may refer to an external environment;
do not delete that environment when cleaning up source checkouts.
Use a fresh frozen-data destination when migrating a release prepared with the
old directory names. Both new backbones can then share that frozen release.
Keep run outputs under the repository root and use a fresh run directory.
Set `RSIFLOW_CODEX_EXECUTABLE` to the actual Codex binary, or make `codex`
available on `PATH`. `--check` reports paths and does not execute the model,
the API, or the official evaluators; a zero exit code is not a full-run guarantee.

Model weights, task data, indexes, API credentials, and the Codex binary are not
included. Use environment variables for credentials. See the repository README
for the Meta endpoint configuration and dependency setup.

The executable Meta-program variant and historical experiment copies were removed
from this distribution. The skill implementation still improves the Task agent's
harness, model, or artifacts and maintains Meta skills between attempts.

Internal package names, Task-generation tool names and manifest kinds now use
RSIFlow. Start fresh runs with the renamed release. Old snapshots and receipts
are not automatically migrated; do not resume them with these renamed interfaces.
The original code remains recoverable from Git history. Third-party source and
license attributions remain in `../THIRD_PARTY_NOTICES.md` and license files.
