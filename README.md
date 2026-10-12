# RSI-Flow

## Introduction

RSI-Flow is a framework for iterative improvement of LLM agent systems through a persistent Meta agent. The Task agent combines a reasoning model, an executable harness, and task artifacts. The Meta agent examines execution feedback, selects a component to improve, constructs a candidate, and compares it with the parent agent on the same tasks.

The framework connects Task improvement with Meta experience. Each attempt contributes evidence to a skill memory that supports later diagnosis and candidate design. This release contains the skill-based variant: Meta maintains experience in a skill memory, while its executable program stays fixed. The 4B and 8B variants execute the same algorithm and differ only in the Task model backbone, Qwen3-4B or Qwen3-8B.

This repository contains the Task runtimes, Meta controllers, harness-generation templates, skill-memory tools, experiment configurations, and evaluation interfaces. The released workflows cover tool use, code generation, and multi-hop question answering. Model checkpoints, datasets, retrieval indexes, and API credentials are supplied separately.

## Repository Structure

```text
RSI-Flow/
|-- RSIFlow/                       # Shared skill-evolution implementation
|   |-- configs/train_4b.json      # Same protocol, Qwen3-4B backbone
|   |-- configs/train_8b.json      # Same protocol, Qwen3-8B backbone
|   |-- runtime/rsiflow/           # Shared Task engine
|   |-- upstream/RSIFlow_task/     # Adapted Task components and templates
|   |-- seed_harness/              # Initial Task harness bundle
|   |-- launcher.py                # Backbone selection and launch dispatch
|   |-- launch_meta.py             # Persistent Meta controller
|   |-- skill_memory.py            # Skill retrieval and maintenance
|   `-- tests/
|-- RSIFlow_4B/                    # Thin 4B launch entry point
|-- RSIFlow_8B/                    # Thin 8B launch entry point
|-- third_party/                   # Evaluation tools and provenance
|-- THIRD_PARTY_NOTICES.md
`-- LICENSE
```

Both entry points call the same controller, Task runtime, and skill-memory code.
Their effective runtime configurations differ only in `task_checkpoint`. Training
data, random seeds, update rules, LoRA settings, and final evaluation are identical.
Run names differ to keep outputs separate. This release does not include the
executable Meta-program evolution variant or the historical experiment copies.

## Setup

### 1. Environment Setup

Use Linux for the experiment workflows. The runtime requires Python 3.11 or later.

Download and extract the anonymous repository archive, then enter its root directory. For a Git checkout, use the repository URL available to you.

```bash
cd RSI-Flow
conda create -n rsiflow python=3.11 -y
conda activate rsiflow
python -m pip install -U pip setuptools wheel

# Choose the workspace to reproduce.
cd RSIFlow
python -m pip install -e './runtime[dev]'
```

Install the dependencies needed by the selected harness and benchmarks; the bundled [harness requirements](RSIFlow/upstream/RSIFlow_task/requirements.txt) provide an additional reference.

### 2. Model Serving and Training Dependencies

Prepare a CUDA/PyTorch environment compatible with your GPUs, Task inference backend, and LoRA training stack. The runtime's `pipeline` extra declares Transformers, PEFT, and Accelerate dependencies:

```bash
python -m pip install -e './runtime[pipeline]'
```

Obtain the Qwen3-4B or Qwen3-8B checkpoint and set `task_checkpoint` in the matching `runtime/configs/base_4b.json` or `base_8b.json`. Both profiles use the same four-GPU configuration. Adjust GPU indices, service ports, concurrency, and training settings together for your machine.

The persistent Meta controller also requires a compatible Codex executable. See the [Codex provenance record](third_party/codex-provenance.json) and [source-release guide](RSIFlow/SOURCE_RELEASE.md) for the recorded environment and external inputs.

### 3. Configure Environment Variables

Set credentials in the shell rather than in tracked files:

```bash
export RSIFLOW_PYTHON="$(command -v python)"
export RSIFLOW_CODEX_EXECUTABLE="$(command -v codex)"
export AUTODL_API_KEY="<your-meta-api-key>"

# Required when running the ACEBench user simulator.
export ACE_USER_API_KEY="<your-user-simulator-api-key>"
export ACE_USER_BASE_URL="<your-user-simulator-api-base>"
```

The checked-in Meta configuration uses a DeepSeek-V4.1-Flash endpoint through the Codex Responses interface. Configure the endpoint and model in the selected workspace's `configs/codex_host.toml` and model catalog.

Before launching, update the local paths in the experiment JSON files: model checkpoint, Task data, retrieval index, evaluator installations, Python interpreters, and output directories. `RSIFLOW_PYTHON` selects the launcher interpreter; it does not replace interpreter paths stored in JSON configuration.

## Benchmark Configuration

The five-round skill-memory experiment uses [train_4b.json](RSIFlow/configs/train_4b.json) or [train_8b.json](RSIFlow/configs/train_8b.json). Each round contains 600 tasks:

| Domain | Source | Tasks per round |
|---|---|---:|
| Tool use | EnvScaler | 200 |
| Code generation | DeepCoder/TACO: 120 STDIO + 80 function tasks | 200 |
| Multi-hop QA | HotpotQA | 100 |
| Multi-hop QA | 2WikiMultiHopQA | 100 |

The five rounds use 3,000 distinct tasks. Configure `source_training_manifest`, `source_validation_manifest`, `dataset_root`, and retrieval paths before preparing the data. EnvScaler environment definitions may be reused across rounds.

Independent evaluation follows [validation_744.json](RSIFlow/configs/validation_744.json):

| Benchmark | Tasks |
|---|---:|
| BFCL-v3 | 152 |
| ACEBench | 50 |
| HumanEval+ | 164 |
| MBPP+ | 178 |
| HotpotQA-dev | 100 |
| 2Wiki-dev | 100 |
| **Total** | **744** |

For this protocol, independent evaluation runs after the fifth round on the selected model, harness, and artifacts. Its results are excluded from Task training and Meta feedback. Both backbone profiles use this same evaluation protocol.

## Inference

Task inference uses the checkpoint, service ports, and runtime configuration of the selected experiment. Prepare the local model and benchmark assets before starting services through the experiment launcher.

From `RSIFlow`, prepare the fixed task batches and run the launch preflight:

```bash
python prepare_600_744.py --config configs/train_4b.json
python launcher.py --backbone 4b --check
python launcher.py --backbone 8b --check
```

These commands prepare data and check launch configuration without starting a GPU/API experiment. The preflight is a configuration check; benchmark-specific dependencies must also be installed.

## Reproduce the Released Experiments

Run either entry point from the repository root:

### Qwen3-4B

```bash
python RSIFlow_4B/launch.py --check
bash RSIFlow_4B/start_5round_600.sh --wait-for-gpus --run-dir /path/to/outputs/rsiflow-4b
```

### Qwen3-8B

```bash
python RSIFlow_8B/launch.py --check
bash RSIFlow_8B/start_5round_600.sh --wait-for-gpus --run-dir /path/to/outputs/rsiflow-8b
```

The shell entry points prepare the configured task data and invoke the same shared
launcher. Set `RSIFLOW_PYTHON` to the Python interpreter in your environment. Use
`--config /absolute/path/to/train.json` for a customized profile. Monitor either run:

```bash
bash RSIFlow/watch_meta.sh --run-dir /path/to/outputs/rsiflow-4b
```

Use a new output directory for each backbone. Relaunching with the same directory
resumes its existing run. Renamed packages, tool names, and bundle identifiers in
this release are intended for fresh runs; previous run records require migration.
No new GPU experiment or performance comparison is claimed by this source cleanup.

## Task Evolution

### Step 1: Collect Feedback

Run the current Task agent and collect scores, costs, and execution trajectories. The Meta agent reads representative successes and failures together with relevant prior experience.

### Step 2: Select a Component

The Meta agent chooses a harness, model, or artifact update based on the observed failures. The controller exposes operations for executing that choice and records their results.

### Step 3: Construct a Candidate

Harness updates follow module localization, improvement planning, and candidate generation. Model updates use successful trajectories for LoRA SFT. Artifact updates revise the task-side files selected by Meta. Candidate versions remain separate from the active parent.

### Step 4: Validate and Compare

Check candidate executability, then evaluate the parent and candidate on the same round tasks. The Meta agent applies the configured acceptance rule and records the selected Task state. A rejected candidate leaves the parent active and contributes evidence for the next attempt.

### Step 5: Save the Selected State

Save version references for the model, harness, artifacts, and Meta memory. These snapshots support resumption, later evaluation, and controlled component combinations. Model checkpoint files must be retained alongside their references.

## Meta Evolution

### Skill Retrieval and Maintenance

Meta memory stores intervention cases, conditional rules, and diagnosed incidents. Before designing an update, the Meta agent retrieves relevant evidence and records how it uses that evidence. After paired evaluation, it can add support, retain counterexamples, revise a skill, merge related skills, or retire an ineffective one. See [META_SKILLS.md](RSIFlow/META_SKILLS.md).

## Evaluation and Tests

Run the selected workspace's offline tests with its runtime on the Python path:

```bash
PYTHONPATH="$PWD/runtime:$PWD" python -m pytest -q tests
```

The suite covers skill maintenance, candidate construction and acceptance, frozen
task cohorts, evaluation isolation, and equality of the two effective backbone
configurations apart from the checkpoint. Full experiments additionally require
external data, model checkpoints, benchmark services, and GPU/API access.

## Citation

Citation details for RSI-Flow will be added after anonymous review.

