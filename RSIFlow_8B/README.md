# RSIFlow with Qwen3-8B

This directory contains only a launch entry point. All algorithm code lives in
[`../RSIFlow`](../RSIFlow/README.md) and is shared by both backbone sizes.

```bash
python launch.py --check
bash start_5round_600.sh --wait-for-gpus --run-dir /path/to/outputs/rsiflow-8b
```

Set the model path in `../RSIFlow/runtime/configs/base_8b.json` and
external inputs in `../RSIFlow/configs/train_8b.json` before launching.
