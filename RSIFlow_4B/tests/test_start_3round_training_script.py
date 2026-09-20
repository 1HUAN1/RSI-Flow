"""Isolated tests for the one-click three-round shell launcher.

The real Meta worker, model services, network, and training entrypoint are never used.
"""
import json
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "start_3round_training.sh"
TOKEN = "a" * 64
OPENROUTER_KEY = "fixture-openrouter-secret"
AUTODL_KEY = "fixture-autodl-secret"


class StartThreeRoundTrainingScript(unittest.TestCase):
    def make_fixture(self, directory, *, token=TOKEN, token_mode=0o600,
                     token_hardlink=False):
        project = Path(directory) / "project"
        project.mkdir()
        output_root = Path(directory) / "Rollout_logs"
        script = project / SCRIPT.name
        shutil.copy2(SCRIPT, script)

        config_dir = project / "configs"
        config_dir.mkdir()
        config = config_dir / "train.json"
        config.write_text(json.dumps({
            "rounds": 3,
            "pause_after_round": None,
            "training_tasks_per_pass": 360,
            "candidate_policy": "single_candidate_strict_positive_gain",
            "output_root": str(output_root),
        }))

        key_file = project / "API_key.md"
        key_file.write_text(
            "openrouter: " + OPENROUTER_KEY + "\n"
            "autodl.art: " + AUTODL_KEY + "\n"
        )
        key_file.chmod(0o600)

        worker_call = project / "worker_call.txt"
        token_file = project / "worker_token"
        worker = project / "start_meta_worker.sh"
        worker_lines = [
            "#!/usr/bin/env bash",
            "set -Eeuo pipefail",
            '[[ "$#" -eq 1 && "$1" == "--ensure" ]]',
            '[[ -z "${RSI_REMOTE_WORKER_TOKEN:-}" ]]',
            '[[ -z "${AUTODL_API_KEY:-}" ]]',
            '[[ -z "${ACE_USER_API_KEY:-}" ]]',
            'printf "%s\\n" "$*" > "$TEST_WORKER_CALL_FILE"',
            "printf '%s\\n' " + shlex.quote(token)
            + ' > "$RSI_REMOTE_WORKER_TOKEN_FILE"',
            "chmod " + format(token_mode, "o")
            + ' "$RSI_REMOTE_WORKER_TOKEN_FILE"',
        ]
        if token_hardlink:
            worker_lines.append(
                'ln "$RSI_REMOTE_WORKER_TOKEN_FILE" '
                '"$RSI_REMOTE_WORKER_TOKEN_FILE.hardlink"'
            )
        worker.write_text("\n".join(worker_lines) + "\n")
        worker.chmod(0o700)

        launch_record = project / "launch_record.json"
        (project / "launch.py").write_text(
            "import json, os, sys\n"
            "from pathlib import Path\n"
            "Path(os.environ['TEST_LAUNCH_RECORD']).write_text(json.dumps({\n"
            "  'argv': sys.argv[1:],\n"
            "  'token': os.environ['RSI_REMOTE_WORKER_TOKEN'],\n"
            "  'token_file': os.environ['RSI_REMOTE_WORKER_TOKEN_FILE'],\n"
            "  'autodl': os.environ['AUTODL_API_KEY'],\n"
            "  'openrouter': os.environ['ACE_USER_API_KEY'],\n"
            "  'output_root': os.environ['RSIFLOW_OUTPUT_ROOT'],\n"
            "}))\n"
        )
        return {
            "project": project,
            "output_root": output_root,
            "script": script,
            "config": config,
            "key_file": key_file,
            "worker_call": worker_call,
            "token_file": token_file,
            "launch_record": launch_record,
        }

    @staticmethod
    def run_fixture(fixture):
        environment = os.environ.copy()
        environment.update({
            "RSIFLOW_CONFIG": str(fixture["config"]),
            "RSIFLOW_API_KEY_FILE": str(fixture["key_file"]),
            "RSIFLOW_PYTHON": sys.executable,
            "RSI_REMOTE_WORKER_TOKEN_FILE": str(fixture["token_file"]),
            "RSI_REMOTE_WORKER_TOKEN": "stale-worker-token",
            "AUTODL_API_KEY": "stale-autodl-key",
            "ACE_USER_API_KEY": "stale-openrouter-key",
            "TEST_WORKER_CALL_FILE": str(fixture["worker_call"]),
            "TEST_LAUNCH_RECORD": str(fixture["launch_record"]),
        })
        return subprocess.run(
            [str(fixture["script"])],
            cwd=fixture["project"],
            env=environment,
            text=True,
            capture_output=True,
            timeout=20,
            check=False,
        )

    def test_ensure_precedes_secret_loading_and_launch(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = self.make_fixture(directory)
            result = self.run_fixture(fixture)

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(fixture["worker_call"].read_text(), "--ensure\n")
            record = json.loads(fixture["launch_record"].read_text())
            self.assertEqual(
                record["argv"],
                ["--config", str(fixture["config"]), "--execute"],
            )
            self.assertEqual(record["token"], TOKEN)
            self.assertEqual(record["token_file"], str(fixture["token_file"]))
            self.assertEqual(record["autodl"], AUTODL_KEY)
            self.assertEqual(record["openrouter"], OPENROUTER_KEY)
            self.assertEqual(record["output_root"], str(fixture["output_root"]))
            self.assertTrue(fixture["output_root"].is_dir())
            self.assertEqual(fixture["output_root"].stat().st_mode & 0o777, 0o700)
            self.assertIn(str(fixture["output_root"]), result.stdout)
            output = result.stdout + result.stderr
            for secret in (
                TOKEN, OPENROUTER_KEY, AUTODL_KEY, "stale-worker-token",
                "stale-autodl-key", "stale-openrouter-key",
            ):
                self.assertNotIn(secret, output)

    def test_rejects_symlink_api_key_before_worker(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = self.make_fixture(directory)
            target = fixture["project"] / "real_api_key"
            fixture["key_file"].replace(target)
            fixture["key_file"].symlink_to(target)

            result = self.run_fixture(fixture)

            self.assertEqual(result.returncode, 2)
            self.assertFalse(fixture["worker_call"].exists())
            self.assertFalse(fixture["launch_record"].exists())

    def test_rejects_invalid_or_insecure_worker_token(self):
        cases = [
            {"token": "g" * 64},
            {"token_mode": 0o644},
            {"token_hardlink": True},
        ]
        for options in cases:
            with self.subTest(options=options), tempfile.TemporaryDirectory() as directory:
                fixture = self.make_fixture(directory, **options)
                result = self.run_fixture(fixture)

                self.assertEqual(result.returncode, 2)
                self.assertEqual(fixture["worker_call"].read_text(), "--ensure\n")
                self.assertFalse(fixture["launch_record"].exists())

    def test_default_token_file_is_private_state_path(self):
        self.assertIn(
            "${RSI_REMOTE_WORKER_TOKEN_FILE:-/root/.config/RSIFlow_4B/meta_worker_token}",
            SCRIPT.read_text(),
        )


if __name__ == "__main__":
    unittest.main()
