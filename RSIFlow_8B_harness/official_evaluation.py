"""Inference-only environment: official outer runners execute tools and score.

An assistant turn is not a completed benchmark task. The terminal ``evaluate``
hook satisfies the Harness runtime contract without inventing a score or label.
Final BFCL/ACE scores are produced separately by ``tool_validation.py``.
"""


class OfficialTurn:
    def tools(self):
        return []

    def step(self, name, args):
        raise ValueError("Tool execution belongs to the official outer evaluator")

    def evaluate(self, final_answer):
        from sia.task_meta.environments import AdapterResult

        return AdapterResult(
            reward=None,
            metrics={},
            verification={
                "status": "pending_official",
                "success": None,
                "verifier_id": "official_outer_evaluator",
            },
            details={"scoring_owner": "official_outer_evaluator", "unit": "assistant_turn"},
        )
