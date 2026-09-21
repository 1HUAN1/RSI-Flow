"""Public tool access for report generation, without training-time scoring."""
from official_evaluation import OfficialTurn


class ReportEnvironment(OfficialTurn):
    """Keep search/code tools; defer all terminal scoring to official runners."""

    def __init__(self, public_environment):
        self.public_environment = public_environment

    @property
    def tools(self):
        return self.public_environment.tools

    def reset(self, task, rollout_id, seed):
        return self.public_environment.reset(task, rollout_id, seed)

    def step(self, name, args):
        return self.public_environment.step(name, args)

    def close(self):
        return self.public_environment.close()


def serializable_trajectory(result):
    """Convert the host-owned terminal receipt, not scores, to JSON evidence."""
    from dataclasses import asdict

    result = dict(result)
    evaluation = result.pop('_evaluation', None)
    result['evaluation'] = asdict(evaluation) if evaluation is not None else None
    return result
