"""Serializable state and contracts for the Task-Meta loop."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field


class TaskUpdateAction(StrEnum):
    HARNESS = "HARNESS"
    MODEL = "MODEL"
    ARTIFACTS = "ARTIFACTS"


class RequestedChange(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str = Field(min_length=1)
    component: TaskUpdateAction
    operation: str = Field(min_length=1)
    target: str = Field(min_length=1)
    instruction: str = Field(min_length=1)
    harness_part: Literal["input", "control", "tools", "memory", "submission"] | None = None


from sia.task_meta.meta_harness.five_stage import Expectations, FiveStageReview


class MetaDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")
    action: TaskUpdateAction
    diagnosis: str = Field(min_length=1)
    evidence: list[str]
    rationale: str = Field(min_length=1)
    proposed_change: str = Field(min_length=1)
    expected_effect: str = Field(min_length=1)
    expected_cost: dict | None = None
    target_components: list[TaskUpdateAction] = Field(default_factory=list)
    requested_changes: list[RequestedChange] = Field(default_factory=list)
    decision_id: str = ""
    decision_source: str = "model"
    diagnosis_kind: str = "hypothesis"
    expectations: Expectations | None = None
    used_principle_ids: list[str] = Field(default_factory=list, max_length=128)


class ExpectedMetaDecision(MetaDecision):
    expectations: Expectations


class MetaHarnessUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    harness: str = Field(min_length=1, description=(
        "The complete UTF-8 instructions.md CONTENT to activate for the next Meta/Codex invocation. "
        "This is instruction text, never a version label, filename, identifier, or summary. "
        "When instructions are unchanged, copy the current instructions.md content exactly. "
        "If bundle_files also includes instructions.md, its value must be exactly equal to this field."))
    rationale: str = Field(min_length=1)
    changed_rules: list[str]
    summary: str = ""
    bundle_files: dict[str, str] = Field(default_factory=dict, description=(
        "Optional mapping of declared G filenames to complete replacement UTF-8 contents. "
        "Use harness as the canonical instructions.md content; omit instructions.md here to avoid duplication. "
        "If supplied here, instructions.md must exactly equal harness. Other G files retain their current content when omitted."))
    status: Literal["UPDATED", "NO_CHANGE"] = "UPDATED"
    request_id: str | None = None
    experience_id: str | None = None
    five_stage: FiveStageReview | None = None


class FiveStageMetaUpdate(MetaHarnessUpdate):
    status: Literal["UPDATED"] = "UPDATED"
    five_stage: FiveStageReview


@dataclass
class ArtifactState:
    directory: str | None = None
    manifest: list[dict] = field(default_factory=list)


@dataclass
class TaskAgentState:
    generation: int
    model_ref: str
    harness_path: str
    artifacts: ArtifactState = field(default_factory=ArtifactState)
    checkpoint_path: str | None = None
    checkpoint_manifest: list[dict] = field(default_factory=list)

    @property
    def version(self) -> str:
        return f"T_{self.generation}"


@dataclass
class MetaAgentState:
    model_ref: str
    harness_path: str
    version: int = 0
    bundle_hash: str | None = None
    bundle_path: str | None = None


@dataclass
class EvaluationResult:
    performance: dict
    trajectories: list[dict]
    cost: dict = field(default_factory=dict)
    evaluated_state: TaskAgentState | None = None
    output_artifacts: ArtifactState | None = None
    artifact_provenance: list[dict] = field(default_factory=list)
    rollout_artifact_diff: dict = field(default_factory=dict)


@dataclass
class TaskUpdate:
    action: TaskUpdateAction
    summary: str
    details: dict = field(default_factory=dict)
    cost: dict = field(default_factory=dict)
    requested_changes: list[dict] = field(default_factory=list)
    applied_changes: list[dict] = field(default_factory=list)
    unapplied_changes: list[dict] = field(default_factory=list)
    files: list[dict] = field(default_factory=list)
    checks: list[dict] = field(default_factory=list)
    semantic_status: str = "unverified"


@dataclass
class ImprovementExperience:
    generation: int
    state_before: dict
    state_after: dict
    decision: dict
    modification: dict
    performance_before: dict
    performance_after: dict
    performance_delta: float
    cost_before: dict
    cost_after: dict
    update_cost: dict
    trajectory_before: str
    trajectory_after: str
    experience_id: str = ""
    evaluated_state_before: dict = field(default_factory=dict)
    intervention_base_state: dict = field(default_factory=dict)
    evaluated_state_after: dict = field(default_factory=dict)
    chosen_action: str = ""
    requested_change: list[dict] = field(default_factory=list)
    actual_change: dict = field(default_factory=dict)
    rollout_artifact_diff: dict = field(default_factory=dict)
    intervention_diff: dict = field(default_factory=dict)
    observed_performance_delta: float = 0.0
    versions: dict = field(default_factory=dict)
    candidate_attempts: list[dict] = field(default_factory=list)
    feedback_root: str | None = None
    deployment_status: str | None = None


@dataclass
class MetaObservation:
    generation: int
    task_agent_version: str
    meta_agent_version: int
    current_performance: dict
    performance_history: list
    trajectory_summary: dict
    success_examples: list
    failure_examples: list
    current_model_ref: str
    current_harness_summary: str
    current_artifact_manifest: list
    previous_action: str | None
    previous_modification_summary: str | None
    previous_performance_delta: float | None
    improvement_history: list
    cost_history: list
    evaluated_state: dict = field(default_factory=dict)
    intervention_base_state: dict = field(default_factory=dict)
    input_artifacts: dict = field(default_factory=dict)
    output_artifacts: dict = field(default_factory=dict)
    rollout_artifact_diff: dict = field(default_factory=dict)
    trajectories: list = field(default_factory=list)
    observation_coverage: dict = field(default_factory=dict)
    available_actions: dict = field(default_factory=dict)
    budget: dict = field(default_factory=dict)
    # Trusted operation inputs, never a fixed-policy preview or model-authored data.
    raw_trajectories: list = field(default_factory=list)
    experience_ledger: list = field(default_factory=list)
    current_harness_files: dict = field(default_factory=dict)
    current_harness_identity: dict = field(default_factory=dict)
    current_artifact_files: dict = field(default_factory=dict)


@dataclass
class GenerationContext:
    generation: int
    directory: Path
    observation: MetaObservation
    evaluation: EvaluationResult
    meta_state: MetaAgentState | None = None


class TaskUpdater(Protocol):
    def apply(
        self, task_state: TaskAgentState, decision: MetaDecision, context: GenerationContext
    ) -> tuple[TaskAgentState, TaskUpdate]: ...


class UpdatePending(RuntimeError):
    """A training request exists, but no new executable checkpoint is available."""


class DecisionConstraintError(ValueError):
    """An uncommitted decision violates current execution capabilities."""
