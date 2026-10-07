"""Validated API request and response types."""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, FiniteFloat


class IngestRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_type: Literal["local", "url", "arxiv"]
    source_ref: str = Field(min_length=1, max_length=4096)


class PlanRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    experiment_id: str | None = None


class ExperimentRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    paper_id: int | None = Field(default=None, gt=0)
    plan_id: int | None = Field(default=None, gt=0)
    epochs: int = Field(default=5, gt=0, le=1000)
    batch_size: int = Field(default=64, gt=0, le=8192)
    learning_rate: FiniteFloat = Field(default=0.001, gt=0)
    device: Literal["auto", "cpu", "cuda"] = "auto"
    max_train_samples: int | None = Field(default=None, gt=0)
    max_test_samples: int | None = Field(default=None, gt=0)
    timeout_seconds: FiniteFloat = Field(default=3600, gt=0, le=86400)


class ComparisonRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    reproduced_max_error_rate_difference: FiniteFloat = Field(default=0.005, ge=0, le=1)
    partially_reproduced_max_error_rate_difference: FiniteFloat = Field(
        default=0.02, gt=0, le=1
    )
    metric_similarity_weight: FiniteFloat = Field(default=0.60, ge=0, le=1)
    setup_similarity_weight: FiniteFloat = Field(default=0.25, ge=0, le=1)
    evidence_quality_weight: FiniteFloat = Field(default=0.15, ge=0, le=1)


class PaperResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: int
    title: str
    source: str
    source_ref: str | None
    abstract: str | None
    status: str
    section_count: int


class PaperListResponse(BaseModel):
    papers: list[PaperResponse]


class ExperimentResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    paper_id: int | None
    plan_id: int | None
    status: str
    started_at: str | None
    finished_at: str | None
    exit_code: int | None
    error: str | None
    config_path: str | None
    metrics_path: str | None
    artifact_directory: str | None


class ExperimentListResponse(BaseModel):
    experiments: list[ExperimentResponse]


class LogItem(BaseModel):
    id: int
    stream: Literal["stdout", "stderr"]
    message: str
    created_at: str


class LogsResponse(BaseModel):
    experiment_id: str
    logs: list[LogItem]
    next_after_id: int | None


class PipelineStatusResponse(BaseModel):
    papers: int
    experiments_by_status: dict[str, int]
    latest_paper_id: int | None
    latest_experiment_id: str | None
    latest_plan_id: int | None
    latest_comparison_id: int | None
    latest_pipeline_status: str


class ApiErrorResponse(BaseModel):
    detail: str


JsonObject = dict[str, Any]
