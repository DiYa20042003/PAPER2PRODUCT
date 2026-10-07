"""Run the complete five-epoch LeCun reproduction through the live API."""

import argparse
import json
import math
import sys
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from typing import Any


BACKEND_DIRECTORY = Path(__file__).resolve().parents[1]
DEFAULT_PDF = BACKEND_DIRECTORY / "data" / "papers" / "lecun98.pdf"
REQUIRED_REPORT_SENTENCE = (
    "Mid-sem reproduction configuration differs from the original experimental setup."
)


def api_request(
    base_url: str,
    path: str,
    *,
    method: str = "GET",
    body: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload = json.dumps(body).encode("utf-8") if body is not None else None
    headers = {"Content-Type": "application/json"} if payload is not None else {}
    request = Request(f"{base_url}{path}", data=payload, headers=headers, method=method)
    try:
        with urlopen(request, timeout=120) as response:
            return json.loads(response.read().decode("utf-8"))
    except HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"API {method} {path} returned HTTP {error.code}: {detail}"
        ) from error
    except URLError as error:
        raise RuntimeError(f"API {method} {path} failed: {error.reason}") from error


def stream_experiment(
    base_url: str,
    experiment: dict[str, Any],
) -> dict[str, Any]:
    experiment_id = experiment["id"]
    cursor = 0
    status = experiment["status"]
    latest = experiment
    print(f"experiment_id={experiment_id} initial_status={status}", flush=True)
    while status in {"QUEUED", "RUNNING"}:
        logs = api_request(
            base_url,
            f"/experiments/{experiment_id}/logs?after_id={cursor}",
        )["logs"]
        for line in logs:
            print(line["message"], flush=True)
            cursor = max(cursor, line["id"])
        latest = api_request(base_url, f"/experiments/{experiment_id}")
        status = latest["status"]
        if status in {"QUEUED", "RUNNING"}:
            time.sleep(2)

    while True:
        final_logs = api_request(
            base_url,
            f"/experiments/{experiment_id}/logs?after_id={cursor}",
        )["logs"]
        if not final_logs:
            break
        for line in final_logs:
            print(line["message"], flush=True)
            cursor = max(cursor, line["id"])

    if status != "COMPLETED":
        detail = latest.get("error") or f"experiment ended with status {status}"
        raise RuntimeError(detail)
    return latest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--api-url",
        default="http://127.0.0.1:8000",
        help="Base URL of the running PAPER2PRODUCT API",
    )
    parser.add_argument(
        "--pdf",
        type=Path,
        default=DEFAULT_PDF,
        help="Path to the downloaded LeCun et al. paper PDF",
    )
    args = parser.parse_args()
    base_url = args.api_url.rstrip("/")
    pdf_path = args.pdf.expanduser().resolve()
    if not pdf_path.is_file():
        raise FileNotFoundError(f"Required reference PDF was not found: {pdf_path}")

    paper = api_request(
        base_url,
        "/papers/ingest",
        method="POST",
        body={"source_type": "local", "source_ref": str(pdf_path)},
    )
    paper_id = paper["id"]
    print(f"ingested paper_id={paper_id} title={paper['title']}", flush=True)

    understanding = api_request(
        base_url, f"/papers/{paper_id}/analyze", method="POST", body={}
    )
    claims = api_request(base_url, f"/papers/{paper_id}/claims")["claims"]
    if not claims or not all(claim["supported"] for claim in claims):
        raise RuntimeError("Paper analysis produced no supported source-backed claims.")
    print(
        f"analysis method={understanding['extraction_method']} "
        f"confidence={understanding['confidence_level']} "
        f"supported_claims={len(claims)}",
        flush=True,
    )

    plan = api_request(
        base_url, f"/papers/{paper_id}/plan", method="POST", body={}
    )
    if plan["status"] != "READY":
        raise RuntimeError(f"Reproduction plan is {plan['status']}: {plan['blockers']}")
    if not plan["differences"]:
        raise RuntimeError("Expected the reproduction plan to record setup differences.")
    print(
        f"plan_id={plan['plan_id']} status={plan['status']} "
        f"setup_differences={len(plan['differences'])}",
        flush=True,
    )

    experiment = api_request(
        base_url,
        "/experiments",
        method="POST",
        body={
            "paper_id": paper_id,
            "plan_id": plan["plan_id"],
            "epochs": 5,
            "batch_size": 64,
            "learning_rate": 0.001,
            "device": "cpu",
            "timeout_seconds": 7200,
        },
    )
    final_experiment = stream_experiment(base_url, experiment)
    metrics_path = Path(final_experiment["metrics_path"])
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    epochs = metrics.get("epochs")
    if (
        metrics.get("status") != "completed"
        or not isinstance(epochs, list)
        or len(epochs) != 5
        or not all(
            isinstance(epoch.get(key), (int, float))
            and math.isfinite(epoch[key])
            and (
                epoch[key] >= 0
                if key != "test_accuracy"
                else 0 <= epoch[key] <= 1
            )
            for epoch in epochs
            for key in ("training_loss", "test_loss", "test_accuracy")
        )
        or metrics.get("training_samples", 0) <= 0
        or metrics.get("test_samples", 0) <= 0
    ):
        raise RuntimeError("The persisted experiment metrics failed validation.")
    print(
        f"experiment_status={final_experiment['status']} "
        f"epochs={len(epochs)} device={metrics['device']}",
        flush=True,
    )

    comparison = api_request(
        base_url, f"/plans/{plan['plan_id']}/compare", method="POST", body={}
    )
    if comparison.get("experiment_id") != final_experiment["id"]:
        raise RuntimeError("Comparison did not use the completed five-epoch experiment.")
    if comparison.get("score") is None or not math.isfinite(comparison["score"]):
        raise RuntimeError("Comparison did not persist a finite evidence score.")
    report = api_request(base_url, f"/papers/{paper_id}/report")
    persisted_comparison = report.get("comparison")
    if (
        not persisted_comparison
        or persisted_comparison.get("comparison_id") != comparison["comparison_id"]
        or persisted_comparison.get("experiment_id") != final_experiment["id"]
    ):
        raise RuntimeError("The paper report does not contain the persisted comparison.")
    if REQUIRED_REPORT_SENTENCE not in persisted_comparison["rationale"]:
        raise RuntimeError("The required setup-difference sentence is missing from the report.")

    print(
        f"comparison_id={comparison['comparison_id']} "
        f"status={comparison['status']} score={comparison['score']}",
        flush=True,
    )
    print("metrics.json:")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    print(
        "report_sentence="
        + REQUIRED_REPORT_SENTENCE
    )
    print(
        "E2E completed: "
        f"paper_id={paper_id} plan_id={plan['plan_id']} "
        f"experiment_id={final_experiment['id']} "
        f"comparison_id={comparison['comparison_id']}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"E2E failed: {error}", file=sys.stderr, flush=True)
        raise SystemExit(1) from error
