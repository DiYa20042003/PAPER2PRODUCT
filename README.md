# PAPER2PRODUCT

PAPER2PRODUCT is a local research-reproduction workspace. It parses a research-paper PDF, extracts claims tied to source sections, builds a reproduction plan, runs a vetted MNIST experiment, and compares observed metrics with the paper evidence. It stores workflow state in SQLite and presents persisted records in a Next.js dashboard.

## Reference paper and experiment

The bundled-at-runtime reference used for the verified demonstration is Yann LeCun, Leon Bottou, Yoshua Bengio, and Patrick Haffner, “Gradient-based learning applied to document recognition,” *Proceedings of the IEEE*, 1998. Its downloaded PDF is `backend/data/papers/lecun98.pdf` in this local workspace; the `backend/data/` directory is intentionally ignored because it also contains the MNIST dataset and SQLite database. The paper discusses MNIST. The implemented experiment trains a LeNet-5-style variant on `torchvision.datasets.MNIST`; it is not an exact reproduction of the paper's full architecture, preprocessing, augmentation, optimizer, or training schedule.

## Implemented workflow roles

These are distinct application services in one API workflow, not autonomous agents:

- **Ingestion:** accepts a local PDF, HTTPS PDF URL, or arXiv identifier; PyMuPDF extracts pages and sections.
- **Understanding and claim extraction:** uses a configured structured LLM provider (Gemini by default, also OpenAI or Anthropic) when its key is available. Without a provider key, a deterministic rule-based extractor is used and labeled `rule_based` with `confidence_level: lower`.
- **Reproduction planning:** compares paper quotes with an available experiment configuration and records differences, blockers, and unknowns as “Not identified” rather than inventing details.
- **Experiment execution:** queues and runs only `backend/experiments/lenet_mnist/run.py` through the vetted LeNet/MNIST entrypoint. It stores real PyTorch metrics, logs, config, and model artifacts.
- **Evidence verification and comparison:** checks quotes against stored source-section text, compares completed run metrics to a reference metric found in paper evidence, and persists a scored report with observed evidence separated from possible explanations.

Generated or extracted text is not executed as code.

## Requirements

- Windows PowerShell instructions below assume the repository is at `C:\Projects\PAPER2PRODUCT`.
- Python 3.10 and the existing `backend/.venv` for the verified local setup.
- Node.js and npm compatible with the versions in `frontend/package-lock.json`.
- Network access is needed on first run if PyTorch must download MNIST or if a PDF must be fetched. Existing local PDF, dataset, and database are ignored runtime files, not tracked deliverables.

## Backend setup and API

From the repository root, install backend requirements into the project environment if needed:

```powershell
cd C:\Projects\PAPER2PRODUCT\backend
py -3.10 -m venv .venv
.venv\Scripts\Activate.ps1
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Optional provider configuration goes in `backend/.env` or process environment variables. Start from `backend/.env.example`; keep real keys local and never commit them. With no configured provider key, paper analysis uses the labeled rule-based fallback.

Start the API:

```powershell
cd C:\Projects\PAPER2PRODUCT\backend
.venv\Scripts\python.exe -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Health: <http://localhost:8000/health>. Interactive OpenAPI documentation: <http://localhost:8000/docs>. The live OpenAPI schema is at <http://localhost:8000/openapi.json>.

Main operations include `POST /papers/ingest`, `POST /papers/upload`, `GET /papers`, `POST /papers/{paper_id}/analyze`, `GET /papers/{paper_id}/claims`, `POST /papers/{paper_id}/plan`, `POST /experiments`, `GET /experiments/{experiment_id}/logs`, `POST /plans/{plan_id}/compare`, and `GET /papers/{paper_id}/report`. See `/docs` for request and response schemas.

## Frontend setup and demo

The frontend defaults to `http://localhost:8000`. To use another API origin, set `NEXT_PUBLIC_API_URL` in `frontend/.env.local` before starting Next.js.

```powershell
cd C:\Projects\PAPER2PRODUCT\frontend
npm ci
npm run dev
```

Open <http://localhost:3000>. The dashboard loads persisted paper and experiment records from the API; it does not seed or substitute mock data. Select the paper in the library, inspect the source-backed claims and reproduction-plan differences, review the saved run's charts and process logs, and inspect the evidence comparison and score. PDF upload and URL/arXiv/local-path ingestion controls are also available.

For a five-command walkthrough, talk track, network/provider fallback, and captured UI screenshots, see [docs/DEMO.md](docs/DEMO.md).

### First-run API demo

A clean checkout has no persisted papers or experiments because the SQLite database, downloaded PDFs, and MNIST data are local runtime files. With the backend running, execute this bounded demo in a second PowerShell window from any directory. It fetches the LeCun paper, analyzes it, runs one epoch on MNIST subsets, and creates a plan and comparison from that completed run:

```powershell
$api = "http://localhost:8000"
$paper = Invoke-RestMethod -Method Post -Uri "$api/papers/ingest" -ContentType "application/json" -Body (@{
    source_type = "url"
    source_ref = "https://hal.science/hal-03926082/document"
} | ConvertTo-Json)

Invoke-RestMethod -Method Post -Uri "$api/papers/$($paper.id)/analyze" -ContentType "application/json" -Body "{}" | Out-Null
$experiment = Invoke-RestMethod -Method Post -Uri "$api/experiments" -ContentType "application/json" -Body (@{
    paper_id = $paper.id
    epochs = 1
    batch_size = 64
    learning_rate = 0.001
    device = "auto"
    max_train_samples = 256
    max_test_samples = 128
} | ConvertTo-Json)

do {
    Start-Sleep -Seconds 2
    $experiment = Invoke-RestMethod -Uri "$api/experiments/$($experiment.id)"
} while ($experiment.status -in @("QUEUED", "RUNNING"))
if ($experiment.status -ne "COMPLETED") {
    throw "Experiment ended with status $($experiment.status): $($experiment.error)"
}

$plan = Invoke-RestMethod -Method Post -Uri "$api/papers/$($paper.id)/plan" -ContentType "application/json" -Body (@{
    experiment_id = $experiment.id
} | ConvertTo-Json)
$comparison = Invoke-RestMethod -Method Post -Uri "$api/plans/$($plan.plan_id)/compare" -ContentType "application/json" -Body "{}"
Invoke-RestMethod -Uri "$api/papers/$($paper.id)/report"
```

The first run downloads the MNIST archives even though training and evaluation are capped. This small subset is only a workflow/UI smoke demo; its metrics are not a paper reproduction result. For a full run, use the experiment controls with default full MNIST data and five epochs, or run the CLI command below; that can take many minutes on CPU. Open or refresh the dashboard after the demo to inspect the newly persisted paper, run, plan, logs, metrics, and comparison.

## Tests and checks

Backend tests (run from `backend/`):

```powershell
.venv\Scripts\python.exe -m pytest tests -q
```

The standalone full-data M1 CLI (also run from `backend/`) is:

```powershell
.venv\Scripts\python.exe -m experiments.lenet_mnist.run --epochs 5 --batch-size 64 --lr 0.001 --device auto
```

Frontend checks (run from `frontend/`):

```powershell
npm run lint
npm run typecheck
npm run build
```

The tests cover ingestion, analysis provenance, planning, comparison/scoring, API contracts, and real subprocess experiment orchestration. A CPU MNIST run may take several minutes depending on machine and dataset availability.

## Known limitations and future work

- The current model is a LeNet-5-style variant, not an exact reconstruction of the 1998 paper's model and preprocessing.
- Rule-based extraction has lower confidence and can return noisy candidates; evidence validation verifies source quotation but does not establish semantic correctness.
- Reported comparison scores are evidence summaries, not proof of causal explanations or exact scientific replication.
- The local workflow uses SQLite and a single vetted experiment runner; it is not a general-purpose arbitrary-code execution or distributed-agent platform.
- Future work could improve PDF layout/table extraction, add reviewable claim curation and authentication, support additional constrained experiments, and provide more explicit report export formats.
