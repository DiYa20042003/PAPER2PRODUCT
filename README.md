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

## Local Windows deployment

The supported local launcher starts the API and frontend, waits for API health
and a responding dashboard, then opens the browser. Install Python 3.10, Node.js
and npm first. From the repository root, create the virtual environment and
install dependencies once:

```powershell
cd C:\Projects\PAPER2PRODUCT\backend
py -3.10 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
cd ..\frontend
npm ci
cd ..
if (-not (Test-Path .env)) { Copy-Item .env.example .env }
```

Edit root `.env` for `API_PORT`, `FRONTEND_PORT`, `NEXT_PUBLIC_API_URL`, and
`PAPER2PRODUCT_CORS_ORIGINS`. The frontend API URL is embedded at build time in
production mode, so set it before starting/building. Process environment values
take precedence over `.env`. Optional provider keys can also be set there; with
no configured provider key, analysis uses the labeled rule-based fallback.
Never commit `.env` or real keys.

Start the development servers and open the dashboard:

```powershell
.\scripts\start.ps1
```

The launcher activates `backend/.venv`, starts Uvicorn and Next.js, checks
`/health` and the dashboard for HTTP 200, then opens the browser. To run the
production frontend, it runs `next build` and starts `next start`:

```powershell
.\scripts\start.ps1 -Production
```

Stop processes started by the launcher:

```powershell
.\scripts\stop.ps1
```

API health and OpenAPI are at `http://127.0.0.1:<API_PORT>/health` and
`/docs`; frontend is at `http://localhost:<FRONTEND_PORT>`. For example, with
the default ports these resolve to `http://127.0.0.1:8000/health`,
`http://127.0.0.1:8000/docs`, and `http://localhost:3000`.

The backend also reads provider-only settings from `backend/.env` for direct
manual Uvicorn launches. The supported launcher reads the root `.env` and
passes its values to both services.

### Deployment verification (2026-11-03)

The launcher was exercised on this Windows machine in development and
production mode, with default ports and with custom ports `8101`/`3101`.
Observed readiness included `API health: HTTP 200`,
`Frontend: HTTP 200`, and a configured CORS preflight returning HTTP 200 with
`Access-Control-Allow-Origin: http://localhost:3102`. Production mode ran
`next build`, started the production server, and returned HTTP 200 from both
the API and dashboard.

A clean local clone was prepared with the commands in this README and launched
using `scripts/start.ps1`; the API health endpoint and dashboard both returned
HTTP 200, then `scripts/stop.ps1` stopped both recorded processes. The first
Next.js startup/build attempt can stop at the security advisory notice; retry
after acknowledging it was successful in these checks.

**Security limitation:** the installed Next.js `16.4.0` emitted a known
security-vulnerability warning. Clean-clone `npm ci` reported 18 dependency
vulnerabilities (16 moderate, 1 high, 1 critical). This deployment check did
not upgrade dependencies. Do not expose this development deployment to the
public internet; review and apply the vendor security update before publishing
or hosting it.

## Backup

Create a ZIP containing the committed source, the latest completed run's
`metrics.json`, `config.json`, and `log.txt`, plus the local SQLite database:

```powershell
.\scripts\backup.ps1
```

The archive is written under `local-backups/`, which is ignored by Git. It
does not include `.env`, downloaded MNIST data, or model weights. Store the ZIP
somewhere separate from the laptop for disaster recovery.

The backup command was run successfully against the local database and the
latest completed run. It printed the ZIP member list, including
`local-data/paper2product.sqlite3` and `recorded-run/{metrics.json,config.json,log.txt}`.
The produced archive was
`local-backups/paper2product-backup-20261103-124803-e035340.zip` on the
verification machine; local backups are ignored and are not committed.

The dashboard loads persisted paper and experiment records from the API; it
does not seed or substitute mock data. Select the paper in the library, inspect
the source-backed claims and reproduction-plan differences, review the saved
run's charts and process logs, and inspect the evidence comparison and score.
PDF upload and URL/arXiv/local-path ingestion controls are also available.

For a five-command walkthrough, talk track, network/provider fallback, and captured UI screenshots, see [docs/DEMO.md](docs/DEMO.md).

## Collaborating

Clone the private repository and start the local demo with:

```powershell
git clone https://github.com/DiYa20042003/PAPER2PRODUCT.git
cd PAPER2PRODUCT
.\scripts\start.ps1
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for setup, tests, branch naming, and
pull request expectations. The pull request checklist is in
[.github/pull_request_template.md](.github/pull_request_template.md).

The GitHub Actions workflow runs backend tests and frontend lint, typecheck,
and production build. **CI is untested until it has run on GitHub.** The build
step retries up to three times because the current Next.js security advisory
gate stopped two of three local build attempts; one local attempt passed.

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
