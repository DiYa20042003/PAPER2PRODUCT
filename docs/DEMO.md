# PAPER2PRODUCT demo guide

This is a short live walkthrough of the persisted research workflow. It uses one
epoch and MNIST subsets for a UI smoke demo; those measurements are not the
five-epoch full-dataset acceptance result and must not be presented as a
scientific reproduction.

## Before the demo

Follow the backend and frontend setup in [README.md](../README.md). Start the
API from `backend/` in Terminal 1 and the Next.js app from `frontend/` in
Terminal 2. Confirm `http://localhost:8000/health` returns `{"status":"ok"}`.
The script below assumes both services are already running and the machine has
network access to fetch the reference PDF and MNIST if they are not cached.
No LLM key is required; absent a configured provider key, the app labels
analysis as `rule_based` with lower confidence.

## Five-command demo script

Run these five numbered PowerShell command blocks from the repository root.
The API calls create real local records; no seeded or fabricated records are
used.

1. Ingest the public LeCun paper:

   ```powershell
   $api = "http://localhost:8000"
   $paper = Invoke-RestMethod -Method Post -Uri "$api/papers/ingest" -ContentType "application/json" -Body (@{ source_type = "url"; source_ref = "https://hal.science/hal-03926082/document" } | ConvertTo-Json)
   "paper_id=$($paper.id) title=$($paper.title)"
   ```

2. Analyze it and display the stored claim count and extraction label:

   ```powershell
   $analysis = Invoke-RestMethod -Method Post -Uri "$api/papers/$($paper.id)/analyze" -ContentType "application/json" -Body "{}"
   $claims = (Invoke-RestMethod -Uri "$api/papers/$($paper.id)/claims").claims
   "method=$($analysis.extraction_method) confidence=$($analysis.confidence_level) supported_claims=$(($claims | Where-Object supported).Count)"
   ```

3. Start an actual one-epoch, subset-sized demo run and wait for its persisted
   terminal status:

   ```powershell
   $run = Invoke-RestMethod -Method Post -Uri "$api/experiments" -ContentType "application/json" -Body (@{ paper_id = $paper.id; epochs = 1; max_train_samples = 256; max_test_samples = 128; timeout_seconds = 600 } | ConvertTo-Json)
   do { Start-Sleep -Seconds 2; $run = Invoke-RestMethod -Uri "$api/experiments/$($run.id)" } while ($run.status -in @("QUEUED", "RUNNING"))
   if ($run.status -ne "COMPLETED") { throw "Experiment $($run.id) ended $($run.status): $($run.error)" }
   "experiment_id=$($run.id) status=$($run.status) metrics=$($run.metrics_path)"
   ```

4. Build the evidence-backed plan, compare the observed run, and show the
   actual verdict and score:

   ```powershell
   $plan = Invoke-RestMethod -Method Post -Uri "$api/papers/$($paper.id)/plan" -ContentType "application/json" -Body (@{ experiment_id = $run.id } | ConvertTo-Json)
   $comparison = Invoke-RestMethod -Method Post -Uri "$api/plans/$($plan.plan_id)/compare" -ContentType "application/json" -Body "{}"
   "plan_id=$($plan.plan_id) status=$($plan.status) comparison=$($comparison.status) score=$($comparison.score)"
   ```

5. Retrieve the persisted report and open the dashboard:

   ```powershell
   $report = Invoke-RestMethod -Uri "$api/papers/$($paper.id)/report"
   "report_status=$($report.comparison.status) observed_run=$($report.comparison.experiment_id)"
   Start-Process "http://localhost:3000"
   ```

The dashboard's paper library selects the new paper. Its Analysis, Experiment,
Verification, and Report views show the records and metrics returned by the API.
The first run may take longer than three minutes if the PDF or MNIST archives
need downloading.

## Three-minute talk track

- **0:00–0:30 — Evidence, not invented context.** Show the ingested paper and
  parsed sections. Explain that paper facts carry a section and quote; the
  deterministic fallback is visibly labeled `rule_based` / lower confidence.
- **0:30–1:00 — Claims and plan.** Point to a supported claim and its source
  excerpt, then show the plan's explicit setup differences and “Not identified”
  values for details the paper does not establish.
- **1:00–1:40 — Measured experiment.** Show the run status, sample limits,
  actual log lines, and loss/accuracy charts. State clearly that this
  one-epoch subset run is a UI demonstration, not the five-epoch acceptance
  experiment.
- **1:40–2:20 — Comparison and verification.** Show the paper's quoted test
  error, its conversion to accuracy, the measured difference, score components,
  and the separate observed-evidence and possible-explanation lists.
- **2:20–3:00 — Report and boundaries.** Open the persisted report and call out
  the original-setup difference sentence. Close with the project boundary:
  only the vetted LeNet/MNIST entrypoint runs; generated code is never executed.

## Network or provider fallback

- **LLM API key absent/unavailable:** do not block the demo or paste a key into
  slides. The application falls back to deterministic rules only when no
  provider key is configured and labels the result `rule_based` with lower
  confidence. A configured provider that errors is reported as an error; it is
  not silently relabeled as fallback output.
- **Network unavailable:** the reference PDF and MNIST archives are runtime
  files and are intentionally not committed. Before the presentation, cache
  them using the setup/demo network while available. If downloads still fail,
  use the captured screenshots and the already generated, real acceptance
  report/metrics from this run; explicitly label them as previously captured,
  do not claim a fresh live run, and do not substitute hand-entered numbers.
- **API unavailable:** restart it from `backend/` using the README command and
  recheck `/health`. If it remains down, present the screenshots and existing
  persisted report only as captured evidence, state that the live API is
  unavailable, and do not imply that controls are live.

## Captured screenshots

The files in [`screenshots/`](./screenshots/) are captured from the running
application and its persisted API records. They are visual aids, not a source
of numeric results; use the run's `metrics.json` and report for exact values.
