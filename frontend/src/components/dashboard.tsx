"use client";

import {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
  type FormEvent,
} from "react";
import Link from "next/link";
import {
  analyzePaper,
  ApiError,
  checkHealth,
  comparePlan,
  createPlan,
  getClaims,
  getExperiment,
  getLogs,
  getPaper,
  getPaperReport,
  getPlan,
  getPipelineStatus,
  getUnderstanding,
  ingestSource,
  listExperiments,
  listPapers,
  startExperiment,
  uploadPaper,
  type Claim,
  type Comparison,
  type Experiment,
  type LogItem,
  type Paper,
  type PaperDetail,
  type Plan,
  type PipelineStatus,
  type Understanding,
} from "@/lib/api";

interface EpochMetrics {
  epoch: number;
  totalEpochs: number;
  trainingLoss: number;
  testLoss: number;
  testAccuracy: number;
}

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : "An unexpected error occurred.";
}

function isTerminal(status: string): boolean {
  return !["QUEUED", "RUNNING"].includes(status.toUpperCase());
}

function parseEpoch(log: LogItem): EpochMetrics | null {
  const epoch = log.message.match(/epoch=(\d+)\/(\d+)/);
  const trainingLoss = log.message.match(/training_loss=([0-9.eE+-]+)/);
  const testLoss = log.message.match(/test_loss=([0-9.eE+-]+)/);
  const testAccuracy = log.message.match(/test_accuracy=([0-9.eE+-]+)/);
  if (!epoch || !trainingLoss || !testLoss || !testAccuracy) return null;

  const values = [
    Number(trainingLoss[1]),
    Number(testLoss[1]),
    Number(testAccuracy[1]),
  ];
  if (!values.every(Number.isFinite)) return null;
  return {
    epoch: Number(epoch[1]),
    totalEpochs: Number(epoch[2]),
    trainingLoss: values[0],
    testLoss: values[1],
    testAccuracy: values[2],
  };
}

function MetricChart({
  title,
  data,
  valueFor,
  color,
  format,
}: {
  title: string;
  data: EpochMetrics[];
  valueFor: (item: EpochMetrics) => number;
  color: string;
  format: (value: number) => string;
}) {
  if (data.length === 0) {
    return (
      <div className="chart-card">
        <div className="chart-title">{title}</div>
        <div className="chart-empty">Epoch data will appear here when the run starts.</div>
      </div>
    );
  }

  const values = data.map(valueFor);
  const minimum = Math.min(...values);
  const maximum = Math.max(...values);
  const spread = maximum - minimum || Math.abs(maximum) * 0.08 || 1;
  const points = values.map((value, index) => {
    const x = data.length === 1 ? 160 : 12 + (index / (data.length - 1)) * 296;
    const y = 82 - ((value - minimum) / spread) * 62;
    return { x, y, value, epoch: data[index].epoch };
  });
  const path = points.map(({ x, y }) => `${x},${y}`).join(" ");
  const latest = points[points.length - 1];

  return (
    <div className="chart-card">
      <div className="chart-heading">
        <div className="chart-title">{title}</div>
        <div className="chart-value">{format(latest.value)}</div>
      </div>
      <svg
        className="metric-chart"
        viewBox="0 0 320 100"
        role="img"
        aria-label={`${title} by epoch`}
      >
        <line x1="8" y1="84" x2="312" y2="84" className="chart-axis" />
        <polyline
          points={path}
          fill="none"
          stroke={color}
          strokeWidth="2.5"
          strokeLinecap="round"
          strokeLinejoin="round"
        />
        {points.map((point) => (
          <circle
            key={point.epoch}
            cx={point.x}
            cy={point.y}
            r="3.5"
            fill={color}
          />
        ))}
      </svg>
      <div className="chart-foot">
        <span>Epoch 1</span>
        <span>Epoch {latest.epoch}</span>
      </div>
    </div>
  );
}

function StatusPill({ status }: { status: string }) {
  const normalized = status.toLowerCase().replaceAll("_", "-");
  return <span className={`status-pill status-${normalized}`}>{status}</span>;
}

export default function Dashboard() {
  const [papers, setPapers] = useState<Paper[]>([]);
  const [selectedPaperId, setSelectedPaperId] = useState<number | null>(null);
  const [paper, setPaper] = useState<PaperDetail | null>(null);
  const [understanding, setUnderstanding] = useState<Understanding | null>(null);
  const [claims, setClaims] = useState<Claim[]>([]);
  const [plan, setPlan] = useState<Plan | null>(null);
  const [comparison, setComparison] = useState<Comparison | null>(null);
  const [experiments, setExperiments] = useState<Experiment[]>([]);
  const [activeExperiment, setActiveExperiment] = useState<Experiment | null>(null);
  const [logs, setLogs] = useState<LogItem[]>([]);
  const [pipeline, setPipeline] = useState<PipelineStatus | null>(null);
  const [apiOnline, setApiOnline] = useState<boolean | null>(null);
  const [loadingPapers, setLoadingPapers] = useState(true);
  const [loadingPaper, setLoadingPaper] = useState(false);
  const [busy, setBusy] = useState<string | null>(null);
  const [actionError, setActionError] = useState("");
  const [notice, setNotice] = useState("");
  const [sourceType, setSourceType] = useState<"url" | "arxiv" | "local">("url");
  const [sourceRef, setSourceRef] = useState("");
  const [uploadFile, setUploadFile] = useState<File | null>(null);
  const [epochs, setEpochs] = useState(5);
  const [batchSize, setBatchSize] = useState(64);
  const [learningRate, setLearningRate] = useState(0.001);
  const logCursor = useRef(0);

  const refreshLists = useCallback(async () => {
    setLoadingPapers(true);
    try {
      const [paperResponse, status, experimentResponse] = await Promise.all([
        listPapers(),
        getPipelineStatus(),
        listExperiments(),
      ]);
      setPapers(paperResponse.papers);
      setPipeline(status);
      setExperiments(experimentResponse.experiments);
      setApiOnline(true);
      setSelectedPaperId((current) => {
        if (current && paperResponse.papers.some((item) => item.id === current)) {
          return current;
        }
        return paperResponse.papers[0]?.id ?? null;
      });
      return true;
    } catch (error) {
      setApiOnline(false);
      setActionError(errorMessage(error));
      return false;
    } finally {
      setLoadingPapers(false);
    }
  }, []);

  useEffect(() => {
    const initializeDashboard = async () => {
      try {
        await refreshLists();
        await checkHealth();
        setApiOnline(true);
      } catch {
        setApiOnline(false);
      }
    };
    void initializeDashboard();
  }, [refreshLists]);

  useEffect(() => {
    const refreshStatus = async () => {
      try {
        const status = await getPipelineStatus();
        setPipeline(status);
        setApiOnline(true);
      } catch {
        setApiOnline(false);
      }
    };
    const timer = window.setInterval(() => void refreshStatus(), 5000);
    return () => window.clearInterval(timer);
  }, []);

  useEffect(() => {
    if (selectedPaperId === null) {
      return;
    }

    let current = true;
    const loadPaperState = async () => {
      setLoadingPaper(true);
      setActionError("");
      try {
        const [detail, paperUnderstanding, claimResponse, paperPlan, report, runResponse] =
          await Promise.all([
            getPaper(selectedPaperId),
            getUnderstanding(selectedPaperId).catch((error: unknown) => {
              if (error instanceof ApiError && [404, 409].includes(error.status)) return null;
              throw error;
            }),
            getClaims(selectedPaperId),
            getPlan(selectedPaperId).catch((error: unknown) => {
              if (error instanceof ApiError && error.status === 404) return null;
              throw error;
            }),
            getPaperReport(selectedPaperId),
            listExperiments(selectedPaperId),
          ]);
        if (!current) return;
        setPaper(detail);
        setUnderstanding(paperUnderstanding);
        setClaims(claimResponse.claims);
        setPlan(paperPlan);
        setComparison(report.comparison);
        setExperiments(runResponse.experiments);
        setActiveExperiment((active) => {
          if (active && runResponse.experiments.some((run) => run.id === active.id)) {
            return active;
          }
          return runResponse.experiments[0] ?? null;
        });
      } catch (error) {
        if (current) setActionError(errorMessage(error));
      } finally {
        if (current) setLoadingPaper(false);
      }
    };
    void loadPaperState();

    return () => {
      current = false;
    };
  }, [selectedPaperId]);

  const activeExperimentId = activeExperiment?.id ?? null;

  useEffect(() => {
    if (!activeExperimentId) {
      logCursor.current = 0;
      return;
    }

    let stopped = false;
    logCursor.current = 0;
    const poll = async () => {
      if (stopped) return;
      try {
        const [latestExperiment, logResponse] = await Promise.all([
          getExperiment(activeExperimentId),
          getLogs(activeExperimentId, logCursor.current),
        ]);
        if (stopped) return;
        setActiveExperiment(latestExperiment);
        setExperiments((previous) =>
          previous.map((item) => (item.id === latestExperiment.id ? latestExperiment : item)),
        );
        setLogs((previous) => {
          const seen = new Set(previous.map((item) => item.id));
          return [...previous, ...logResponse.logs.filter((item) => !seen.has(item.id))];
        });
        if (logResponse.logs.length) {
          logCursor.current = logResponse.logs[logResponse.logs.length - 1].id;
        }
        if (isTerminal(latestExperiment.status)) stopped = true;
      } catch (error) {
        if (!stopped) setActionError(errorMessage(error));
      }
    };

    void poll();
    const timer = window.setInterval(() => void poll(), 1800);
    return () => {
      stopped = true;
      window.clearInterval(timer);
    };
  }, [activeExperimentId]);

  const epochMetrics = useMemo(
    () =>
      logs
        .map(parseEpoch)
        .filter((item): item is EpochMetrics => item !== null)
        .sort((left, right) => left.epoch - right.epoch),
    [logs],
  );

  async function runAction<T>(
    name: string,
    action: () => Promise<T>,
  ): Promise<T | null> {
    setBusy(name);
    setActionError("");
    setNotice("");
    try {
      return await action();
    } catch (error) {
      setActionError(errorMessage(error));
      return null;
    } finally {
      setBusy(null);
    }
  }

  async function handleIngest(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const source = sourceRef.trim();
    if (!source) return;
    const created = await runAction("ingest", () => ingestSource(sourceType, source));
    if (!created) return;
    setSourceRef("");
    setNotice("Paper ingested from the selected source.");
    setActiveExperiment(null);
    setLogs([]);
    setPapers((previous) => [created, ...previous.filter((item) => item.id !== created.id)]);
    setSelectedPaperId(created.id);
    void refreshLists();
  }

  async function handleUpload(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!uploadFile) return;
    const created = await runAction("upload", () => uploadPaper(uploadFile));
    if (!created) return;
    setUploadFile(null);
    event.currentTarget.reset();
    setNotice("PDF uploaded and parsed.");
    setActiveExperiment(null);
    setLogs([]);
    setPapers((previous) => [created, ...previous.filter((item) => item.id !== created.id)]);
    setSelectedPaperId(created.id);
    void refreshLists();
  }

  async function handleAnalyze() {
    if (!selectedPaperId) return;
    const result = await runAction("analyze", async () => {
      const paperUnderstanding = await analyzePaper(selectedPaperId);
      const [claimResponse, report] = await Promise.all([
        getClaims(selectedPaperId),
        getPaperReport(selectedPaperId),
      ]);
      return { paperUnderstanding, claimResponse, report };
    });
    if (!result) return;
    setUnderstanding(result.paperUnderstanding);
    setClaims(result.claimResponse.claims);
    setComparison(result.report.comparison);
    setNotice("Evidence-checked analysis saved.");
  }

  async function handlePlan() {
    if (!selectedPaperId) return;
    const result = await runAction("plan", async () => {
      const paperPlan = await createPlan(selectedPaperId);
      const report = await getPaperReport(selectedPaperId);
      return { paperPlan, report };
    });
    if (!result) return;
    setPlan(result.paperPlan);
    setComparison(result.report.comparison);
    setNotice(`Reproduction plan ${result.paperPlan.status.toLowerCase()} and saved.`);
  }

  async function handleExperiment() {
    if (!selectedPaperId) return;
    const created = await runAction("experiment", () =>
      startExperiment(selectedPaperId, plan?.plan_id, {
        epochs,
        batch_size: batchSize,
        learning_rate: learningRate,
        device: "auto",
      }),
    );
    if (!created) return;
    setLogs([]);
    setActiveExperiment(created);
    setExperiments((previous) => [created, ...previous.filter((item) => item.id !== created.id)]);
    setNotice("Experiment queued. Live output will appear below.");
  }

  async function handleCompare() {
    if (!plan) return;
    const result = await runAction("compare", () => comparePlan(plan.plan_id));
    if (!result) return;
    setComparison(result);
    setNotice("Comparison saved from the available paper and run evidence.");
  }

  const epochsComplete = epochMetrics.length
    ? `${epochMetrics.length} / ${epochMetrics[epochMetrics.length - 1].totalEpochs}`
    : "—";
  const paperName = paper?.title ?? "Select or add a paper";

  return (
    <div className="app-frame min-h-screen bg-[#0b1118] text-[#edf5f4]">
      <header className="topbar sticky top-0 z-20">
        <Link className="brand" href="/" aria-label="PAPER2PRODUCT home">
          <span className="brand-mark">P</span>
          <span>
            <strong>paper<span>2</span>product</strong>
            <small>REPRODUCTION WORKSPACE</small>
          </span>
        </Link>
        <div className="topbar-meta">
          <span className={`api-indicator ${apiOnline ? "online" : apiOnline === false ? "offline" : ""}`}>
            <i />
            {apiOnline ? "API connected" : apiOnline === false ? "API unavailable" : "Connecting"}
          </span>
          <span className="topbar-divider" />
          <span className="topbar-caption">LOCAL RESEARCH ENVIRONMENT</span>
          <button
            className="refresh-button"
            type="button"
            aria-label="Refresh papers and workspace status"
            onClick={() => {
              setLoadingPapers(true);
              void refreshLists();
              void checkHealth()
                .then(() => setApiOnline(true))
                .catch(() => setApiOnline(false));
            }}
          >
            ↻
          </button>
        </div>
      </header>

      <main className="workspace">
        <section className="welcome-row">
          <div>
            <div className="eyebrow"><span className="eyebrow-line" /> PAPER REPRODUCTION, WITH RECEIPTS</div>
            <h1>From paper to <em>evidence.</em></h1>
            <p className="welcome-copy">
              Inspect a paper, reconstruct its method, run an experiment, and compare what the evidence supports.
            </p>
          </div>
          <div className="pipeline-card">
            <div className="pipeline-label">WORKSPACE STATUS</div>
            <div className="pipeline-state">
              <span className="pipeline-dot" />
              {pipeline?.latest_pipeline_status ?? (apiOnline ? "READY" : "WAITING FOR API")}
            </div>
            <div className="pipeline-counts">
              <span><strong>{pipeline?.papers ?? "—"}</strong> papers</span>
              <span className="count-divider" />
              <span><strong>{Object.values(pipeline?.experiments_by_status ?? {}).reduce((sum, value) => sum + value, 0) || "—"}</strong> runs</span>
            </div>
          </div>
        </section>

        {(actionError || notice) && (
          <div className={`notice-banner ${actionError ? "notice-error" : "notice-success"}`} role={actionError ? "alert" : "status"}>
            <span>{actionError || notice}</span>
            <button
              type="button"
              className="notice-close"
              aria-label="Dismiss message"
              onClick={() => {
                setActionError("");
                setNotice("");
              }}
            >
              ×
            </button>
          </div>
        )}

        <div className="dashboard-grid">
          <aside className="sidebar">
            <section className="panel source-panel">
              <div className="panel-heading">
                <div>
                  <div className="eyebrow-small">START HERE</div>
                  <h2>Add a paper</h2>
                </div>
                <span className="step-number">01</span>
              </div>
              <p className="panel-hint">Bring a source into the workspace. The API parses the original document.</p>

              <form onSubmit={handleIngest} className="source-form">
                <label className="field-label" htmlFor="source-kind">SOURCE TYPE</label>
                <select
                  id="source-kind"
                  value={sourceType}
                  onChange={(event) => setSourceType(event.target.value as "url" | "arxiv" | "local")}
                >
                  <option value="url">Paper URL</option>
                  <option value="arxiv">arXiv identifier or URL</option>
                  <option value="local">Local PDF path</option>
                </select>
                <label className="field-label" htmlFor="source-ref">
                  {sourceType === "local" ? "FILE PATH" : sourceType === "arxiv" ? "ARXIV SOURCE" : "PDF URL"}
                </label>
                <input
                  id="source-ref"
                  value={sourceRef}
                  onChange={(event) => setSourceRef(event.target.value)}
                  placeholder={
                    sourceType === "local"
                      ? "C:\\papers\\research.pdf"
                      : sourceType === "arxiv"
                        ? "2401.12345"
                        : "https://example.org/paper.pdf"
                  }
                  required
                />
                <button className="button button-dark button-full" type="submit" disabled={busy !== null || !sourceRef.trim()}>
                  {busy === "ingest" ? <><span className="spinner" /> Parsing source…</> : <>Ingest source <span>↗</span></>}
                </button>
              </form>

              <div className="form-separator"><span>OR UPLOAD A PDF</span></div>
              <form onSubmit={handleUpload} className="upload-form">
                <label className="upload-drop" htmlFor="paper-upload">
                  <span className="upload-icon">↑</span>
                  <span className="upload-text">{uploadFile?.name ?? "Choose a PDF file"}</span>
                  <span className="upload-subtext">PDF · up to 50 MB</span>
                </label>
                <input
                  id="paper-upload"
                  className="visually-hidden"
                  type="file"
                  accept="application/pdf,.pdf"
                  onChange={(event) => setUploadFile(event.target.files?.[0] ?? null)}
                />
                <button className="button button-outline button-full" type="submit" disabled={busy !== null || !uploadFile}>
                  {busy === "upload" ? <><span className="spinner" /> Uploading…</> : "Upload and parse"}
                </button>
              </form>
            </section>

            <section className="panel library-panel">
              <div className="panel-heading">
                <div>
                  <div className="eyebrow-small">YOUR LIBRARY</div>
                  <h2>Research papers</h2>
                </div>
                <span className="library-count">{loadingPapers ? "…" : papers.length.toString().padStart(2, "0")}</span>
              </div>
              {loadingPapers && papers.length === 0 ? (
                <div className="empty-library"><span className="spinner" /> Loading saved papers…</div>
              ) : papers.length === 0 ? (
                <div className="empty-library">
                  <span className="empty-icon">⌁</span>
                  <strong>No papers yet</strong>
                  <span>Ingest a URL or upload a PDF to start a reproducible workflow.</span>
                </div>
              ) : (
                <ul className="paper-list">
                  {papers.map((item, index) => (
                    <li key={item.id}>
                      <button
                        type="button"
                        className={`paper-list-item ${selectedPaperId === item.id ? "selected" : ""}`}
                        onClick={() => {
                          setSelectedPaperId(item.id);
                          setActiveExperiment(null);
                          setLogs([]);
                        }}
                      >
                        <span className="paper-index">{String(index + 1).padStart(2, "0")}</span>
                        <span className="paper-list-copy">
                          <strong>{item.title}</strong>
                          <small>{item.source} · {item.section_count} sections</small>
                        </span>
                        <span className="paper-arrow">↗</span>
                      </button>
                    </li>
                  ))}
                </ul>
              )}
            </section>
          </aside>

          <div className="main-column">
            <section className="panel paper-overview">
              {loadingPaper ? (
                <div className="loading-state"><span className="spinner" /> Loading paper evidence…</div>
              ) : paper ? (
                <>
                  <div className="paper-overview-top">
                    <div className="paper-source-tag"><span /> {paper.source} SOURCE · ID {paper.id}</div>
                    <StatusPill status={paper.status} />
                  </div>
                  <h2 className="paper-title">{paperName}</h2>
                  <p className="paper-authors">{paper.authors?.join(" · ") || paper.source_ref || "Authors not identified"}</p>
                  <p className="paper-abstract">{paper.abstract || "No abstract was identified in the ingested document."}</p>
                  <div className="paper-facts">
                    <span><strong>{paper.sections.length}</strong> parsed sections</span>
                    <span className="fact-divider" />
                    <span><strong>{claims.length}</strong> supported claims</span>
                    <span className="fact-divider" />
                    <span><strong>{experiments.length}</strong> experiment{experiments.length === 1 ? "" : "s"}</span>
                  </div>
                </>
              ) : (
                <div className="paper-placeholder">
                  <span className="placeholder-mark">P.</span>
                  <div>
                    <div className="eyebrow-small">YOUR NEXT REPRODUCTION STARTS HERE</div>
                    <h2>{apiOnline === false ? "Connect to the PAPER2PRODUCT API" : "No paper selected"}</h2>
                    <p>{apiOnline === false ? "Start the backend API, then refresh to load persisted research and experiment data." : "Ingest a paper or select one from your library to inspect source-backed evidence."}</p>
                  </div>
                </div>
              )}
            </section>

            {paper && (
              <>
                <section className="workflow-section">
                  <div className="section-heading">
                    <div>
                      <div className="eyebrow-small">EVIDENCE PIPELINE</div>
                      <h2>Move from reading to testing</h2>
                    </div>
                    <span className="section-note">Each stage is saved to the API</span>
                  </div>
                  <div className="workflow-grid">
                    <article className={`workflow-card ${understanding ? "workflow-complete" : ""}`}>
                      <div className="workflow-number">01 <span>UNDERSTAND</span></div>
                      <h3>Extract source facts</h3>
                      <p>Find methods, datasets, and results with quotes tied to parsed sections.</p>
                      <button className="text-button" type="button" onClick={handleAnalyze} disabled={busy !== null}>
                        {busy === "analyze" ? "Analyzing…" : understanding ? "Refresh analysis" : "Analyze paper"} <span>→</span>
                      </button>
                    </article>
                    <article className={`workflow-card ${plan ? "workflow-complete" : ""}`}>
                      <div className="workflow-number">02 <span>PLAN</span></div>
                      <h3>Map the reproduction</h3>
                      <p>Compare paper-reported setup with the available run and record missing details.</p>
                      <button className="text-button" type="button" onClick={handlePlan} disabled={busy !== null || !understanding}>
                        {busy === "plan" ? "Building plan…" : plan ? "Refresh plan" : "Create plan"} <span>→</span>
                      </button>
                    </article>
                    <article className={`workflow-card ${activeExperiment?.status === "COMPLETED" ? "workflow-complete" : ""}`}>
                      <div className="workflow-number">03 <span>EXPERIMENT</span></div>
                      <h3>Run and observe</h3>
                      <p>Launch the real PyTorch experiment and stream persisted stdout and metrics.</p>
                      <button className="text-button" type="button" onClick={handleExperiment} disabled={busy !== null}>
                        {busy === "experiment" ? "Queueing…" : "Configure a run"} <span>→</span>
                      </button>
                    </article>
                    <article className={`workflow-card ${comparison ? "workflow-complete" : ""}`}>
                      <div className="workflow-number">04 <span>COMPARE</span></div>
                      <h3>Report what matches</h3>
                      <p>Score only when source quotes and completed run metrics are available.</p>
                      <button className="text-button" type="button" onClick={handleCompare} disabled={busy !== null || !plan}>
                        {busy === "compare" ? "Comparing…" : comparison ? "Refresh comparison" : "Compare evidence"} <span>→</span>
                      </button>
                    </article>
                  </div>
                  {!understanding && (
                    <p className="workflow-note">Analyze this paper first to unlock evidence-backed planning.</p>
                  )}
                </section>

                {understanding && (
                  <section className="panel analysis-panel">
                    <div className="section-heading">
                      <div>
                        <div className="eyebrow-small">PAPER UNDERSTANDING</div>
                        <h2>Claims with source evidence</h2>
                      </div>
                      <span className={`confidence-tag ${understanding.confidence_level === "lower" ? "confidence-low" : ""}`}>
                        {understanding.extraction_method} · {Math.round(understanding.confidence_score * 100)}% confidence
                      </span>
                    </div>
                    {claims.length === 0 ? (
                      <div className="inline-empty">Analysis completed, but no supported claims were extracted from the available sections.</div>
                    ) : (
                      <div className="claim-list">
                        {claims.slice(0, 8).map((claim) => (
                          <article className="claim-card" key={claim.id}>
                            <div className="claim-meta">
                              <span className="claim-kind">{claim.claim_type}</span>
                              <span>SECTION {claim.section_id}</span>
                              <span>{Math.round(claim.confidence * 100)}% confidence</span>
                            </div>
                            <p>{claim.statement}</p>
                            <blockquote>“{claim.evidence}”</blockquote>
                          </article>
                        ))}
                        {claims.length > 8 && <p className="more-claims">Showing 8 of {claims.length} source-backed claims.</p>}
                      </div>
                    )}
                    {understanding.unsupported_facts.length > 0 && (
                      <p className="unsupported-note">
                        {understanding.unsupported_facts.length} candidate fact{understanding.unsupported_facts.length === 1 ? "" : "s"} were excluded because source support could not be verified.
                      </p>
                    )}
                  </section>
                )}

                {plan && (
                  <section className="panel plan-panel">
                    <div className="section-heading">
                      <div>
                        <div className="eyebrow-small">REPRODUCTION PLAN · {plan.plan_id}</div>
                        <h2>Setup, differences, unknowns</h2>
                      </div>
                      <StatusPill status={plan.status} />
                    </div>
                    <p className="plan-objective">{plan.objective}</p>
                    {plan.blockers.length > 0 && (
                      <div className="blocker-box">
                        <strong>Plan blockers</strong>
                        {plan.blockers.map((blocker) => <span key={blocker}>• {blocker}</span>)}
                      </div>
                    )}
                    {plan.differences.length > 0 ? (
                      <div className="difference-list">
                        {plan.differences.map((difference) => (
                          <article className="difference-row" key={difference.dimension}>
                            <strong>{difference.dimension}</strong>
                            <div>
                              <span className="difference-label">PAPER</span>
                              <p>{difference.original_setup}</p>
                            </div>
                            <div>
                              <span className="difference-label">OBSERVED RUN</span>
                              <p>{difference.observed_setup}</p>
                            </div>
                            <p className="difference-significance">{difference.significance}</p>
                          </article>
                        ))}
                      </div>
                    ) : (
                      <div className="inline-empty">No setup differences were available from the persisted plan.</div>
                    )}
                    {plan.missing_information.length > 0 && (
                      <div className="missing-info">
                        <span className="difference-label">NOT IDENTIFIED IN SOURCE</span>
                        <p>{plan.missing_information.join(" · ")}</p>
                      </div>
                    )}
                  </section>
                )}

                <section className="panel experiment-panel">
                  <div className="section-heading">
                    <div>
                      <div className="eyebrow-small">LIVE EXPERIMENT</div>
                      <h2>Training metrics &amp; output</h2>
                    </div>
                    {activeExperiment && <StatusPill status={activeExperiment.status} />}
                  </div>

                  <div className="run-toolbar">
                    <div className="run-config">
                      <label>
                        <span>EPOCHS</span>
                        <input type="number" min={1} max={1000} value={epochs} onChange={(event) => setEpochs(Number(event.target.value))} />
                      </label>
                      <label>
                        <span>BATCH SIZE</span>
                        <input type="number" min={1} max={8192} value={batchSize} onChange={(event) => setBatchSize(Number(event.target.value))} />
                      </label>
                      <label>
                        <span>LEARNING RATE</span>
                        <input type="number" min={0.000001} step="any" value={learningRate} onChange={(event) => setLearningRate(Number(event.target.value))} />
                      </label>
                      <button className="button button-accent" type="button" onClick={handleExperiment} disabled={busy !== null || epochs < 1 || batchSize < 1 || learningRate <= 0}>
                        {busy === "experiment" ? <><span className="spinner" /> Queuing</> : <>Start experiment <span>↗</span></>}
                      </button>
                    </div>
                    <div className="run-selection">
                      <label htmlFor="experiment-select">VIEW SAVED RUN</label>
                      <select
                        id="experiment-select"
                        value={activeExperiment?.id ?? ""}
                        onChange={(event) => {
                          setLogs([]);
                          setActiveExperiment(experiments.find((item) => item.id === event.target.value) ?? null);
                        }}
                      >
                        <option value="">No experiment selected</option>
                        {experiments.map((item) => (
                          <option value={item.id} key={item.id}>
                            {item.started_at ? new Date(item.started_at).toLocaleString() : item.id.slice(0, 8)} · {item.status}
                          </option>
                        ))}
                      </select>
                    </div>
                  </div>

                  {activeExperiment ? (
                    <>
                      <div className="run-summary">
                        <span className="run-id">RUN <code>{activeExperiment.id}</code></span>
                        <span>{epochsComplete} epochs observed</span>
                        {activeExperiment.error && <span className="run-error">{activeExperiment.error}</span>}
                      </div>
                      <div className="charts-grid">
                        <MetricChart
                          title="Training loss"
                          data={epochMetrics}
                          valueFor={(item) => item.trainingLoss}
                          color="#c66c42"
                          format={(value) => value.toFixed(4)}
                        />
                        <MetricChart
                          title="Test loss"
                          data={epochMetrics}
                          valueFor={(item) => item.testLoss}
                          color="#518c7b"
                          format={(value) => value.toFixed(4)}
                        />
                        <MetricChart
                          title="Test accuracy"
                          data={epochMetrics}
                          valueFor={(item) => item.testAccuracy * 100}
                          color="#6475a7"
                          format={(value) => `${value.toFixed(2)}%`}
                        />
                      </div>
                      <div className="logs-heading">
                        <span>PROCESS OUTPUT</span>
                        <span>{logs.length} persisted lines</span>
                      </div>
                      {logs.length === 0 ? (
                        <div className="log-empty">{isTerminal(activeExperiment.status) ? "This run has no persisted output." : "Waiting for the experiment process to emit output…"}</div>
                      ) : (
                        <div className="log-window" role="log" aria-live="polite">
                          {logs.map((log) => (
                            <div className={`log-line log-${log.stream}`} key={log.id}>
                              <span className="log-time">{new Date(log.created_at).toLocaleTimeString()}</span>
                              <span className="log-stream">{log.stream}</span>
                              <code>{log.message}</code>
                            </div>
                          ))}
                        </div>
                      )}
                    </>
                  ) : (
                    <div className="run-empty">
                      <span className="run-empty-mark">∿</span>
                      <strong>No experiment selected</strong>
                      <span>Start a run to see real epoch metrics and process output. Values appear only after they are recorded by the backend.</span>
                    </div>
                  )}
                </section>

                {comparison && (
                  <section className="panel comparison-panel">
                    <div className="section-heading">
                      <div>
                        <div className="eyebrow-small">EVIDENCE COMPARISON</div>
                        <h2>What the results support</h2>
                      </div>
                      <StatusPill status={comparison.status} />
                    </div>
                    <div className="comparison-score">
                      <strong>{comparison.score === null ? "—" : comparison.score.toFixed(2)}</strong>
                      <span>{comparison.score === null ? "No score computed" : "reproducibility score / 100"}</span>
                    </div>
                    <p className="comparison-rationale">{comparison.rationale}</p>
                    {comparison.blockers.length > 0 && (
                      <div className="blocker-box">
                        <strong>Evidence blockers</strong>
                        {comparison.blockers.map((blocker) => <span key={blocker}>• {blocker}</span>)}
                      </div>
                    )}
                    <div className="verification-grid">
                      {Object.entries(comparison.verification).map(([heading, items]) => (
                        <div key={heading}>
                          <span className="difference-label">{heading}</span>
                          {items.map((item) => <p key={item}>{item}</p>)}
                        </div>
                      ))}
                    </div>
                  </section>
                )}
              </>
            )}
          </div>
        </div>

        <footer className="footer">
          <span>PAPER2PRODUCT <i>·</i> LOCAL-FIRST RESEARCH TOOLING</span>
          <span>{pipeline?.latest_comparison_id ? `LAST SAVED COMPARISON #${pipeline.latest_comparison_id}` : "NO COMPARISON SAVED"}</span>
        </footer>
      </main>
    </div>
  );
}
