# Milestone progress

| Milestone | Status | Evidence |
| --- | --- | --- |
| M1 — LeNet-MNIST experiment | Completed | Four pytest tests passed, including an actual 256-sample MNIST smoke run. The requested five-epoch CLI run completed on CPU with 60,000 training and 10,000 test examples. Final test accuracy: 0.9856. Run artifacts: `backend/experiments/lenet_mnist/runs/20261007T071717455463Z-011023e1` (ignored by Git). |
| M2 — Persistence and experiment orchestration | Completed | Eight backend pytest tests passed. SQLAlchemy models cover all nine requested SQLite entities. A subprocess integration test ran the actual M1 CLI, streamed stdout into database log rows, and verified terminal status, metrics, and all four artifacts; failure, timeout, stderr, and file-backed SQLite persistence are also tested. |
| M3 — Paper ingestion | Completed | Fourteen backend pytest tests passed. Retrieved the original PDF from the HAL repository mirror at `https://hal.science/hal-03926082/document`, verified its title (“Gradient-based learning applied to document recognition”), and parsed/stored 48 pages, author metadata, and 18 sections in SQLite. The PDF and database are under ignored `backend/data/`. |
| M4 — Paper understanding and claim extraction | Not started | — |
| M5 — Reproduction planning | Not started | — |
| M6 — Comparison and reproducibility scoring | Not started | — |
| M7 — API | Not started | — |
| M8 — Frontend | Not started | — |
| M9 — End-to-end workflow | Not started | — |
