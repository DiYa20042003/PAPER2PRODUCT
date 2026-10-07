import math

from sqlalchemy import URL, create_engine, inspect, select, text
from sqlalchemy.orm import sessionmaker

from app.database import initialize_database, make_engine
from app.experiment_service import run_m1_experiment
from app.models import Artifact, Base, Experiment, ExperimentLog, Metric, Paper


def make_test_factory() -> sessionmaker:
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine, expire_on_commit=False)


def test_experiment_subprocess_streams_logs_and_persists_artifacts() -> None:
    session_factory = make_test_factory()
    experiment_id = run_m1_experiment(
        session_factory,
        epochs=1,
        batch_size=64,
        device="cpu",
        max_train_samples=256,
        max_test_samples=128,
        timeout_seconds=120,
    )

    with session_factory() as session:
        experiment = session.get(Experiment, experiment_id)
        logs = list(
            session.scalars(
                select(ExperimentLog).where(ExperimentLog.experiment_id == experiment_id)
            )
        )
        artifacts = list(
            session.scalars(select(Artifact).where(Artifact.experiment_id == experiment_id))
        )
        metrics = list(
            session.scalars(select(Metric).where(Metric.experiment_id == experiment_id))
        )
        assert experiment is not None
        assert experiment.status == "COMPLETED"
        assert experiment.exit_code == 0
        assert experiment.started_at is not None
        assert experiment.finished_at is not None
        assert experiment.config_path is not None
        assert experiment.metrics_path is not None
        assert any(log.stream == "stdout" and "epoch=1/1" in log.message for log in logs)
        assert {artifact.kind for artifact in artifacts} == {
            "config",
            "metrics",
            "log",
            "model",
        }
        assert {"training_loss", "test_loss", "test_accuracy"} <= {
            metric.name for metric in metrics
        }
        assert all(math.isfinite(metric.value) for metric in metrics)


def test_timed_out_experiment_is_marked_failed() -> None:
    session_factory = make_test_factory()
    experiment_id = run_m1_experiment(
        session_factory,
        epochs=5,
        device="cpu",
        timeout_seconds=0.01,
    )

    with session_factory() as session:
        experiment = session.get(Experiment, experiment_id)
        assert experiment is not None
        assert experiment.status == "FAILED"
        assert experiment.error is not None
        assert "timeout" in experiment.error.lower()
        assert experiment.finished_at is not None


def test_failure_stderr_is_persisted_and_all_m2_tables_exist() -> None:
    session_factory = make_test_factory()
    experiment_id = run_m1_experiment(
        session_factory,
        epochs=0,
        device="cpu",
        timeout_seconds=120,
    )

    with session_factory() as session:
        experiment = session.get(Experiment, experiment_id)
        logs = list(
            session.scalars(
                select(ExperimentLog).where(ExperimentLog.experiment_id == experiment_id)
            )
        )
        assert experiment is not None
        assert experiment.status == "FAILED"
        assert any(
            log.stream == "stderr" and "--epochs must be greater than zero" in log.message
            for log in logs
        )

    assert set(Base.metadata.tables) == {
        "papers",
        "paper_sections",
        "claims",
        "reproduction_plans",
        "experiments",
        "metrics",
        "logs",
        "artifacts",
        "reproducibility_scores",
    }


def test_sqlite_database_persists_paper_records(tmp_path) -> None:
    database_path = tmp_path / "paper2product.sqlite3"
    engine = make_engine(URL.create("sqlite", database=str(database_path)))
    initialize_database(engine)
    session_factory = sessionmaker(bind=engine, expire_on_commit=False)
    with session_factory() as session:
        session.add(Paper(title="Persisted paper", source="test"))
        session.commit()
    engine.dispose()

    reopened_engine = create_engine(
        URL.create("sqlite", database=str(database_path)),
        connect_args={"check_same_thread": False},
    )
    with sessionmaker(bind=reopened_engine)() as session:
        paper = session.scalar(select(Paper).where(Paper.title == "Persisted paper"))
        assert paper is not None
        assert paper.source == "test"
    reopened_engine.dispose()


def test_database_initialization_adds_new_paper_metadata_columns() -> None:
    engine = create_engine("sqlite://")
    with engine.begin() as connection:
        connection.execute(
            text(
                "CREATE TABLE papers ("
                "id INTEGER PRIMARY KEY, title TEXT, source VARCHAR(32), "
                "source_ref TEXT, abstract TEXT, status VARCHAR(32), created_at DATETIME)"
            )
        )
    initialize_database(engine)
    columns = {column["name"] for column in inspect(engine).get_columns("papers")}
    assert {"authors_json", "metadata_json", "understanding_json"} <= columns
    engine.dispose()


def test_database_initialization_makes_score_and_run_optional() -> None:
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    with engine.begin() as connection:
        connection.exec_driver_sql("DROP TABLE reproducibility_scores")
        connection.exec_driver_sql(
            "CREATE TABLE reproducibility_scores ("
            "id INTEGER PRIMARY KEY, experiment_id VARCHAR(36) NOT NULL, "
            "score FLOAT NOT NULL, status VARCHAR(32) NOT NULL, "
            "explanation_json TEXT NOT NULL, created_at DATETIME NOT NULL)"
        )
        connection.exec_driver_sql(
            "CREATE INDEX ix_reproducibility_scores_experiment_id "
            "ON reproducibility_scores (experiment_id)"
        )
    initialize_database(engine)
    columns = {
        column["name"]: column for column in inspect(engine).get_columns("reproducibility_scores")
    }
    assert columns["experiment_id"]["nullable"] is True
    assert columns["score"]["nullable"] is True
    assert "plan_id" in columns
    engine.dispose()
