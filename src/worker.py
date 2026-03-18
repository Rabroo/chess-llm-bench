"""Worker process for benchmark job execution."""

import logging
import os
import time
from typing import Any

from .data_loader import DataLoader
from .engine_wrapper import StockfishEngine
from .evaluator import score_all, should_trigger_correction
from .feedback_loop import CorrectionLoopManager
from .job_queue import JobQueue
from .llm_client import OllamaClient, build_prompt, parse_response
from .result_writer import ResultWriter, build_result_record, get_completed_job_ids
from .utils import load_config

logger = logging.getLogger("chess_llm_bench")


class Worker:
    """Benchmark worker that processes jobs from the queue."""

    def __init__(
        self,
        worker_id: str,
        config: dict[str, Any],
        dry_run: bool = False,
    ):
        """Initialize worker.

        Args:
            worker_id: Unique identifier for this worker
            config: Configuration dictionary
            dry_run: If True, don't write results
        """
        self.worker_id = worker_id
        self.config = config
        self.dry_run = dry_run

        # Initialize components — resolve paths to absolute to survive multiprocessing cwd changes
        paths = config.get("paths", {})
        self.job_queue = JobQueue(os.path.abspath(paths.get("jobs_db", "jobs/jobs.db")))
        self.result_writer = ResultWriter(
            os.path.abspath(paths.get("results_file", "results/evaluations.jsonl"))
        )
        self.data_loader = DataLoader(os.path.abspath(paths.get("data_dir", "data")))

        # Ollama client
        ollama_config = config.get("ollama", {})
        self.llm_client = OllamaClient(
            base_url=ollama_config.get("base_url", "http://localhost:11434"),
            timeout=ollama_config.get("timeout", 180),
            max_retries=ollama_config.get("max_retries", 3),
        )

        # Stockfish engine (for CPL calculation)
        stockfish_config = config.get("stockfish", {})
        self.engine: StockfishEngine | None = None
        try:
            self.engine = StockfishEngine(
                path=stockfish_config.get("path", "/usr/games/stockfish"),
                depth=stockfish_config.get("depth", 22),
                threads=stockfish_config.get("threads", 1),
            )
        except RuntimeError as e:
            logger.warning(f"Stockfish not available: {e}")

        # Correction loop manager
        self.correction_manager = CorrectionLoopManager(
            self.data_loader,
            self.job_queue,
            config,
        )

        # Evaluation settings
        eval_config = config.get("evaluation", {})
        eval_range = eval_config.get("centipawn_eval_range", {})
        self.eval_range = (
            eval_range.get("min", -2000),
            eval_range.get("max", 2000),
        )
        self.cpl_threshold = eval_config.get("cpl_threshold", 50)

        # Track completed jobs (for duplicate detection)
        self.completed_jobs = get_completed_job_ids(
            paths.get("results_file", "results/evaluations.jsonl")
        )

    def process_job(self, job: dict[str, Any]) -> dict[str, Any] | None:
        """Process a single benchmark job.

        Args:
            job: Job dictionary from the queue

        Returns:
            Result record or None if failed
        """
        job_id = job["job_id"]

        # Check if already completed
        if job_id in self.completed_jobs:
            logger.debug(f"Job {job_id} already completed, skipping")
            self.job_queue.complete_job(job_id)
            return None

        logger.info(f"Processing job {job_id} ({job['model']})")

        # Build position dict for scoring
        position = {
            "id": job["position_id"],
            "fen": job["fen"],
            "stockfish_eval": job.get("stockfish_eval", 0),
            "stockfish_best_move": job.get("stockfish_best_move", ""),
            "theme": job.get("theme", ""),
        }

        # If stockfish eval not in job, try to get from data loader
        if "stockfish_eval" not in job:
            loaded_pos = self.data_loader.get_by_id(job["position_id"])
            if loaded_pos:
                position["stockfish_eval"] = loaded_pos.get("stockfish_eval", 0)
                position["stockfish_best_move"] = loaded_pos.get(
                    "stockfish_best_move", ""
                )

        # Build prompt
        prompt = build_prompt(
            fen=job["fen"],
            pgn_moves=job.get("pgn_moves"),
            prompt_format=job.get("prompt_format", "pgn+fen"),
        )

        # Send to LLM
        llm_result = self.llm_client.chat(job["model"], prompt)

        if not llm_result["success"]:
            error_msg = llm_result.get("error", "Unknown error")
            logger.error(f"Job {job_id} failed: {error_msg}")
            self.job_queue.fail_job(job_id, error_msg)
            return None

        # Parse response
        parsed = parse_response(llm_result["response"])

        # Check if all fields are missing
        if (
            parsed["eval"] is None
            and parsed["move"] is None
            and parsed["explanation"] is None
        ):
            error_msg = "All three fields missing from response"
            logger.error(f"Job {job_id}: {error_msg}")
            self.job_queue.fail_job(job_id, error_msg)
            return None

        # Score all tasks
        scores = score_all(
            parsed_response=parsed,
            position=position,
            engine=self.engine,
            eval_range=self.eval_range,
        )

        # Build result record
        result = build_result_record(
            job=job,
            parsed_response=parsed,
            scores=scores,
            inference_ms=llm_result["inference_ms"],
        )

        # Write result
        if not self.dry_run:
            self.result_writer.write_result(result)
            self.completed_jobs.add(job_id)

        # Mark job complete
        self.job_queue.complete_job(job_id)

        # Check for correction loop trigger
        correction_enabled = self.config.get("correction_loop", {}).get(
            "enabled", True
        )
        if (
            correction_enabled
            and job.get("job_type") == "standard"
            and should_trigger_correction(scores.get("t2_cpl"), self.cpl_threshold)
        ):
            self.correction_manager.trigger_correction(job, result)

        logger.info(
            f"Completed job {job_id}: "
            f"T1_err={scores.get('t1_absolute_error')}, "
            f"T2_legal={scores.get('t2_legal')}, "
            f"T3_score={scores.get('t3_score')}"
        )

        return result

    def run(self, max_jobs: int | None = None) -> int:
        """Run the worker loop.

        Args:
            max_jobs: Maximum number of jobs to process (None for unlimited)

        Returns:
            Number of jobs processed
        """
        # Check Ollama availability
        if not self.llm_client.is_available():
            logger.error("Ollama is not running. Aborting.")
            raise RuntimeError("Ollama not available")

        jobs_processed = 0

        try:
            while True:
                # Check job limit
                if max_jobs is not None and jobs_processed >= max_jobs:
                    logger.info(f"Reached job limit ({max_jobs})")
                    break

                # Claim next job
                job = self.job_queue.claim_job(self.worker_id)

                if job is None:
                    logger.info("No more pending jobs")
                    break

                # Process job
                try:
                    result = self.process_job(job)
                    if result is not None:
                        jobs_processed += 1
                except Exception as e:
                    logger.exception(f"Error processing job {job['job_id']}: {e}")
                    self.job_queue.fail_job(job["job_id"], str(e))

        finally:
            # Cleanup
            if self.engine:
                self.engine.close()

        logger.info(f"Worker {self.worker_id} processed {jobs_processed} jobs")
        return jobs_processed


def run_worker(
    worker_id: str,
    config_path: str = "config/config.yaml",
    max_jobs: int | None = None,
    dry_run: bool = False,
) -> int:
    """Run a single worker process.

    Args:
        worker_id: Unique worker identifier
        config_path: Path to configuration file
        max_jobs: Maximum jobs to process
        dry_run: If True, don't write results

    Returns:
        Number of jobs processed
    """
    config = load_config(config_path)
    worker = Worker(worker_id, config, dry_run=dry_run)
    return worker.run(max_jobs=max_jobs)
