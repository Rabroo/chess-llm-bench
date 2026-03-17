"""Tests for SQLite job queue."""

import os
import tempfile

import pytest

from src.job_queue import JobQueue


@pytest.fixture
def temp_db():
    """Create a temporary database file."""
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    yield path
    os.unlink(path)


@pytest.fixture
def job_queue(temp_db):
    """Create a job queue with temporary database."""
    return JobQueue(temp_db)


class TestJobQueue:
    def test_insert_job(self, job_queue, sample_job):
        result = job_queue.insert_job(sample_job)
        assert result is True
        assert job_queue.count_total() == 1

    def test_duplicate_hash_rejected(self, job_queue, sample_job):
        job_queue.insert_job(sample_job)
        result = job_queue.insert_job(sample_job)  # Same hash
        assert result is False
        assert job_queue.count_total() == 1

    def test_claim_job(self, job_queue, sample_job):
        job_queue.insert_job(sample_job)
        claimed = job_queue.claim_job("worker_1")
        assert claimed is not None
        assert claimed["job_id"] == sample_job["job_id"]
        assert claimed["status"] == "in_progress"

    def test_claim_empty_queue(self, job_queue):
        claimed = job_queue.claim_job("worker_1")
        assert claimed is None

    def test_complete_job(self, job_queue, sample_job):
        job_queue.insert_job(sample_job)
        job_queue.claim_job("worker_1")
        job_queue.complete_job(sample_job["job_id"])

        counts = job_queue.count_by_status()
        assert counts.get("done", 0) == 1
        assert counts.get("in_progress", 0) == 0

    def test_fail_job(self, job_queue, sample_job):
        job_queue.insert_job(sample_job)
        job_queue.claim_job("worker_1")
        job_queue.fail_job(sample_job["job_id"], "Test error")

        job = job_queue.get_job(sample_job["job_id"])
        assert job["status"] == "failed"
        assert job["error_message"] == "Test error"

    def test_reset_job(self, job_queue, sample_job):
        job_queue.insert_job(sample_job)
        job_queue.claim_job("worker_1")
        job_queue.reset_job(sample_job["job_id"])

        job = job_queue.get_job(sample_job["job_id"])
        assert job["status"] == "pending"
        assert job["worker_id"] is None

    def test_progress(self, job_queue, sample_job):
        job_queue.insert_job(sample_job)
        progress = job_queue.get_progress()
        assert progress["total"] == 1
        assert progress["pending"] == 1
        assert progress["percent_complete"] == 0
