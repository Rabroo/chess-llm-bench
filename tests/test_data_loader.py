"""Tests for data loader."""

import json
import os
import tempfile

import pytest

from src.data_loader import DataLoader


@pytest.fixture
def temp_data_dir():
    """Create temporary data directory with sample files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create sample dataset
        positions = [
            {
                "id": 1,
                "fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
                "theme": "opening",
                "difficulty": "easy",
                "phase": "opening",
                "source": "generated",
            },
            {
                "id": 2,
                "fen": "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3",
                "theme": "tactics",
                "difficulty": "easy",
                "phase": "opening",
                "source": "lichess_puzzles",
            },
        ]

        with open(os.path.join(tmpdir, "easy.json"), "w") as f:
            json.dump(positions, f)

        with open(os.path.join(tmpdir, "medium.json"), "w") as f:
            json.dump([], f)

        yield tmpdir


class TestDataLoader:
    def test_load_tier(self, temp_data_dir):
        loader = DataLoader(temp_data_dir)
        positions = loader.load_tier("easy")
        assert len(positions) == 2

    def test_load_missing_tier(self, temp_data_dir):
        loader = DataLoader(temp_data_dir)
        positions = loader.load_tier("extreme")  # Doesn't exist
        assert len(positions) == 0

    def test_filter_by_source(self, temp_data_dir):
        loader = DataLoader(temp_data_dir)
        positions = loader.filter(source="lichess_puzzles")
        assert len(positions) == 1
        assert positions[0]["id"] == 2

    def test_filter_by_theme(self, temp_data_dir):
        loader = DataLoader(temp_data_dir)
        positions = loader.filter(theme="opening")
        assert len(positions) == 1
        assert positions[0]["id"] == 1

    def test_get_by_id(self, temp_data_dir):
        loader = DataLoader(temp_data_dir)
        pos = loader.get_by_id(2)
        assert pos is not None
        assert pos["theme"] == "tactics"

    def test_get_by_id_not_found(self, temp_data_dir):
        loader = DataLoader(temp_data_dir)
        pos = loader.get_by_id(999)
        assert pos is None

    def test_sample(self, temp_data_dir):
        loader = DataLoader(temp_data_dir)
        positions = loader.sample(count=1, seed=42)
        assert len(positions) == 1

    def test_get_stats(self, temp_data_dir):
        loader = DataLoader(temp_data_dir)
        stats = loader.get_stats()
        assert stats["total"] == 2
        assert stats["by_difficulty"]["easy"] == 2
