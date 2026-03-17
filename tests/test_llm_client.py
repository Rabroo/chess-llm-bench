"""Tests for LLM client response parsing."""

import pytest

from src.llm_client import parse_response, build_prompt


class TestParseResponse:
    def test_complete_response(self, sample_llm_response):
        result = parse_response(sample_llm_response)
        assert result["eval"] == 45
        assert result["move"] == "Nf6"
        assert "Equal" in result["explanation"]
        assert result["side_claimed"] == "Equal"
        assert len(result["parse_errors"]) == 0

    def test_missing_eval(self):
        response = """Move: Nf6
Explanation: White is better — material advantage."""
        result = parse_response(response)
        assert result["eval"] is None
        assert result["move"] == "Nf6"
        assert "Missing Eval field" in result["parse_errors"]

    def test_missing_move(self):
        response = """Eval: 100
Explanation: White is better — material advantage."""
        result = parse_response(response)
        assert result["eval"] == 100
        assert result["move"] is None
        assert "Missing Move field" in result["parse_errors"]

    def test_missing_explanation(self):
        response = """Eval: 100
Move: Nf6"""
        result = parse_response(response)
        assert result["eval"] == 100
        assert result["move"] == "Nf6"
        assert result["explanation"] is None
        assert "Missing Explanation field" in result["parse_errors"]

    def test_negative_eval(self):
        response = """Eval: -150
Move: e5
Explanation: Black is better — White has weak pawns."""
        result = parse_response(response)
        assert result["eval"] == -150
        assert result["side_claimed"] == "Black"

    def test_eval_with_extra_text(self):
        response = """Eval: 100 centipawns
Move: Nf6
Explanation: White is better."""
        result = parse_response(response)
        assert result["eval"] == 100


class TestBuildPrompt:
    def test_fen_only(self):
        prompt = build_prompt(
            fen="rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            prompt_format="fen_only",
        )
        assert "Current position (FEN):" in prompt
        assert "Moves played so far:" not in prompt
        assert "Think step by step" not in prompt

    def test_pgn_fen(self):
        prompt = build_prompt(
            fen="rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1",
            pgn_moves="1. e4",
            prompt_format="pgn+fen",
        )
        assert "Moves played so far:" in prompt
        assert "1. e4" in prompt
        assert "Think step by step" not in prompt

    def test_cot_format(self):
        prompt = build_prompt(
            fen="rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1",
            pgn_moves="1. e4",
            prompt_format="cot",
        )
        assert "Think step by step" in prompt
        assert "Moves played so far:" in prompt
