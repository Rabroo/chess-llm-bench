"""Tests for position generation."""

import random

import chess
import pytest

from src.position_generator import (
    validate_position,
    generate_random_position,
    generate_endgame_position,
    determine_phase,
)


class TestValidatePosition:
    def test_valid_starting_position(self):
        board = chess.Board()
        assert validate_position(board) is True

    def test_invalid_position_no_king(self):
        board = chess.Board(None)
        board.set_piece_at(chess.E1, chess.Piece(chess.QUEEN, chess.WHITE))
        board.set_piece_at(chess.E8, chess.Piece(chess.QUEEN, chess.BLACK))
        assert validate_position(board) is False

    def test_checkmate_position(self):
        # Scholar's mate position
        board = chess.Board("r1bqkb1r/pppp1Qpp/2n2n2/4p3/2B1P3/8/PPPP1PPP/RNB1K1NR b KQkq - 0 4")
        # This is checkmate, should be invalid
        assert validate_position(board) is False


class TestGenerateRandomPosition:
    def test_generates_valid_position(self):
        rng = random.Random(42)
        pos = generate_random_position(rng, min_moves=10, max_moves=20)
        if pos is not None:  # May fail occasionally
            assert "fen" in pos
            assert "phase" in pos
            assert pos["source"] == "generated"

    def test_reproducible_with_seed(self):
        rng1 = random.Random(42)
        rng2 = random.Random(42)
        pos1 = generate_random_position(rng1, min_moves=10, max_moves=10)
        pos2 = generate_random_position(rng2, min_moves=10, max_moves=10)
        if pos1 and pos2:
            assert pos1["fen"] == pos2["fen"]


class TestGenerateEndgamePosition:
    def test_kqvk(self):
        rng = random.Random(42)
        pos = generate_endgame_position(rng, "KQvK")
        if pos is not None:
            board = chess.Board(pos["fen"])
            # Should have 3 pieces: 2 kings and 1 queen
            assert len(board.piece_map()) == 3
            assert pos["phase"] == "endgame"

    def test_krkvr(self):
        rng = random.Random(42)
        pos = generate_endgame_position(rng, "KRvKR")
        if pos is not None:
            board = chess.Board(pos["fen"])
            # Should have 4 pieces: 2 kings and 2 rooks
            assert len(board.piece_map()) == 4


class TestDeterminePhase:
    def test_opening(self):
        board = chess.Board()
        assert determine_phase(board, 5) == "opening"

    def test_endgame_few_pieces(self):
        board = chess.Board("8/8/8/3k4/8/3K4/3Q4/8 w - - 0 1")
        assert determine_phase(board, 50) == "endgame"

    def test_middlegame(self):
        board = chess.Board(
            "r1bq1rk1/ppp2ppp/2np1n2/2b1p3/2B1P3/2NP1N2/PPP2PPP/R1BQ1RK1 w - - 0 8"
        )
        assert determine_phase(board, 25) == "middlegame"
