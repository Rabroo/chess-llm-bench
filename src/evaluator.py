"""Scoring logic for T1, T2, and T3 benchmark tasks."""

import logging
from typing import Any

import chess

from .utils import clamp

logger = logging.getLogger("chess_llm_bench")


# Theme synonyms for T3 scoring
THEME_SYNONYMS = {
    "fork": ["fork", "double attack", "knight fork", "family fork"],
    "pin": ["pin", "pinned", "pinning", "absolute pin", "relative pin"],
    "skewer": ["skewer", "skewered"],
    "passed_pawn": ["passed pawn", "passer", "outside passer", "connected passers"],
    "discovery": ["discovered", "discovery", "discovered attack", "discovered check"],
    "deflection": ["deflection", "deflect", "decoy"],
    "sacrifice": ["sacrifice", "sac", "sacrificing"],
    "back_rank": ["back rank", "backrank", "back-rank mate"],
    "hanging": ["hanging", "undefended", "loose piece"],
    "trapped": ["trapped", "trap"],
    "overloaded": ["overloaded", "overworked"],
    "zwischenzug": ["zwischenzug", "intermezzo", "in-between move", "intermediate"],
    "interference": ["interference"],
    "clearance": ["clearance"],
    "undermining": ["undermining", "removing the defender"],
    "attraction": ["attraction", "attract"],
    "mate": ["mate", "checkmate", "mating"],
    "endgame": ["endgame", "ending"],
    "tactics": ["tactics", "tactical"],
    "game_position": ["position", "positional"],
    "random_play": ["position", "positional"],
}


def get_direction(eval_cp: int, threshold: int = 50) -> str:
    """Determine which side is better based on centipawn evaluation.

    Args:
        eval_cp: Centipawn evaluation from White's perspective
        threshold: Threshold for considering position equal

    Returns:
        "White", "Black", or "Equal"
    """
    if eval_cp > threshold:
        return "White"
    elif eval_cp < -threshold:
        return "Black"
    else:
        return "Equal"


def score_t1(
    model_eval: int | None,
    stockfish_eval: int,
    eval_range: tuple[int, int] = (-2000, 2000),
) -> dict[str, Any]:
    """Score Task 1: Centipawn Evaluation.

    Args:
        model_eval: Model's centipawn evaluation (may be None if parsing failed)
        stockfish_eval: Stockfish's ground truth evaluation
        eval_range: Range to clamp model evaluation

    Returns:
        Dictionary with T1 scoring results
    """
    if model_eval is None:
        return {
            "t1_model_eval": None,
            "t1_stockfish_eval": stockfish_eval,
            "t1_absolute_error": None,
            "t1_direction_correct": None,
        }

    # Clamp model evaluation to range
    clamped_eval = int(clamp(model_eval, eval_range[0], eval_range[1]))

    absolute_error = abs(clamped_eval - stockfish_eval)
    direction_correct = get_direction(clamped_eval) == get_direction(stockfish_eval)

    return {
        "t1_model_eval": clamped_eval,
        "t1_stockfish_eval": stockfish_eval,
        "t1_absolute_error": absolute_error,
        "t1_direction_correct": direction_correct,
    }


def score_t2(
    model_move: str | None,
    fen: str,
    stockfish_best_move: str,
    stockfish_eval: int,
    engine=None,
) -> dict[str, Any]:
    """Score Task 2: Best Move.

    Args:
        model_move: Model's move in SAN notation (may be None if parsing failed)
        fen: Position FEN
        stockfish_best_move: Stockfish's best move
        stockfish_eval: Stockfish evaluation before the move
        engine: Optional Stockfish engine for CPL calculation

    Returns:
        Dictionary with T2 scoring results
    """
    if model_move is None:
        return {
            "t2_move": None,
            "t2_best_move": stockfish_best_move,
            "t2_legal": False,
            "t2_cpl": None,
        }

    # Check legality
    try:
        board = chess.Board(fen)
        move = board.parse_san(model_move)
        is_legal = move in board.legal_moves
    except Exception:
        is_legal = False

    if not is_legal:
        return {
            "t2_move": model_move,
            "t2_best_move": stockfish_best_move,
            "t2_legal": False,
            "t2_cpl": None,
        }

    # Check if it's the best move
    is_best = model_move == stockfish_best_move

    # Calculate CPL if engine is available
    cpl = None
    if engine is not None:
        try:
            eval_after = engine.evaluate_after_move(fen, model_move)
            # CPL from the perspective of the side that moved
            board = chess.Board(fen)
            if board.turn == chess.WHITE:
                cpl = stockfish_eval - eval_after
            else:
                cpl = eval_after - stockfish_eval
            # CPL should be non-negative (best move has CPL 0)
            cpl = max(0, cpl)
        except Exception as e:
            logger.warning(f"CPL calculation failed: {e}")
            cpl = None
    elif is_best:
        cpl = 0

    return {
        "t2_move": model_move,
        "t2_best_move": stockfish_best_move,
        "t2_legal": True,
        "t2_cpl": cpl,
    }


def score_t3(
    explanation: str | None,
    side_claimed: str | None,
    stockfish_eval: int,
    theme: str,
) -> dict[str, Any]:
    """Score Task 3: Positional Explanation (Option A).

    Two binary criteria:
    - Point 1: Correct side identification
    - Point 2: Theme mention

    Args:
        explanation: Model's explanation text
        side_claimed: Side the model claims is better
        stockfish_eval: Stockfish evaluation for ground truth
        theme: Expected theme tag

    Returns:
        Dictionary with T3 scoring results
    """
    if explanation is None:
        return {
            "t3_explanation": None,
            "t3_side_claimed": side_claimed,
            "t3_p1_side_correct": None,
            "t3_p2_theme_correct": None,
            "t3_score": None,
        }

    # Point 1: Side identification
    ground_truth_side = get_direction(stockfish_eval)
    p1 = 1 if side_claimed == ground_truth_side else 0

    # Point 2: Theme identification
    p2 = 0
    explanation_lower = explanation.lower()

    # Get synonyms for the theme
    synonyms = THEME_SYNONYMS.get(theme, [theme])
    if isinstance(theme, str):
        synonyms = synonyms + [theme.lower().replace("_", " ")]

    for synonym in synonyms:
        if synonym.lower() in explanation_lower:
            p2 = 1
            break

    return {
        "t3_explanation": explanation,
        "t3_side_claimed": side_claimed,
        "t3_p1_side_correct": p1,
        "t3_p2_theme_correct": p2,
        "t3_score": p1 + p2,
    }


def score_all(
    parsed_response: dict[str, Any],
    position: dict[str, Any],
    engine=None,
    eval_range: tuple[int, int] = (-2000, 2000),
) -> dict[str, Any]:
    """Score all three tasks for a position.

    Args:
        parsed_response: Parsed LLM response with eval, move, explanation
        position: Position dictionary with fen, stockfish_eval, stockfish_best_move, theme
        engine: Optional Stockfish engine for CPL calculation
        eval_range: Range to clamp model evaluation

    Returns:
        Combined scoring results for T1, T2, T3
    """
    results = {}

    # T1 scoring
    t1_results = score_t1(
        model_eval=parsed_response.get("eval"),
        stockfish_eval=position.get("stockfish_eval", 0),
        eval_range=eval_range,
    )
    results.update(t1_results)

    # T2 scoring
    t2_results = score_t2(
        model_move=parsed_response.get("move"),
        fen=position["fen"],
        stockfish_best_move=position.get("stockfish_best_move", ""),
        stockfish_eval=position.get("stockfish_eval", 0),
        engine=engine,
    )
    results.update(t2_results)

    # T3 scoring
    t3_results = score_t3(
        explanation=parsed_response.get("explanation"),
        side_claimed=parsed_response.get("side_claimed"),
        stockfish_eval=position.get("stockfish_eval", 0),
        theme=position.get("theme", ""),
    )
    results.update(t3_results)

    return results


def should_trigger_correction(
    t2_cpl: int | None,
    threshold: int = 50,
) -> bool:
    """Determine if a correction loop should be triggered.

    Args:
        t2_cpl: Centipawn loss from T2 scoring
        threshold: CPL threshold for triggering correction

    Returns:
        True if correction should be triggered
    """
    if t2_cpl is None:
        return False
    return t2_cpl > threshold
