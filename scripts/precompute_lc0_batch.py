#!/usr/bin/env python3
"""Ultra-fast Lc0 batch evaluation using ONNX Runtime GPU."""

import argparse
import json
import sys
from pathlib import Path
import numpy as np
import chess

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.utils import setup_logging

import onnxruntime as ort
from tqdm import tqdm


def encode_position(board: chess.Board) -> np.ndarray:
    """Encode a chess position to Lc0's 112-plane input format.

    The format is:
    - Planes 0-5: our pieces (P, N, B, R, Q, K)
    - Planes 6-11: their pieces
    - Planes 12-103: 7 previous positions (we use current position repeated)
    - Planes 104-111: auxiliary (castling, en passant, etc.)
    """
    planes = np.zeros((112, 8, 8), dtype=np.float32)

    # Determine perspective (always from side to move's POV for Lc0)
    flip = board.turn == chess.BLACK

    def sq_to_plane_idx(sq):
        """Convert square to plane index, flipping if black to move."""
        rank = chess.square_rank(sq)
        file = chess.square_file(sq)
        if flip:
            rank = 7 - rank
        return rank, file

    # Piece type to plane offset
    piece_to_plane = {
        chess.PAWN: 0, chess.KNIGHT: 1, chess.BISHOP: 2,
        chess.ROOK: 3, chess.QUEEN: 4, chess.KING: 5
    }

    # Fill piece planes for current position (and repeat for history)
    for history_idx in range(8):
        base = history_idx * 13

        for sq in chess.SQUARES:
            piece = board.piece_at(sq)
            if piece is None:
                continue

            rank, file = sq_to_plane_idx(sq)

            # Is this our piece or theirs?
            is_ours = (piece.color == board.turn)
            plane_offset = 0 if is_ours else 6

            plane_idx = base + plane_offset + piece_to_plane[piece.piece_type]
            planes[plane_idx, rank, file] = 1.0

        # Repetition plane (plane 12 + history_idx * 13)
        # We're not tracking repetitions, so leave at 0

    # Auxiliary planes (104-111)
    # Plane 104: castling us kingside
    # Plane 105: castling us queenside
    # Plane 106: castling them kingside
    # Plane 107: castling them queenside
    # Plane 108: side to move (all 1s if we are white from our POV, but since we flip, always 1)
    # Plane 109: 50-move counter
    # Plane 110: zeros
    # Plane 111: ones

    if board.turn == chess.WHITE:
        if board.has_kingside_castling_rights(chess.WHITE):
            planes[104, :, :] = 1.0
        if board.has_queenside_castling_rights(chess.WHITE):
            planes[105, :, :] = 1.0
        if board.has_kingside_castling_rights(chess.BLACK):
            planes[106, :, :] = 1.0
        if board.has_queenside_castling_rights(chess.BLACK):
            planes[107, :, :] = 1.0
    else:
        if board.has_kingside_castling_rights(chess.BLACK):
            planes[104, :, :] = 1.0
        if board.has_queenside_castling_rights(chess.BLACK):
            planes[105, :, :] = 1.0
        if board.has_kingside_castling_rights(chess.WHITE):
            planes[106, :, :] = 1.0
        if board.has_queenside_castling_rights(chess.WHITE):
            planes[107, :, :] = 1.0

    planes[108, :, :] = 1.0  # Side to move (always 1 from our perspective)
    planes[109, :, :] = board.halfmove_clock / 100.0  # 50-move counter normalized
    # Plane 110 stays zeros
    planes[111, :, :] = 1.0  # All ones

    return planes


def wdl_to_centipawns(wdl: np.ndarray) -> int:
    """Convert WDL (win/draw/loss) probabilities to centipawn evaluation."""
    w, d, l = wdl[0], wdl[1], wdl[2]
    # Expected score
    score = w + d * 0.5
    # Convert to centipawns (logit-like transformation)
    if score >= 0.999:
        return 10000
    elif score <= 0.001:
        return -10000
    else:
        # Approximate centipawn conversion
        cp = int(111.714 * np.tan(1.5620688 * (score - 0.5)))
        return max(-10000, min(10000, cp))


def decode_policy_to_move(policy: np.ndarray, board: chess.Board) -> str:
    """Decode policy output to best legal move."""
    # Lc0 policy encoding is complex - we'll just return None for now
    # The eval is what matters most
    return None


def main():
    parser = argparse.ArgumentParser(description="Fast batch Lc0 evaluation")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    import logging
    setup_logging(level=logging.DEBUG if args.verbose else logging.INFO)
    logger = logging.getLogger("chess_llm_bench")

    # Load ONNX model
    print("Loading Lc0 ONNX model...")
    sess = ort.InferenceSession(
        "/home/rabrew/lc0-nets/network.onnx",
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
    )
    print(f"Using providers: {sess.get_providers()}")

    # Load positions
    data_dir = Path(args.data_dir)
    tiers = ["easy", "medium", "hard", "extreme"]

    tier_data = {}
    positions_to_eval = []

    for tier in tiers:
        file_path = data_dir / f"{tier}.json"
        if not file_path.exists():
            continue
        with open(file_path, "r") as f:
            positions = json.load(f)
        tier_data[tier] = positions
        for i, pos in enumerate(positions):
            if "stockfish_eval" not in pos:
                positions_to_eval.append((tier, i, pos["fen"]))

    if not positions_to_eval:
        print("All positions already evaluated.")
        return

    print(f"\n{'='*60}")
    print(f"  LC0 BATCH GPU EVALUATION")
    print(f"  Positions: {len(positions_to_eval):,}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Expected: ~3000 pos/sec")
    print(f"{'='*60}\n")

    # Process in batches
    CHECKPOINT_EVERY = 50000
    last_checkpoint = 0
    evaluated = 0
    failed = 0

    def save_checkpoint():
        for tier, positions in tier_data.items():
            with open(data_dir / f"{tier}.json", "w") as f:
                json.dump(positions, f)
        logger.info(f"Checkpoint: {evaluated:,} positions")

    try:
        pbar = tqdm(total=len(positions_to_eval), desc="Evaluating", unit="pos")

        batch_data = []
        batch_meta = []  # (tier, idx, board, fen)

        for tier, idx, fen in positions_to_eval:
            try:
                board = chess.Board(fen)
                encoded = encode_position(board)
                batch_data.append(encoded)
                batch_meta.append((tier, idx, board, fen))
            except Exception as e:
                failed += 1
                pbar.update(1)
                continue

            # Process batch when full
            if len(batch_data) >= args.batch_size:
                # Run inference
                batch_input = np.stack(batch_data, axis=0)
                outputs = sess.run(None, {"/input/planes": batch_input})
                wdl_batch = outputs[1]  # WDL output

                # Store results
                for i, (t, idx, board, fen) in enumerate(batch_meta):
                    wdl = wdl_batch[i]
                    cp = wdl_to_centipawns(wdl)

                    # Adjust for side to move (Lc0 outputs from side-to-move perspective)
                    if board.turn == chess.BLACK:
                        cp = -cp

                    tier_data[t][idx]["stockfish_eval"] = cp
                    tier_data[t][idx]["stockfish_best_move"] = None  # Not decoding policy
                    evaluated += 1

                pbar.update(len(batch_data))
                pbar.set_postfix({"done": evaluated, "failed": failed})

                batch_data = []
                batch_meta = []

                # Checkpoint
                if evaluated - last_checkpoint >= CHECKPOINT_EVERY:
                    save_checkpoint()
                    last_checkpoint = evaluated

        # Process remaining
        if batch_data:
            batch_input = np.stack(batch_data, axis=0)
            outputs = sess.run(None, {"/input/planes": batch_input})
            wdl_batch = outputs[1]

            for i, (t, idx, board, fen) in enumerate(batch_meta):
                wdl = wdl_batch[i]
                cp = wdl_to_centipawns(wdl)
                if board.turn == chess.BLACK:
                    cp = -cp
                tier_data[t][idx]["stockfish_eval"] = cp
                tier_data[t][idx]["stockfish_best_move"] = None
                evaluated += 1

            pbar.update(len(batch_data))

        pbar.close()

    except KeyboardInterrupt:
        print("\nInterrupted!")
    finally:
        save_checkpoint()

    print(f"\nDone: {evaluated:,} evaluated, {failed} failed")


if __name__ == "__main__":
    main()
