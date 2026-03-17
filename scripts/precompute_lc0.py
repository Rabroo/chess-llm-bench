#!/usr/bin/env python3
"""Pre-compute Lc0 evaluations for all positions (GPU-accelerated, parallel)."""

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock

from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.engine_wrapper import Lc0Engine
from src.utils import load_config, setup_logging

# Global lock for file writing
file_lock = Lock()


def create_engine(nodes):
    """Create an Lc0 engine instance."""
    return Lc0Engine(
        path="/home/rabrew/lc0-src/build/release/lc0",
        weights="/home/rabrew/lc0-nets/network.pb",
        nodes=nodes,
        backend="cuda-auto",
    )


def evaluate_position(args):
    """Evaluate a single position using the provided engine."""
    engine, fen = args
    try:
        result = engine.evaluate(fen)
        return fen, result["eval"], result["best_move"], None
    except Exception as e:
        return fen, None, None, str(e)


def main():
    parser = argparse.ArgumentParser(
        description="Pre-compute Lc0 (GPU) evaluations for all dataset positions"
    )
    parser.add_argument(
        "--config",
        default="config/config.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--data-dir",
        default="data",
        help="Directory containing dataset JSON files",
    )
    parser.add_argument(
        "--nodes",
        type=int,
        default=400,
        help="Nodes per position (higher = more accurate, slower)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of parallel Lc0 instances",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose output",
    )

    args = parser.parse_args()

    # Setup logging
    import logging
    level = logging.DEBUG if args.verbose else logging.INFO
    setup_logging(level=level)
    logger = logging.getLogger("chess_llm_bench")

    # Collect all positions that need evaluation
    data_dir = Path(args.data_dir)
    tiers = ["easy", "medium", "hard", "extreme"]

    tier_data = {}
    positions_to_eval = []

    for tier in tiers:
        file_path = data_dir / f"{tier}.json"
        if not file_path.exists():
            logger.warning(f"Dataset file not found: {file_path}")
            continue

        with open(file_path, "r") as f:
            positions = json.load(f)

        tier_data[tier] = positions

        for i, pos in enumerate(positions):
            if "stockfish_eval" not in pos or "stockfish_best_move" not in pos:
                positions_to_eval.append((tier, i, pos["fen"]))

    if not positions_to_eval:
        print("All positions already have evaluations.")
        return

    print(f"\n{'='*60}")
    print(f"  LC0 GPU EVALUATION (PARALLEL)")
    print(f"  Positions: {len(positions_to_eval):,}")
    print(f"  Nodes: {args.nodes} per position")
    print(f"  Workers: {args.workers} parallel Lc0 instances")
    print(f"  Backend: cuda-auto (RTX 5080)")
    print(f"{'='*60}\n")

    # Create engine pool
    print(f"Starting {args.workers} Lc0 engines...")
    engines = []
    for i in range(args.workers):
        try:
            engine = create_engine(args.nodes)
            engines.append(engine)
            print(f"  Engine {i+1}/{args.workers} ready")
        except Exception as e:
            logger.error(f"Failed to start engine {i+1}: {e}")
            # Clean up already started engines
            for eng in engines:
                eng.close()
            sys.exit(1)

    # Checkpoint settings
    CHECKPOINT_EVERY = 1000
    last_checkpoint = 0
    last_checkpoint_time = time.time()
    CHECKPOINT_MINUTES = 30

    results = {}
    evaluated = 0
    failed = 0

    def save_checkpoint():
        """Save current progress."""
        with file_lock:
            for tier, idx, fen in positions_to_eval:
                if fen in results:
                    eval_score, best_move = results[fen]
                    tier_data[tier][idx]["stockfish_eval"] = eval_score
                    tier_data[tier][idx]["stockfish_best_move"] = best_move

            for t, positions in tier_data.items():
                file_path = data_dir / f"{t}.json"
                with open(file_path, "w") as f:
                    json.dump(positions, f)

    try:
        # Round-robin assignment to engines
        work_items = []
        for i, (tier, idx, fen) in enumerate(positions_to_eval):
            engine = engines[i % len(engines)]
            work_items.append((engine, fen, tier, idx))

        pbar = tqdm(total=len(work_items), desc="Evaluating", unit="pos")

        # Process with thread pool (engines handle their own work)
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            # Submit all work
            future_to_info = {}
            for engine, fen, tier, idx in work_items:
                future = executor.submit(evaluate_position, (engine, fen))
                future_to_info[future] = (tier, idx, fen)

            for future in as_completed(future_to_info):
                tier, idx, fen = future_to_info[future]
                fen_result, eval_score, best_move, error = future.result()

                if eval_score is not None:
                    results[fen] = (eval_score, best_move)
                    evaluated += 1
                else:
                    failed += 1
                    if error:
                        logger.debug(f"Failed {fen}: {error}")

                pbar.update(1)
                pbar.set_postfix({"done": evaluated, "failed": failed})

                # Checkpoint
                time_since = (time.time() - last_checkpoint_time) / 60
                if evaluated - last_checkpoint >= CHECKPOINT_EVERY or time_since >= CHECKPOINT_MINUTES:
                    pbar.set_description("Saving checkpoint")
                    save_checkpoint()
                    last_checkpoint = evaluated
                    last_checkpoint_time = time.time()
                    logger.info(f"Checkpoint saved: {evaluated:,} positions")
                    pbar.set_description("Evaluating")

        pbar.close()

    except KeyboardInterrupt:
        print("\n\nInterrupted! Saving progress...")
    finally:
        # Final save
        save_checkpoint()
        for tier in tier_data:
            logger.info(f"Saved {data_dir / f'{tier}.json'}")

        # Close all engines
        for engine in engines:
            engine.close()

    print(f"\nEvaluated {evaluated:,} positions ({failed} failed)")


if __name__ == "__main__":
    main()
