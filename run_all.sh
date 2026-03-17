#!/bin/bash
set -e

cd /home/rabrew/Desktop/chess-llm-bench

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color
BOLD='\033[1m'

print_header() {
    echo ""
    echo -e "${BLUE}════════════════════════════════════════════════════════════════${NC}"
    echo -e "${BOLD}${CYAN}  $1${NC}"
    echo -e "${BLUE}════════════════════════════════════════════════════════════════${NC}"
    echo ""
}

print_step() {
    echo -e "${GREEN}▶${NC} ${BOLD}$1${NC}"
}

print_info() {
    echo -e "  ${YELLOW}→${NC} $1"
}

print_header "CHESS LLM BENCHMARK"
echo -e "  Started: $(date)"
echo -e "  Host:    $(hostname)"
echo -e "  CPUs:    $(nproc) cores"
echo ""

# Create venv if it doesn't exist
if [ ! -d "venv" ]; then
    print_step "[0/7] Creating virtual environment..."
    python3 -m venv venv
fi

# Activate venv
source venv/bin/activate

print_step "[1/7] Installing dependencies..."
pip install -r requirements.txt -q
print_info "Done"

print_header "STEP 2/7: BUILDING DATASET"
# Skip if dataset already exists with stockfish evals
if python3 -c "import json; d=json.load(open('data/easy.json')); exit(0 if 'stockfish_eval' in d[0] else 1)" 2>/dev/null; then
    print_info "Dataset with Stockfish evals already exists - SKIPPING rebuild"
    print_info "Delete data/*.json to force rebuild"
else
    print_info "Loading 5.8M puzzles from Lichess database..."
    print_info "This will validate positions using $(($(nproc))) CPU cores"
    echo ""
    python scripts/build_dataset.py
fi

print_header "STEP 3/7: LC0 GPU EVALUATION"
print_info "Using Lc0 ONNX batch inference (RTX 5080)"
print_info "Speed: ~3,000 pos/sec | Est. time: ~32 min for 5.8M positions"
echo ""
python scripts/precompute_lc0_batch.py --batch-size 256

print_header "STEP 4/7: PULLING OLLAMA MODELS"
print_info "Downloading LLM models for benchmarking..."
echo ""
python scripts/pull_models.py

print_header "STEP 5/7: GENERATING JOBS"
print_info "Creating benchmark job queue..."
echo ""
python scripts/generate_jobs.py

print_header "STEP 6/7: RUNNING BENCHMARK"
print_info "Testing LLMs against chess positions..."
print_info "Workers: $(grep -oP 'count:\s*\K\d+' config/config.yaml || echo '4')"
echo ""
python scripts/run_workers.py --workers 4

print_header "STEP 7/7: GENERATING RESULTS"
print_info "Creating plots and metrics..."
echo ""
python scripts/generate_plots.py --save-metrics

print_header "COMPLETE!"
echo -e "  Finished: $(date)"
echo ""
echo -e "  ${GREEN}Results:${NC}  results/evaluations.jsonl"
echo -e "  ${GREEN}Plots:${NC}    results/plots/"
echo -e "  ${GREEN}Metrics:${NC}  results/metrics/"
echo ""
echo -e "${BLUE}════════════════════════════════════════════════════════════════${NC}"
