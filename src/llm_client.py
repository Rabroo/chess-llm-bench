"""Ollama LLM client for chess position analysis."""

import logging
import re
import time
from typing import Any

import requests

logger = logging.getLogger("chess_llm_bench")


class OllamaClient:
    """HTTP client for Ollama API."""

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        timeout: int = 180,
        max_retries: int = 3,
    ):
        """Initialize Ollama client.

        Args:
            base_url: Ollama server URL
            timeout: Request timeout in seconds
            max_retries: Maximum retry attempts on failure
        """
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries

    def is_available(self) -> bool:
        """Check if Ollama server is running and accessible."""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except requests.RequestException:
            return False

    def list_models(self) -> list[str]:
        """List available models on the Ollama server.

        Returns:
            List of model tags
        """
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=10)
            response.raise_for_status()
            data = response.json()
            return [model["name"] for model in data.get("models", [])]
        except requests.RequestException as e:
            logger.error(f"Failed to list models: {e}")
            return []

    def pull_model(self, model: str) -> bool:
        """Pull a model from Ollama registry.

        Args:
            model: Model tag to pull

        Returns:
            True if successful, False otherwise
        """
        try:
            response = requests.post(
                f"{self.base_url}/api/pull",
                json={"name": model},
                timeout=3600,  # Long timeout for model downloads
                stream=True,
            )
            for line in response.iter_lines():
                if line:
                    logger.debug(line.decode())
            return response.status_code == 200
        except requests.RequestException as e:
            logger.error(f"Failed to pull model {model}: {e}")
            return False

    def chat(self, model: str, prompt: str) -> dict[str, Any]:
        """Send a chat request to Ollama.

        Args:
            model: Model tag to use
            prompt: User prompt

        Returns:
            Dictionary with 'response' (raw text), 'inference_ms' (duration),
            and 'success' (boolean)
        """
        start_time = time.time()
        last_error = None

        for attempt in range(self.max_retries):
            try:
                response = requests.post(
                    f"{self.base_url}/api/chat",
                    json={
                        "model": model,
                        "messages": [{"role": "user", "content": prompt}],
                        "stream": False,
                    },
                    timeout=self.timeout,
                )
                response.raise_for_status()
                data = response.json()

                inference_ms = int((time.time() - start_time) * 1000)

                return {
                    "response": data.get("message", {}).get("content", ""),
                    "inference_ms": inference_ms,
                    "success": True,
                    "model": model,
                }

            except requests.Timeout:
                last_error = "Request timed out"
                logger.warning(f"Attempt {attempt + 1}/{self.max_retries} timed out")
            except requests.RequestException as e:
                last_error = str(e)
                logger.warning(f"Attempt {attempt + 1}/{self.max_retries} failed: {e}")

            if attempt < self.max_retries - 1:
                time.sleep(2 ** attempt)  # Exponential backoff

        inference_ms = int((time.time() - start_time) * 1000)
        return {
            "response": "",
            "inference_ms": inference_ms,
            "success": False,
            "error": last_error,
            "model": model,
        }


def build_prompt(
    fen: str,
    pgn_moves: str | None = None,
    prompt_format: str = "pgn+fen",
) -> str:
    """Build the standard three-task prompt for a chess position.

    Args:
        fen: FEN string of the position
        pgn_moves: PGN move history (optional)
        prompt_format: One of "fen_only", "pgn+fen", or "cot"

    Returns:
        Formatted prompt string
    """
    # Base prompt parts
    intro = "You are analysing a chess position."

    moves_section = ""
    if prompt_format in ("pgn+fen", "cot") and pgn_moves:
        moves_section = f"\nMoves played so far:\n{pgn_moves}\n"

    fen_section = f"\nCurrent position (FEN):\n{fen}"

    questions = """
Answer all three questions below.

1. What is the centipawn evaluation of this position from White's perspective?
   A positive number means White is better. A negative number means Black is better.
   0 means equal. Give only a number, no explanation.

2. What is the best move for the side to play? Give only the move in SAN notation.

3. Who stands better in this position — White, Black, or is it equal?
   Give a one-sentence explanation of the key reason."""

    cot_section = ""
    if prompt_format == "cot":
        cot_section = """

Think step by step before answering:
- What are the key features of this position?
- Who controls more space? Who has more active pieces?
- What is the most forcing move available?"""

    response_format = """

Respond using this exact format:
Eval: <integer>
Move: <SAN move>
Explanation: <White / Black / Equal> — <one sentence reason>"""

    return intro + moves_section + fen_section + questions + cot_section + response_format


def parse_response(response_text: str) -> dict[str, Any]:
    """Parse the LLM response to extract Eval, Move, and Explanation.

    Args:
        response_text: Raw response text from the LLM

    Returns:
        Dictionary with 'eval', 'move', 'explanation', 'side_claimed',
        and 'parse_errors' list
    """
    result = {
        "eval": None,
        "move": None,
        "explanation": None,
        "side_claimed": None,
        "parse_errors": [],
    }

    lines = response_text.strip().split("\n")

    for line in lines:
        line = line.strip()

        # Parse Eval
        if line.lower().startswith("eval:"):
            eval_str = line[5:].strip()
            # Extract integer from the string
            match = re.search(r"-?\d+", eval_str)
            if match:
                try:
                    result["eval"] = int(match.group())
                except ValueError:
                    result["parse_errors"].append(f"Invalid eval value: {eval_str}")
            else:
                result["parse_errors"].append(f"Could not parse eval: {eval_str}")

        # Parse Move
        elif line.lower().startswith("move:"):
            move_str = line[5:].strip()
            # Clean up common formatting issues
            move_str = move_str.split()[0] if move_str.split() else ""
            move_str = move_str.rstrip(".,;")
            if move_str:
                result["move"] = move_str
            else:
                result["parse_errors"].append("Empty move field")

        # Parse Explanation
        elif line.lower().startswith("explanation:"):
            explanation = line[12:].strip()
            result["explanation"] = explanation

            # Extract side claimed
            explanation_lower = explanation.lower()
            if explanation_lower.startswith("white"):
                result["side_claimed"] = "White"
            elif explanation_lower.startswith("black"):
                result["side_claimed"] = "Black"
            elif explanation_lower.startswith("equal"):
                result["side_claimed"] = "Equal"
            else:
                # Try to find side mention in the explanation
                if "white" in explanation_lower and "black" not in explanation_lower:
                    result["side_claimed"] = "White"
                elif "black" in explanation_lower and "white" not in explanation_lower:
                    result["side_claimed"] = "Black"
                elif "equal" in explanation_lower or "draw" in explanation_lower:
                    result["side_claimed"] = "Equal"

    # Check for missing fields
    if result["eval"] is None:
        result["parse_errors"].append("Missing Eval field")
    if result["move"] is None:
        result["parse_errors"].append("Missing Move field")
    if result["explanation"] is None:
        result["parse_errors"].append("Missing Explanation field")

    return result
