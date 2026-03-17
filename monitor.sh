#!/bin/bash
# Monitor script - constantly checks tmux session for errors

LOGFILE="/home/rabrew/chess-llm-bench/crash_report.txt"
SESSION="chess"
LAST_PROGRESS=""

echo "=== Chess Benchmark Monitor ===" > "$LOGFILE"
echo "Started: $(date)" >> "$LOGFILE"
echo "Checking every 10 seconds..." >> "$LOGFILE"
echo "" >> "$LOGFILE"

while true; do
    TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')

    # Check if tmux session exists
    if ! tmux has-session -t "$SESSION" 2>/dev/null; then
        echo "" >> "$LOGFILE"
        echo "========================================" >> "$LOGFILE"
        echo "[$TIMESTAMP] SESSION ENDED" >> "$LOGFILE"
        echo "========================================" >> "$LOGFILE"

        # Check if it completed successfully or crashed
        if [ -f "/home/rabrew/chess-llm-bench/results/evaluations.jsonl" ]; then
            echo "STATUS: Completed successfully (results file exists)" >> "$LOGFILE"
        else
            echo "STATUS: May have crashed (no results file yet)" >> "$LOGFILE"
        fi

        # Capture any final output
        echo "" >> "$LOGFILE"
        echo "Check tmux output with: tmux capture-pane -t chess -p -S -100" >> "$LOGFILE"
        echo "" >> "$LOGFILE"
        echo "Monitor stopping." >> "$LOGFILE"
        exit 0
    fi

    # Capture recent tmux output
    RECENT_OUTPUT=$(tmux capture-pane -t "$SESSION" -p -S -30 2>/dev/null)

    # Check for error patterns (excluding the old KeyboardInterrupt)
    CURRENT_ERRORS=$(echo "$RECENT_OUTPUT" | grep -iE "(error|exception|failed|crash|killed|oom|memory)" | grep -v "KeyboardInterrupt" | tail -3)

    if [ -n "$CURRENT_ERRORS" ]; then
        echo "" >> "$LOGFILE"
        echo "========================================" >> "$LOGFILE"
        echo "[$TIMESTAMP] ERROR DETECTED" >> "$LOGFILE"
        echo "========================================" >> "$LOGFILE"
        echo "$CURRENT_ERRORS" >> "$LOGFILE"
        echo "" >> "$LOGFILE"
        echo "Full recent output:" >> "$LOGFILE"
        echo "$RECENT_OUTPUT" >> "$LOGFILE"
        echo "========================================" >> "$LOGFILE"
    fi

    # Get current progress line
    PROGRESS_LINE=$(echo "$RECENT_OUTPUT" | grep -E "(%\||Evaluating|Validating|STEP|Processing)" | tail -1)

    # Only log if progress changed
    if [ -n "$PROGRESS_LINE" ] && [ "$PROGRESS_LINE" != "$LAST_PROGRESS" ]; then
        echo "[$TIMESTAMP] $PROGRESS_LINE" >> "$LOGFILE"
        LAST_PROGRESS="$PROGRESS_LINE"
    fi

    # Sleep 10 seconds
    sleep 10
done
