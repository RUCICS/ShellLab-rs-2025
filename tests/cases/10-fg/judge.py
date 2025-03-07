#!/usr/bin/env python3
import json
import re
import sys


def judge():
    # Read input data
    input_data = json.load(sys.stdin)
    stdout = input_data["stdout"]
    max_score = input_data["max_score"]

    # Print the output for debugging
    # print(f"Received output:\n{stdout}", file=sys.stderr)

    # Define the key events we need to check
    events = [
        {
            "name": "Background job creation",
            "pattern": r"\[1\]\s+\(\d+\)\s+\./myspin 4 &",
            "required": True,
        },
        {
            "name": "Job stopped by signal",
            "pattern": r"Job \[1\]\s+\(\d+\)\s+stopped by signal (20|SIGTSTP)",
            "required": True,
        },
        {
            "name": "Job listed as stopped",
            "pattern": r"\[1\]\s+\(\d+\)\s+Stopped \./myspin 4 &",
            "required": True,
        },
    ]

    # Check each event
    messages = []
    passed_all_required = True

    for event in events:
        match = re.search(event["pattern"], stdout, re.MULTILINE)
        if match:
            messages.append(f"✓ {event['name']}: Found")
        else:
            messages.append(f"✗ {event['name']}: Not found")
            if event["required"]:
                passed_all_required = False

    # Check for correct event sequence
    sequence_correct = True

    # Find positions of key events
    bg_job_pos = stdout.find("./myspin 4 &")
    fg_cmd_pos = stdout.find("fg %1", bg_job_pos if bg_job_pos >= 0 else 0)
    stopped_pos = stdout.find("stopped by signal", fg_cmd_pos if fg_cmd_pos >= 0 else 0)
    jobs_after_stop_pos = stdout.find("jobs", stopped_pos if stopped_pos >= 0 else 0)
    stopped_job_pos = stdout.find(
        "Stopped", jobs_after_stop_pos if jobs_after_stop_pos >= 0 else 0
    )
    final_fg_pos = stdout.find("fg %1", stopped_job_pos if stopped_job_pos >= 0 else 0)
    final_jobs_pos = stdout.find("jobs", final_fg_pos if final_fg_pos >= 0 else 0)

    # Check if events occurred in correct order
    if not all(
        pos >= 0
        for pos in [
            bg_job_pos,
            fg_cmd_pos,
            stopped_pos,
            jobs_after_stop_pos,
            stopped_job_pos,
            final_fg_pos,
            final_jobs_pos,
        ]
    ):
        sequence_correct = False

    if sequence_correct:
        messages.append("✓ Overall command sequence is correct")
    else:
        messages.append("✗ Overall command sequence is incorrect")
        passed_all_required = False

    # Determine success based on required events
    success = passed_all_required

    # Build the result message
    message = "\n".join(messages)

    # Create the result
    result = {
        "success": success,
        "message": message,
        "score": max_score if success else 0,
    }

    # Output the result
    print(json.dumps(result))


if __name__ == "__main__":
    judge()
