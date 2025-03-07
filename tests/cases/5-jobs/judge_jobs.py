#!/usr/bin/env python3
import json
import re
import sys


def check_jobs_output(stdout):
    """
    Check if the jobs output correctly shows two background jobs:
    - myspin 2 & (job ID 1)
    - myspin 3 & (job ID 2)
    Both should be marked as Running
    """
    lines = stdout.strip().split("\n")

    # Look for job output lines in the last part of output (after 'jobs' command)
    jobs_section = False
    job_lines = []

    for line in lines:
        if line.strip() == "jobs":
            jobs_section = True
            continue
        if jobs_section and line.strip():
            job_lines.append(line)

    # Check if we have exactly two job entries
    if len(job_lines) != 2:
        return False, f"Expected 2 job entries, found {len(job_lines)}"

    # Parse job entries
    job1_found = False
    job2_found = False

    for line in job_lines:
        # Match format like "[1] (26256) Running ./myspin 2 &"
        match = re.match(r"\[(\d+)\]\s+\(\d+\)\s+(Running)\s+(./myspin\s+\d+.*)", line)
        if not match:
            continue

        job_id = int(match.group(1))
        status = match.group(2)
        command = match.group(3)

        if job_id == 1 and "myspin 2" in command and status == "Running":
            job1_found = True
        elif job_id == 2 and "myspin 3" in command and status == "Running":
            job2_found = True

    if job1_found and job2_found:
        return True, "Both jobs correctly listed with proper status"

    missing = []
    if not job1_found:
        missing.append("Job 1 (myspin 2)")
    if not job2_found:
        missing.append("Job 2 (myspin 3)")

    return False, f"Missing or incorrect entries for: {', '.join(missing)}"


def main():
    # Read input from stdin (passed by grader.py)
    input_data = json.load(sys.stdin)
    stdout = input_data["stdout"]
    max_score = input_data["max_score"]

    success, message = check_jobs_output(stdout)

    result = {
        "success": success,
        "message": message,
        "score": max_score if success else 0,
    }

    print(json.dumps(result))


if __name__ == "__main__":
    main()
