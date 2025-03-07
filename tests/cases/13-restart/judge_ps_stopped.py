#!/usr/bin/env python3
import json
import re
import sys


def judge_ps_output(stdout):
    # Convert stdout to lines for easier processing
    lines = stdout.strip().split("\n")

    # Find where the ps output starts (after "tsh> /bin/ps a")
    ps_start = -1
    for i, line in enumerate(lines):
        if line.strip() == "tsh> /bin/ps a":
            ps_start = i + 1
            break

    if ps_start == -1 or ps_start >= len(lines):
        return False, "Could not find ps output"

    # Extract ps output lines
    ps_lines = []
    for i in range(ps_start, len(lines)):
        line = lines[i].strip()
        if line.startswith("tsh>"):
            break
        if line:
            ps_lines.append(line)

    # Check for header line
    if not ps_lines or not re.match(
        r"\s*PID\s+TTY\s+STAT\s+TIME\s+COMMAND", ps_lines[0]
    ):
        return False, "ps output missing proper header"

    # Check for mysplit processes in T (stopped) state
    mysplit_stopped = False
    for line in ps_lines:
        if "mysplit" in line and " T " in line:
            mysplit_stopped = True
            break

    if not mysplit_stopped:
        return False, "No stopped mysplit process found in ps output"

    return True, "ps output shows mysplit processes properly stopped"


# Read input from stdin (provided by the grader)
input_data = json.loads(sys.stdin.read())
stdout = input_data.get("stdout", "")
max_score = input_data.get("max_score", 0)

success, message = judge_ps_output(stdout)

# Return result
result = {"success": success, "message": message, "score": max_score if success else 0}

print(json.dumps(result))
