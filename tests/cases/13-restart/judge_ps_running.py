#!/usr/bin/env python3
import json
import re
import sys


def judge_ps_output(stdout):
    # Convert stdout to lines for easier processing
    lines = stdout.strip().split("\n")

    # Find the second instance of ps output (after fg command)
    ps_instances = []
    for i, line in enumerate(lines):
        if line.strip() == "tsh> /bin/ps a":
            ps_instances.append(i)

    if len(ps_instances) < 2:
        return False, "Could not find second ps output after fg command"

    ps_start = ps_instances[1] + 1

    if ps_start >= len(lines):
        return False, "Second ps output is missing"

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
        return False, "Second ps output missing proper header"

    # After fg, we should NOT see any mysplit processes in T (stopped) state
    # They should either be running or completed
    for line in ps_lines:
        if "mysplit" in line and " T " in line:
            return False, "mysplit process still in stopped state after fg command"

    return True, "ps output shows mysplit processes properly resumed"


# Read input from stdin (provided by the grader)
input_data = json.loads(sys.stdin.read())
stdout = input_data.get("stdout", "")
max_score = input_data.get("max_score", 0)

success, message = judge_ps_output(stdout)

# Return result
result = {"success": success, "message": message, "score": max_score if success else 0}

print(json.dumps(result))
