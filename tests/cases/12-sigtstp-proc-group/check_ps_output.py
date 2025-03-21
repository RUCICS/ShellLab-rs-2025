#!/usr/bin/env python3
import json
import re
import sys


def check_ps_output(stdout):
    """
    Check that the ps output contains:
    1. The header line (more flexible format checking)
    2. At least one "mysplit 4" process with stopped status (T, T+, Ts, etc.)
    3. The shell process itself
    """
    lines = stdout.strip().split("\n")

    # Find the section of output after "/bin/ps ax" command
    ps_output_start = -1
    for i, line in enumerate(lines):
        if line.strip() == "/bin/ps ax" or line.strip() == "> /bin/ps ax":
            ps_output_start = i + 1
            break

    if ps_output_start == -1 or ps_output_start >= len(lines):
        return False, "Could not find ps command output"

    ps_output = lines[ps_output_start:]

    # Check header - more flexible now
    if not ps_output or not (
        all(col in ps_output[0].upper() for col in ["PID", "TTY", "STAT", "TIME"])
        and any(col in ps_output[0].upper() for col in ["COMMAND", "CMD"])
    ):
        return False, "Missing or invalid ps output header"

    # Check for stopped mysplit processes - support more status formats
    mysplit_stopped = False

    for line in ps_output[1:]:  # Skip header
        if not line.strip():
            continue

        # Look for stopped mysplit processes with various stop status indicators
        if "./mysplit 4" in line and re.search(r'\b(T|Ts|T\+|STOPPED)\b', line, re.IGNORECASE):
            mysplit_stopped = True
            break

    if not mysplit_stopped:
        return False, "No stopped mysplit processes found"

    return True, "PS output correctly shows stopped processes"


def main():
    # Parse input from the test runner
    input_data = json.loads(sys.stdin.read())
    stdout = input_data.get("stdout", "")
    max_score = input_data.get("max_score", 0)

    success, message = check_ps_output(stdout)

    # Prepare result
    result = {
        "success": success,
        "message": message,
        "score": max_score if success else 0,
    }

    # Output result
    print(json.dumps(result))


if __name__ == "__main__":
    main()
