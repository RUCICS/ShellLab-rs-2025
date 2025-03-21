import json
import re
import sys


def main():
    # 读取输入
    input_data = json.loads(sys.stdin.read())
    stdout = input_data["stdout"]
    max_score = input_data.get("max_score", 2.0)

    # 提取/bin/ps命令的输出
    # 方法：找到包含"tsh> /bin/ps ax"的行，然后获取它后面的内容直到下一个提示符
    ps_output = ""
    ps_command_pattern = r"> /bin/ps ax\s*\n"
    match = re.search(ps_command_pattern, stdout)

    if match:
        ps_start = match.end()
        # 查找下一个提示符或EOF
        next_prompt = re.search(r"tsh>", stdout[ps_start:])
        if next_prompt:
            ps_end = ps_start + next_prompt.start()
            ps_output = stdout[ps_start:ps_end].strip()
        else:
            ps_output = stdout[ps_start:].strip()

    # 如果找不到明确的ps命令输出，使用整个stdout中包含PID TTY的部分
    if not ps_output:
        ps_header_match = re.search(r"PID\s+TTY.*COMMAND", stdout)
        if ps_header_match:
            ps_start = ps_header_match.start()
            next_prompt = re.search(r"tsh>", stdout[ps_start:])
            if next_prompt:
                ps_end = ps_start + next_prompt.start()
                ps_output = stdout[ps_start:ps_end].strip()
            else:
                ps_output = stdout[ps_start:].strip()

    if not ps_output:
        # 无法找到ps命令的输出，返回错误
        result = {
            "success": False,
            "message": "Could not find ps command output",
            "score": 0.0,
        }
        print(json.dumps(result))
        return

    # 检查是否还有mysplit进程在运行中
    # 我们只关注ps命令输出中的进程状态
    mysplit_running = False

    # 分割输出行
    lines = ps_output.split("\\n")
    for line in lines:
        # 查找包含mysplit的行
        if "mysplit" in line:
            # 检查进程状态
            # 正常情况下，如果SIGINT正确传递，mysplit进程应该已经终止
            # 我们只关心它不再是"R"(运行)状态
            if re.search(r"\s+R\s+", line):
                mysplit_running = True
                break

    if mysplit_running:
        # 如果发现任何运行中的mysplit进程，说明SIGINT没有正确转发
        result = {
            "success": False,
            "message": "Found running mysplit process after SIGINT - signal was not properly forwarded",
            "score": 0.0,
        }
    else:
        # 所有mysplit进程都不再运行，测试通过
        result = {
            "success": True,
            "message": "No running mysplit processes found - SIGINT was properly forwarded",
            "score": max_score,
        }

    print(json.dumps(result))


if __name__ == "__main__":
    main()
