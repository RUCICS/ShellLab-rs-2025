import json
import sys


def judge():
    # 读取测试数据
    input_data = json.load(sys.stdin)
    stdout = input_data["stdout"]
    max_score = input_data["max_score"]

    prompt = "> "
    success = True
    message_parts = []
    score = 0

    # 分析输出，查找命令和它们的输出
    commands_with_output = []

    # 使用提示符分割输出
    lines = stdout.split("\n")
    prompt_line_indices = []

    # 找出所有提示符所在行的索引
    for i, line in enumerate(lines):
        if line.startswith(prompt.strip()):
            prompt_line_indices.append(i)

    # 解析每个命令和它的输出
    for i, prompt_index in enumerate(prompt_line_indices):
        # 获取命令
        cmd_line = lines[prompt_index]
        cmd = cmd_line[len(prompt.strip()) :].strip()

        # 获取输出: 从命令行的下一行开始，到下一个提示符行（或文件结束）
        output_start = prompt_index + 1
        output_end = (
            prompt_line_indices[i + 1]
            if i + 1 < len(prompt_line_indices)
            else len(lines)
        )

        # 如果没有输出行或者下一行就是提示符，则输出为空
        if output_start >= output_end:
            output = ""
        else:
            output = "\n".join(lines[output_start:output_end])

        commands_with_output.append((cmd, output))

    # 检查重定向测试
    output_redirection_success = False
    input_redirection_success = False

    for cmd, output in commands_with_output:
        # 检查输出重定向测试
        if "cat output.txt" in cmd:
            # 输出应该包含 "Hello, World!"
            if "Hello, World!" in output:
                output_redirection_success = True

        # 检查输入重定向测试
        if "cat < input.txt" in cmd:
            # 输出应该包含 "Test input redirection"
            if "Test input redirection" in output:
                input_redirection_success = True

    # 评估测试结果
    if output_redirection_success:
        message_parts.append("Output redirection test passed")
    else:
        success = False
        message_parts.append(
            "Output redirection test failed: Failed to correctly write to or read from output.txt"
        )

    if input_redirection_success:
        message_parts.append("Input redirection test passed")
    else:
        success = False
        message_parts.append(
            "Input redirection test failed: Failed to correctly read input from input.txt"
        )

    message = "; ".join(message_parts)

    # 计算得分
    if success:
        score = max_score
    else:
        # 根据通过的测试计算部分分数
        passed_tests = (1 if output_redirection_success else 0) + (
            1 if input_redirection_success else 0
        )
        total_tests = 2
        score = max_score * (passed_tests / total_tests)

    # 返回结果
    result = {"success": success, "message": message, "score": score}
    print(json.dumps(result))


if __name__ == "__main__":
    judge()
