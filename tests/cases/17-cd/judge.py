import json
import os
import re
import sys


def judge():
    # 读取测试数据
    input_data = json.load(sys.stdin)
    stdout = input_data["stdout"]
    max_score = input_data["max_score"]

    prompt = "> "
    success = True
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

    # 提取 pwd 命令的输出
    pwd_outputs = []
    
    for cmd, output in commands_with_output:
        if cmd.strip() == "pwd":
            pwd_outputs.append(output.strip())

    # 检查是否有至少两个不同的目录输出（初始目录和 cd 后的目录）
    success = False
    message = "Failed to verify directory change"
    score = 0

    if len(pwd_outputs) >= 2:
        # 检查第一个和第二个目录是否不同
        if pwd_outputs[0] != pwd_outputs[1]:
            # 检查第二个目录是否是第一个目录的父目录
            try:
                first_dir = os.path.normpath(pwd_outputs[0])
                second_dir = os.path.normpath(pwd_outputs[1])
                
                # 支持中文路径，确保正确处理路径分隔符
                parent_dir = os.path.normpath(os.path.dirname(first_dir))

                if second_dir == parent_dir:
                    success = True
                    message = "Directory change verified successfully"
                    score = max_score
                else:
                    message = f"Second directory ({second_dir}) is not the parent of first directory ({first_dir})"
                    score = max_score * 0.5  # 部分得分，因为目录确实改变了，但不是预期的父目录
            except Exception as e:
                message = f"Error comparing directories: {str(e)}"
        else:
            message = "Directory did not change after cd command"
    else:
        message = f"Could not find enough pwd outputs (found {len(pwd_outputs)})"

    # 返回结果
    result = {"success": success, "message": message, "score": score}

    print(json.dumps(result))


if __name__ == "__main__":
    judge()
