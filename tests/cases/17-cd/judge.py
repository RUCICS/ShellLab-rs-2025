import json
import os
import re
import sys


def judge():
    # 读取测试数据
    input_data = json.load(sys.stdin)
    stdout = input_data["stdout"]
    max_score = input_data["max_score"]

    # 查找 shell 提示符模式，以便正确识别命令输出
    prompt_pattern = r"[\w\d\-_]+[@:][\w\d\-_/]+[#$%>]"
    prompt_matches = re.findall(prompt_pattern, stdout)
    prompt = None
    if prompt_matches:
        # 使用最常见的提示符
        from collections import Counter

        prompt = Counter(prompt_matches).most_common(1)[0][0]

    # 提取 pwd 命令的输出
    pwd_outputs = []
    if prompt:
        # 查找 pwd 命令的输出
        pwd_pattern = rf"{re.escape(prompt)} pwd\s*\n(.*?)(?={re.escape(prompt)}|$)"
        pwd_matches = re.findall(pwd_pattern, stdout, re.DOTALL)
        pwd_outputs = [match.strip() for match in pwd_matches]
    else:
        # 如果找不到提示符，尝试直接提取可能的路径
        path_pattern = r"/[a-zA-Z0-9_\-./]+"
        pwd_outputs = re.findall(path_pattern, stdout)

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
                parent_dir = os.path.normpath(os.path.join(first_dir, ".."))

                if second_dir == parent_dir:
                    success = True
                    message = "Directory change verified successfully"
                    score = max_score
                else:
                    message = f"Second directory ({second_dir}) is not the parent of first directory ({first_dir})"
                    score = (
                        max_score * 0.5
                    )  # 部分得分，因为目录确实改变了，但不是预期的父目录
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
