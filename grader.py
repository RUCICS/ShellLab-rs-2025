import argparse
import difflib
import io
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
import traceback
import venv
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Tuple


def create_venv(venv_path):
    if venv_path.exists():
        shutil.rmtree(venv_path)
    print("Creating virtual environment...", flush=True)
    venv.create(venv_path, with_pip=True)
    print(f"Virtual environment will be created at: {venv_path}", flush=True)


def install_requirements(venv_path):
    pip_path = venv_path / ("Scripts" if sys.platform == "win32" else "bin") / "pip"
    requirements_path = Path(__file__).parent / "requirements.txt"
    print("Installing dependencies...", flush=True)
    subprocess.run(
        [
            str(pip_path),
            "install",
            "-r",
            str(requirements_path),
            "-i",
            "https://pypi.tuna.tsinghua.edu.cn/simple",
        ],
        check=True,
    )


def ensure_venv():
    # 首先检查本地是否已安装必需的包
    try:
        import rich
        import tomli

        return True
    except ImportError:
        pass

    venv_dir = Path(__file__).parent / ".venv"
    python_path = (
        venv_dir / ("Scripts" if sys.platform == "win32" else "bin") / "python"
    )
    pip_path = venv_dir / ("Scripts" if sys.platform == "win32" else "bin") / "pip"

    # 检查是否已在虚拟环境中运行
    if os.environ.get("GRADER_VENV"):
        try:
            install_requirements(venv_dir)
            return True
        except Exception as e:
            print(
                f"Error: Failed to set up virtual environment: {str(e)}",
                file=sys.stderr,
            )
            if venv_dir.exists():
                shutil.rmtree(venv_dir)

            sys.exit(1)

    try:
        # 如果存在虚拟环境，直接使用
        if venv_dir.exists() and python_path.exists() and pip_path.exists():
            pass
        else:
            # 创建新的虚拟环境
            create_venv(venv_dir)
            install_requirements(venv_dir)

        # 使用虚拟环境重新运行脚本
        env = os.environ.copy()
        env["GRADER_VENV"] = "1"
        subprocess.run(
            [str(python_path), __file__] + sys.argv[1:], env=env, check=False
        )
        return False

    except Exception as e:
        print(f"Error: Failed to set up virtual environment: {str(e)}", file=sys.stderr)
        # 如果虚拟环境创建失败，清理现有的虚拟环境
        if venv_dir.exists():
            shutil.rmtree(venv_dir)
        sys.exit(1)


if __name__ == "__main__":
    if not ensure_venv():
        sys.exit(0)


import tomli  # noqa: E402
from rich.console import Console  # noqa: E402
from rich.panel import Panel  # noqa: E402
from rich.progress import Progress, SpinnerColumn, TextColumn  # noqa: E402
from rich.table import Table  # noqa: E402


@dataclass
class TestResult:
    success: bool
    message: str
    time: float
    score: float
    max_score: float
    step_scores: List[Tuple[str, float, float]] = None
    error_details: Optional[List[Dict[str, Any]]] = None

    @property
    def status(self) -> str:
        if not self.success:
            return "FAIL"
        if self.score == self.max_score:
            return "PASS"
        return "PARTIAL"

    def to_dict(self):
        return {
            "success": self.success,
            "status": self.status,
            "message": self.message,
            "time": self.time,
            "score": self.score,
            "max_score": self.max_score,
            "step_scores": self.step_scores,
            "error_details": self.error_details,
        }


@dataclass
class TestCase:
    path: Path
    meta: Dict[str, Any]
    run_steps: List[Dict[str, Any]]


class Config:
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self._config = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        config_path = self.project_root / "grader_config.toml"
        if not config_path.exists():
            return {
                "paths": {
                    "tests_dir": "tests",
                    "cases_dir": "tests/cases",
                    "common_dir": "tests/common",
                },
                "debug": {
                    "default_type": "gdb",  # or "lldb", "python", "rust"
                    "show_test_build_hint": True,  # 是否在失败时显示 TEST_BUILD 环境变量设置提示
                },
            }
        with open(config_path, "rb") as f:
            return tomli.load(f)

    @property
    def paths(self) -> Dict[str, Path]:
        return {
            "tests_dir": self.project_root / self._config["paths"]["tests_dir"],
            "cases_dir": self.project_root / self._config["paths"]["cases_dir"],
            "common_dir": self.project_root / self._config["paths"]["common_dir"],
        }

    @property
    def setup_steps(self) -> List[Dict[str, Any]]:
        return self._config.get("setup", {}).get("steps", [])

    @property
    def groups(self) -> Dict[str, List[str]]:
        """获取测试组配置"""
        return self._config.get("groups", {})

    @property
    def debug_config(self) -> Dict[str, Any]:
        """Get debug configuration from config file"""
        return self._config.get(
            "debug",
            {
                "default_type": "gdb",
                "show_test_build_hint": True,
            },
        )

    @property
    def executables(self) -> Dict[str, str]:
        """获取预定义的可执行文件配置"""
        return self._config.get("executables", {})


class OutputChecker(Protocol):
    def check(
        self,
        step: Dict[str, Any],
        output: str,
        error: str,
        return_code: int,
        test_dir: Path,
    ) -> Tuple[bool, str, Optional[float]]:
        pass


class StandardOutputChecker:
    def check(
        self,
        step: Dict[str, Any],
        output: str,
        error: str,
        return_code: int,
        test_dir: Path,
    ) -> Tuple[bool, str, Optional[float]]:
        check = step.get("check", {})

        # 检查返回值
        if "return_code" in check and return_code != check["return_code"]:
            return (
                False,
                f"Expected return code {check['return_code']}, got {return_code}",
                None,
            )

        # 检查文件是否存在
        if "files" in check:
            for file_path in check["files"]:
                resolved_path = Path(self._resolve_path(file_path, test_dir))
                if not resolved_path.exists():
                    return False, f"Required file '{file_path}' not found", None

        # 检查标准输出
        if "stdout" in check:
            expect_file = test_dir / check["stdout"]
            if not expect_file.exists():
                return False, f"Expected output file {check['stdout']} not found", None
            with open(expect_file) as f:
                expected = f.read()
            if check.get("ignore_whitespace", False):
                output = " ".join(output.split())
                expected = " ".join(expected.split())
            if output.rstrip() != expected.rstrip():
                return False, "Output does not match expected content", None

        # 检查标准错误
        if "stderr" in check:
            expect_file = test_dir / check["stderr"]
            if not expect_file.exists():
                return False, f"Expected error file {check['stderr']} not found", None
            with open(expect_file) as f:
                expected = f.read()
            if check.get("ignore_whitespace", False):
                error = " ".join(error.split())
                expected = " ".join(expected.split())
            if error.rstrip() != expected.rstrip():
                return False, "Error output does not match expected content", None

        return True, "All checks passed", None

    def _resolve_path(self, path: str, test_dir: Path) -> str:
        build_dir = test_dir / "build"
        build_dir.mkdir(exist_ok=True)

        replacements = {
            "${test_dir}": str(test_dir),
            "${build_dir}": str(build_dir),
        }

        for var, value in replacements.items():
            path = path.replace(var, value)
        return path


class SpecialJudgeChecker:
    def check(
        self,
        step: Dict[str, Any],
        output: str,
        error: str,
        return_code: int,
        test_dir: Path,
    ) -> Tuple[bool, str, Optional[float]]:
        check = step.get("check", {})
        if "special_judge" not in check:
            return True, "No special judge specified", None

        judge_script = test_dir / check["special_judge"]
        if not judge_script.exists():
            return (
                False,
                f"Special judge script {check['special_judge']} not found",
                None,
            )

        input_data = {
            "stdout": output,
            "stderr": error,
            "return_code": return_code,
            "test_dir": str(test_dir),
            "max_score": step.get("score", 0),
        }

        try:
            process = subprocess.run(
                [sys.executable, str(judge_script)],
                input=json.dumps(input_data),
                capture_output=True,
                text=True,
            )
            result = json.loads(process.stdout)
            if "score" in result:
                result["score"] = min(result["score"], step.get("score", 0))
            return (
                result["success"],
                result.get("message", "No message provided"),
                result.get("score", None),
            )
        except Exception as e:
            return False, f"Special judge failed: {str(e)}", None


class PatternChecker:
    def check(
        self,
        step: Dict[str, Any],
        output: str,
        error: str,
        return_code: int,
        test_dir: Path,
    ) -> Tuple[bool, str, Optional[float]]:
        check = step.get("check", {})

        if "stdout_pattern" in check:
            if not re.search(check["stdout_pattern"], output, re.MULTILINE):
                return (
                    False,
                    f"Output does not match pattern {check['stdout_pattern']!r}",
                    None,
                )

        if "stderr_pattern" in check:
            if not re.search(check["stderr_pattern"], error, re.MULTILINE):
                return (
                    False,
                    f"Error output does not match pattern {check['stderr_pattern']!r}",
                    None,
                )

        return True, "All pattern checks passed", None


class SequenceChecker:
    """检查输出中的模式是否按指定顺序出现，支持部分无序"""

    def check(
        self,
        step: Dict[str, Any],
        output: str,
        error: str,
        return_code: int,
        test_dir: Path,
    ) -> Tuple[bool, str, Optional[float]]:
        # 检查配置中是否有序列检查相关配置
        check_config = step.get("check", {})
        if "sequence" not in check_config:
            return True, "No sequence check specified", None

        # 获取序列检查配置
        seq_config = check_config["sequence"]
        patterns = seq_config.get("patterns", [])
        case_sensitive = seq_config.get("case_sensitive", True)
        regex_mode = seq_config.get("regex_mode", True)
        allow_partial = seq_config.get("allow_partial", False)
        verify_end = seq_config.get("verify_end", False)
        whitespace_chars = seq_config.get("whitespace_chars", " \t\n\r")

        # 执行序列检查
        success, score_ratio, message = self._check_sequence(
            output,
            patterns,
            case_sensitive,
            regex_mode,
            allow_partial,
            verify_end,
            whitespace_chars,
        )

        if success:
            score = step.get("score", 0.0) * score_ratio
        else:
            score = 0.0

        return success, message, score

    def _check_sequence(
        self,
        text: str,
        patterns: List[Dict[str, Any]],
        case_sensitive: bool,
        regex_mode: bool,
        allow_partial: bool,
        verify_end: bool = False,
        whitespace_chars: str = " \t\n\r",
    ) -> Tuple[bool, float, str]:
        """
        Checks if the text matches the sequence of patterns.

        Args:
            text: The text to check.
            patterns: A list of patterns.
            case_sensitive: Whether to distinguish case sensitivity.
            regex_mode: Whether to use regular expressions.
            allow_partial: Whether to allow partial matching.
            verify_end: Whether to verify that there is no more output after the last pattern.
            whitespace_chars: The set of characters considered whitespace.

        Returns:
            A tuple containing (success, match score ratio, error message).
        """

        if not case_sensitive:
            text = text.lower()

        current_pos = 0
        matched_count = 0
        total_items = 0
        total_weight = 0
        matched_weight = 0
        errors = []

        for pattern_item in patterns:
            if "pattern" in pattern_item:
                # 单个模式
                pattern = pattern_item["pattern"]
                required = pattern_item.get("required", True)
                weight = pattern_item.get("weight", 1.0)
                description = pattern_item.get("description", f"Pattern: {pattern}")

                total_items += 1
                total_weight += weight

                # 处理大小写敏感性
                if not case_sensitive and regex_mode:
                    pattern = f"(?i){pattern}"
                elif not case_sensitive:
                    pattern = pattern.lower()

                # 在当前位置之后搜索模式
                match_pos = self._find_pattern(text, pattern, current_pos, regex_mode)

                if match_pos >= 0:
                    # 找到匹配，更新位置并计数
                    current_pos = match_pos
                    matched_count += 1
                    matched_weight += weight
                elif required:
                    # 必需的模式没找到
                    error_msg = f"Required pattern not found: {description}"
                    errors.append(error_msg)
                    if not allow_partial:
                        return False, 0.0, error_msg

            elif "unordered" in pattern_item:
                # 无序组
                unordered_patterns = pattern_item["unordered"]
                required = pattern_item.get("required", True)
                group_weight = pattern_item.get("weight", len(unordered_patterns))
                group_description = pattern_item.get("description", "Unordered pattern group")

                total_items += len(unordered_patterns)
                total_weight += group_weight

                # 处理无序组
                result, end_pos, matched_item_count, group_matched_weight, unmatched_patterns = (
                    self._match_unordered_group(
                        text,
                        unordered_patterns,
                        current_pos,
                        case_sensitive,
                        regex_mode,
                    )
                )

                if result:
                    # 找到所有无序模式
                    current_pos = end_pos
                    matched_count += matched_item_count
                    # 按比例分配组权重
                    if len(unordered_patterns) > 0:
                        matched_weight += group_weight * (
                            matched_item_count / len(unordered_patterns)
                        )
                elif required:
                    # 必需的无序组没找到
                    error_msg = f"Required unordered group not fully matched: {group_description} - {matched_item_count}/{len(unordered_patterns)} patterns"
                    
                    # 如果有没匹配的模式，添加它们的描述
                    if unmatched_patterns:
                        error_msg += "\nUnmatched patterns:"
                        for p in unmatched_patterns:
                            p_desc = p.get("description", f"Pattern: {p.get('pattern')}")
                            error_msg += f"\n  - {p_desc}"
                    
                    errors.append(error_msg)
                    if not allow_partial:
                        return False, 0.0, error_msg

        if verify_end and matched_count == total_items and current_pos < len(text):
            # 检查剩余文本是否只包含空白字符
            remaining_text = text[current_pos:]
            if any(c not in whitespace_chars for c in remaining_text):
                # 发现非空白字符
                return (
                    False,
                    0.0,
                    f"Unexpected output after the last pattern: '{remaining_text.strip()}'",
                )

        # 计算匹配分数
        score_ratio = matched_weight / total_weight if total_weight > 0 else 0.0

        # 判断匹配结果和计算得分
        if matched_count == total_items:
            # 完全匹配
            return True, 1.0, "All patterns matched in sequence"
        elif allow_partial and matched_count > 0:
            # 部分匹配且允许部分得分
            score_ratio = matched_weight / total_weight if total_weight > 0 else 0.0
            return (
                True,
                score_ratio,
                f"Partial match: {matched_count}/{total_items} patterns",
            )
        else:
            # 部分匹配但不允许部分得分，或完全不匹配
            error_detail = "\n".join(errors) if errors else ""
            return False, 0.0, f"Only {matched_count}/{total_items} patterns matched\n{error_detail}"

    def _find_pattern(
        self, text: str, pattern: str, start_pos: int, regex_mode: bool
    ) -> int:
        """
        在文本中从指定位置开始查找模式的第一次出现

        返回:
            匹配结束的位置，未找到则返回-1
        """
        if regex_mode:
            match = re.search(pattern, text[start_pos:], re.MULTILINE)
            if match:
                match_end = start_pos + match.end()
                return match_end
            return -1
        else:
            # 字符串精确匹配
            pos = text.find(pattern, start_pos)
            if pos >= 0:
                return pos + len(pattern)
            return -1

    def _match_unordered_group(
        self,
        text: str,
        patterns: List[Any],
        start_pos: int,
        case_sensitive: bool,
        regex_mode: bool,
    ) -> Tuple[bool, int, int, float, List[Dict[str, Any]]]:
        """
        尝试匹配无序组中的所有模式

        返回:
            (是否全部匹配成功, 最后匹配位置, 匹配的模式数量, 匹配的权重总和, 未匹配的模式)
        """
        # 转换模式格式和提取权重
        processed_patterns = []
        for p in patterns:
            if isinstance(p, dict):
                pattern = p.get("pattern", "")
                weight = p.get("weight", 1.0)
                # 保存原始模式对象的引用
                original_pattern = p
            else:
                pattern = p
                weight = 1.0
                # 为字符串模式创建一个简单的字典
                original_pattern = {"pattern": p}

            if not case_sensitive and regex_mode:
                pattern = f"(?i){pattern}"
            elif not case_sensitive:
                pattern = pattern.lower()

            processed_patterns.append(
                {
                    "pattern": pattern, 
                    "matched": False, 
                    "weight": weight,
                    "original": original_pattern
                }
            )

        # 当前搜索范围
        current_text = text[start_pos:]
        last_match_end = 0
        matched_count = 0
        matched_weight = 0

        # 尝试匹配所有模式
        for _ in range(len(processed_patterns)):
            best_match = None
            best_pattern_idx = -1

            # 在所有未匹配的模式中找最早出现的
            for i, state in enumerate(processed_patterns):
                if state["matched"]:
                    continue

                if regex_mode:
                    match = re.search(state["pattern"], current_text)
                    if match and (best_match is None or match.start() < best_match[0]):
                        best_match = (match.start(), match.end())
                        best_pattern_idx = i
                else:
                    pos = current_text.find(state["pattern"])
                    if pos >= 0 and (best_match is None or pos < best_match[0]):
                        best_match = (pos, pos + len(state["pattern"]))
                        best_pattern_idx = i

            if best_pattern_idx >= 0:
                # 标记为已匹配
                processed_patterns[best_pattern_idx]["matched"] = True
                matched_weight += processed_patterns[best_pattern_idx]["weight"]
                matched_count += 1

                # 更新最后匹配位置
                last_match_end = max(last_match_end, best_match[1])
            else:
                # 找不到更多匹配
                break

        # 收集所有未匹配的模式
        unmatched_patterns = [
            p["original"] for p in processed_patterns if not p["matched"]
        ]

        # 检查是否所有模式都已匹配
        all_matched = matched_count == len(processed_patterns)

        return all_matched, start_pos + last_match_end, matched_count, matched_weight, unmatched_patterns


class CompositeChecker:
    def __init__(self):
        self.checkers = [
            StandardOutputChecker(),
            SpecialJudgeChecker(),
            PatternChecker(),
            SequenceChecker(),
        ]

    def check(
        self,
        step: Dict[str, Any],
        output: str,
        error: str,
        return_code: int,
        test_dir: Path,
    ) -> Tuple[bool, str, Optional[float]]:
        for checker in self.checkers:
            success, message, score = checker.check(
                step, output, error, return_code, test_dir
            )
            if not success:
                return success, message, score
        return True, "All checks passed", None


class InteractiveProcess:
    """管理与进程的交互会话"""

    def __init__(
        self,
        cmd: List[str],
        cwd: str,
        timeout: float = 30.0,
        stderr_to_stdout: bool = False,
    ):
        self.cmd = cmd
        self.cwd = cwd
        self.timeout = timeout
        self.process = None
        self.stdout_buffer = io.StringIO()
        self.stderr_buffer = io.StringIO()
        self.stdout_data = ""  # 存储当前已读取的所有输出
        self.stderr_data = ""
        self.closed = False
        self.start_time = None
        self.prompt = "> "  # Shell 提示符
        self.stderr_to_stdout = stderr_to_stdout
        # 用于同步的事件和锁
        self.output_lock = threading.Lock()
        self.output_event = threading.Event()  # 用于检测输出是否稳定

    def start(self):
        """启动交互式进程"""
        self.process = subprocess.Popen(
            self.cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE if not self.stderr_to_stdout else subprocess.STDOUT,
            text=True,
            bufsize=0,  # 无缓冲
            cwd=self.cwd,
        )
        self.start_time = time.time()

        # 启动读取线程
        self._start_reading_threads()

    def _start_reading_threads(self):
        """启动读取输出的线程"""
        self.stdout_thread = threading.Thread(
            target=self._read_output,
            args=(self.process.stdout, self.stdout_buffer, "stdout"),
        )
        self.stdout_thread.daemon = True
        self.stdout_thread.start()
        if not self.stderr_to_stdout:
            self.stderr_thread = threading.Thread(
                target=self._read_output,
                args=(self.process.stderr, self.stderr_buffer, "stderr"),
            )
            self.stderr_thread.daemon = True
            self.stderr_thread.start()

    def _read_output(self, pipe, buffer, stream_name):
        """持续读取管道输出到缓冲区"""
        for line in iter(pipe.readline, ""):
            with self.output_lock:
                buffer.write(line)
                if stream_name == "stdout":
                    self.stdout_data += line
                else:
                    self.stderr_data += line
                # 通知有新输出
                self.output_event.set()

    def send_input(self, text: str, echo: bool = True, wait_for_output: bool = True):
        """
        发送文本到进程的标准输入

        Args:
            text: 要发送的文本
            echo: 是否在输出中显示命令 (默认: True)
            wait_for_output: 是否等待程序输出稳定 (默认: True)
        """
        if self.closed or self.process.stdin.closed:
            raise IOError("Standard input is closed")

        # 在发送前记录当前输出长度
        current_stdout_length = len(self.stdout_data)

        # 如果需要回显，使用提示符格式
        if echo:
            with self.output_lock:
                self.stdout_buffer.write(f"{self.prompt}{text}\n")
                self.stdout_data += f"{self.prompt}{text}\n"

        # 发送实际输入到进程
        self.process.stdin.write(text + "\n")
        self.process.stdin.flush()

        # 等待输出稳定 (如果需要)
        if wait_for_output:
            self._wait_for_output_stabilize(current_stdout_length)

    def _wait_for_output_stabilize(
        self, previous_length, timeout=2.0, check_interval=0.05
    ):
        """
        等待输出稳定 (没有新输出一段时间后认为已稳定)

        Args:
            previous_length: 发送命令前的输出长度
            timeout: 最长等待时间 (秒)
            check_interval: 检查间隔 (秒)
        """
        start_wait = time.time()
        last_change_time = start_wait
        last_length = previous_length

        while time.time() - start_wait < timeout:
            # 等待通知有新输出
            self.output_event.wait(check_interval)
            self.output_event.clear()

            # 检查输出是否有变化
            current_length = len(self.stdout_data)
            if current_length > last_length:
                last_length = current_length
                last_change_time = time.time()

            # 如果超过200ms没有新输出，且与输入前相比有变化，认为输出已稳定
            if (
                time.time() - last_change_time > 0.2
                and current_length > previous_length
            ):
                return

    def close_input(self):
        """关闭标准输入，发送EOF"""
        if not self.closed and self.process.stdin and not self.process.stdin.closed:
            self.process.stdin.close()
            self.closed = True

    def send_signal(self, sig: str):
        """发送信号给进程"""
        if not self.process:
            return

        signal_map = {
            "INT": signal.SIGINT,
            "TSTP": signal.SIGTSTP,
            "QUIT": signal.SIGQUIT,
            "KILL": signal.SIGKILL,
            "TERM": signal.SIGTERM,
        }

        if sig in signal_map:
            # 记录发送信号前的状态
            current_stdout_length = len(self.stdout_data)
            was_running = self.is_running()

            # 发送信号
            self.process.send_signal(signal_map[sig])

            # 对于可能终止进程的信号，检查进程状态
            if sig in ["INT", "TERM", "KILL"]:
                # 先给进程一点时间来处理信号
                time.sleep(0.1)

                # 检查进程状态是否改变
                if was_running and not self.is_running():
                    # 进程已终止，不需要等待输出
                    return

            # 固定等待一段时间，让信号处理和可能的输出有时间完成
            time.sleep(0.2)

            # 然后再尝试等待输出稳定（如果有新输出的话）
            if len(self.stdout_data) > current_stdout_length:
                self._wait_for_output_stabilize(current_stdout_length, timeout=2.0)

    def wait(self, timeout: Optional[float] = None):
        """等待进程终止"""
        if not self.process:
            return None

        try:
            return self.process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            return None

    def get_outputs(self) -> Tuple[str, str]:
        """获取当前累积的标准输出和标准错误"""
        stdout = self.stdout_buffer.getvalue()
        stderr = self.stderr_buffer.getvalue()
        return stdout, stderr

    def is_running(self) -> bool:
        """检查进程是否仍在运行"""
        if not self.process:
            return False
        return self.process.poll() is None

    def check_timeout(self) -> bool:
        """检查是否超时"""
        if not self.start_time:
            return False
        return time.time() - self.start_time > self.timeout

    def terminate(self):
        """终止进程"""
        if self.process and self.is_running():
            self.close_input()
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()

    def print_verbose(self, console: Console):
        console.print(f"[bold cyan]Command: {' '.join(self.cmd)}[/bold cyan]")
        console.print(f"[bold cyan]Current working directory: {self.cwd}[/bold cyan]")
        stdout, stderr = self.get_outputs()
        if stdout.strip():
            console.print("[bold cyan]Standard Output:[/bold cyan]")
            console.print(stdout)
        if stderr.strip():
            console.print("[bold cyan]Standard Error:[/bold cyan]")
            console.print(stderr)


class TestRunner:
    def __init__(
        self,
        config: Config,
        console: Optional[Console] = None,
        verbose: bool = False,
        dry_run: bool = False,
        no_check: bool = False,
        compare: bool = False,
    ):
        self.config = config
        self.console = console
        self.checker = CompositeChecker()
        self.verbose = verbose
        self.dry_run = dry_run
        self.no_check = no_check
        self.compare = compare

    def run_test(self, test: TestCase) -> TestResult:
        start_time = time.perf_counter()
        try:
            # 清理和创建构建目录
            build_dir = test.path / "build"
            if build_dir.exists():
                for file in build_dir.iterdir():
                    if file.is_file():
                        file.unlink()
            build_dir.mkdir(exist_ok=True)

            # 在dry-run模式下，显示测试点信息
            if self.dry_run:
                if self.console and not isinstance(self.console, type):
                    self.console.print(f"[bold]Test case:[/bold] {test.meta['name']}")
                    if "description" in test.meta:
                        self.console.print(
                            f"[bold]Description:[/bold] {test.meta['description']}"
                        )
                return self._execute_test_steps(test)

            result = None
            if self.console and not isinstance(self.console, type):
                # 在 rich 环境下显示进度条
                status_icons = {
                    "PASS": "[green]✓[/green]",
                    "PARTIAL": "[yellow]~[/yellow]",
                    "FAIL": "[red]✗[/red]",
                }
                with Progress(
                    SpinnerColumn(finished_text=status_icons["FAIL"]),
                    TextColumn("[progress.description]{task.description}"),
                    console=self.console,
                ) as progress:
                    total_steps = len(test.run_steps)
                    if total_steps == 1:
                        task_description = f"Running {test.meta['name']}..."
                    else:
                        task_description = f"Running {test.meta['name']} [0/{total_steps}]..."
                    
                    task = progress.add_task(
                        task_description,
                        total=total_steps,
                    )
                    result = self._execute_test_steps(test, progress, task)
                    # 根据状态设置图标
                    progress.columns[0].finished_text = status_icons[result.status]
                    # 更新最终状态，移除Running字样，加上结果提示
                    final_status = {
                        "PASS": "[green]Passed[/green]",
                        "PARTIAL": "[yellow]Partial[/yellow]",
                        "FAIL": "[red]Failed[/red]",
                    }[result.status]
                    progress.update(
                        task,
                        completed=total_steps,
                        description=f"{test.meta['name']} [{total_steps}/{total_steps}]: {final_status}",
                    )

                # 如果测试失败，在进度显示完成后输出失败信息
                if not result.success and not self.dry_run:
                    for error_details in result.error_details:
                        # 获取失败的步骤信息
                        step_index = error_details["step"]

                        self.console.print(
                            f"\n[red]Test '{test.meta['name']}' failed at step {step_index}:[/red]"
                        )
                        self.console.print(f"Command: {error_details['command']}")

                        if "stdout" in error_details:
                            self.console.print("\nActual output:")
                            self.console.print(error_details["stdout"].strip())

                        if "stderr" in error_details:
                            self.console.print("\nError output:")
                            self.console.print(error_details["stderr"].strip())

                        if "expected_output" in error_details:
                            self.console.print("\nExpected output:")
                            self.console.print(error_details["expected_output"])

                        if "error_message" in error_details:
                            self.console.print("\nError details:")
                            self.console.print(f"  {error_details['error_message']}")

                        if "return_code" in error_details:
                            self.console.print(
                                f"\nReturn code: {error_details['return_code']}"
                            )

                        self.console.print()  # 添加一个空行作为分隔
                return result
            else:
                # 在非 rich 环境下直接执行
                return self._execute_test_steps(test)
        except Exception as e:
            print(e)
            print(traceback.format_exc())
            return TestResult(
                success=False,
                message=f"Error: {str(e)}",
                time=time.perf_counter() - start_time,
                score=0,
                max_score=test.meta["score"],
            )

    def _execute_test_steps(
        self,
        test: TestCase,
        progress: Optional[Progress] = None,
        task: Optional[Any] = None,
    ) -> TestResult:
        start_time = time.perf_counter()
        step_scores = []
        total_score = 0
        has_step_scores = any("score" in step for step in test.run_steps)
        max_possible_score = (
            sum(step.get("score", 0) for step in test.run_steps)
            if has_step_scores
            else test.meta["score"]
        )
        steps_error_details = []
        test_success = True

        for i, step in enumerate(test.run_steps, 1):
            if progress is not None and task is not None:
                step_name = step.get("name", step["command"])
                if len(test.run_steps) == 1:
                    progress.update(
                        task,
                        description=f"Running {test.meta['name']}: {step_name}",
                        completed=i - 1,
                    )
                else:
                    progress.update(
                        task,
                        description=f"Running {test.meta['name']} [{i}/{len(test.run_steps)}]: {step_name}",
                        completed=i - 1,
                    )
                
                # 保存当前的进度和任务，以便交互式测试使用
                self.current_progress = progress
                self.current_task = task

            result = self._execute_single_step(test, step, i)

            # 收集结果信息，无论成功或失败
            if not result.success and not self.dry_run:
                steps_error_details.append(result.error_details)
                if step.get("must_pass", True):
                    test_success = False

            # 更新分数信息
            total_score += result.score
            if result.step_scores:
                step_scores.extend(result.step_scores)

            # 更新进度
            if progress is not None and task is not None:
                progress.update(
                    task,
                    description=f"Running {test.meta['name']} [{i}/{len(test.run_steps)}]: {step_name}",
                    completed=i,
                )

            # 如果步骤失败且必须通过，但继续执行剩余步骤以便verbose模式显示
            if not result.success and step.get("must_pass", True) and not test_success:
                # 不提前返回，继续执行后续步骤，但标记测试已失败
                continue

        # 如果有分步给分，确保总分不超过测试用例的总分
        if has_step_scores:
            total_score = min(total_score, test.meta["score"])
        else:
            total_score = test.meta["score"] if test_success else 0
            step_scores = None

        success = True if self.dry_run else (test_success and total_score > 0)
        return TestResult(
            success=success,
            message="All steps completed" if success else "Some steps failed",
            time=time.perf_counter() - start_time,
            score=total_score if success else 0,
            max_score=max_possible_score,
            step_scores=step_scores,
            error_details=steps_error_details if steps_error_details else None,
        )

    def _execute_interactive_mode(
        self, test: TestCase, step: Dict[str, Any], step_index: int
    ) -> TestResult:
        """执行交互式模式测试，可选择与参考实现比较"""
        start_time = time.perf_counter()
        pwd = test.path
        if "pwd" in step:
            pwd = Path(self._resolve_path(step["pwd"], test.path, test.path, False))

        # 解析命令和参数
        cmd = [self._resolve_path(step["command"], test.path, pwd)]
        args = [
            self._resolve_path(str(arg), test.path, pwd) for arg in step.get("args", [])
        ]

        # 在dry-run模式下，只打印命令和交互步骤
        if self.dry_run:
            return self._handle_dry_run_interactive(
                test, step, step_index, start_time, pwd, cmd, args
            )

        # 创建交互式进程
        try:
            interactive_process, ref_process = self._create_interactive_processes(
                test, step, pwd, cmd, args
            )

            # 如果有进度条，获取当前任务描述
            current_progress = getattr(self, "current_progress", None)
            current_task = getattr(self, "current_task", None)
            current_description = None
            
            if current_progress is not None and current_task is not None:
                current_description = current_progress.tasks[current_task].description
            
            # 执行交互步骤
            interactive_result = self._execute_interactive_steps(
                test,
                step,
                step_index,
                start_time,
                interactive_process,
                ref_process,
                cmd,
                args,
                current_progress,
                current_task,
                current_description,
            )

            # 如果交互执行返回了结果，直接返回
            if interactive_result:
                return interactive_result

            return TestResult(
                success=True,
                message="Interactive test completed successfully",
                time=time.perf_counter() - start_time,
                score=self.total_score,
                max_score=self.max_score
                if self.max_score > 0
                else step.get("score", test.meta["score"]),
                step_scores=self.step_scores,
                error_details=None,
            )

        except Exception as e:
            if interactive_process:
                interactive_process.terminate()
            if ref_process:
                ref_process.terminate()

            return TestResult(
                success=False,
                message=f"Interactive test error: {str(e)}",
                time=time.perf_counter() - start_time,
                score=0,
                max_score=step.get("score", test.meta["score"]),
                step_scores=[],
                error_details={
                    "step": step_index,
                    "step_name": step.get("name", step["command"]),
                    "error_message": str(e),
                    "command": " ".join(cmd + args),
                },
            )

    def _execute_interactive_steps(
        self,
        test: TestCase,
        step: Dict[str, Any],
        step_index: int,
        start_time: float,
        interactive_process: InteractiveProcess,
        ref_process: Optional[InteractiveProcess],
        cmd: List[str],
        args: List[str],
        current_progress: Optional[Progress] = None,
        current_task: Optional[Any] = None,
        current_description: Optional[str] = None,
    ) -> Optional[TestResult]:
        """执行交互步骤并收集结果"""
        # 初始化状态变量（使用属性而不是局部变量，避免nonlocal声明问题）
        self.step_scores = []
        self.total_score = 0
        self.max_score = 0
        self.previous_step_type = ""
        self.diff_results = []  # 存储每一步的差异结果
        
        # 启动进程
        interactive_process.start()
        if ref_process:
            ref_process.start()
            
        # 交互步骤总数
        total_interactive_steps = len(step.get("steps", []))
        
        # 获取步骤描述基础信息
        step_name = step.get("name", step["command"])
        base_description = current_description if current_description else f"Running {test.meta['name']}: {step_name}"
        result = None

        for i, interaction_step in enumerate(step.get("steps", []), 1):
            if result:
                break
            
            step_type = interaction_step.get("type", "")
            
            # 更新进度显示，显示内部步骤进度
            if current_progress is not None and current_task is not None and total_interactive_steps > 1:
                step_info = f"{step_type}"
                if step_type == "input" and "content" in interaction_step:
                    content = interaction_step["content"]
                    # 如果内容太长，截断显示
                    if len(content) > 30:
                        content = content[:27] + "..."
                    step_info = f"input: {content}"
                elif step_type == "signal" and "signal" in interaction_step:
                    step_info = f"signal: {interaction_step['signal']}"
                
                current_progress.update(
                    current_task,
                    description=f"{base_description} [{i}/{total_interactive_steps}] - {step_info}"
                )

            # 检查超时
            if interactive_process.check_timeout():
                if ref_process:
                    ref_process.terminate()
                return TestResult(
                    success=False,
                    message=f"Interactive process timed out after {step.get('timeout', 30.0)}s",
                    time=time.perf_counter() - start_time,
                    score=self.total_score,
                    max_score=step.get("score", test.meta["score"]),
                    step_scores=self.step_scores,
                    error_details={
                        "step": step_index,
                        "step_name": step.get("name", step["command"]),
                        "error_message": "Process timed out",
                        "command": " ".join(cmd + args),
                    },
                )

            # 处理不同类型的交互步骤
            if step_type == "input":
                self._handle_input_step(
                    interaction_step,
                    interactive_process,
                    ref_process,
                    self.previous_step_type,
                )
            elif step_type == "close":
                self._handle_close_step(interactive_process, ref_process)
            elif step_type == "wait":
                result = self._handle_wait_step(
                    interaction_step,
                    interactive_process,
                    ref_process,
                    test,
                    step,
                    step_index,
                    start_time,
                    cmd,
                    args,
                )
            elif step_type == "sleep":
                self._handle_sleep_step(interaction_step)
            elif step_type == "signal":
                self._handle_signal_step(
                    interaction_step, interactive_process, ref_process
                )
            elif step_type == "check":
                result = self._handle_check_step(
                    i,
                    interaction_step,
                    interactive_process,
                    ref_process,
                    test,
                    step,
                    step_index,
                    start_time,
                    cmd,
                    args,
                )

            self.previous_step_type = step_type
            
        # 交互测试结束后，恢复原来的描述
        if current_progress is not None and current_task is not None:
            current_progress.update(
                current_task,
                description=base_description
            )
            
        if self.verbose and self.console and not isinstance(self.console, type):
            self.console.print("[bold cyan]Interactive test finished[/bold cyan]")
            interactive_process.print_verbose(self.console)
            if ref_process:
                self.console.print("[bold]Reference program output:[/bold]")
                ref_process.print_verbose(self.console)

        if (
            self.compare
            and ref_process
            and hasattr(self, "diff_results")
            and self.diff_results
            and len(self.diff_results) > 0
        ):
            if self.console and not isinstance(self.console, type):
                has_diff = any(not diff["match"] for diff in self.diff_results)
                if has_diff:
                    self._print_diff_results(
                        self.diff_results, len(self.diff_results)
                    )
                else:
                    self.console.print(
                        "\n[bold green]All outputs match with reference implementation[/bold green]"
                    )
                    
        # 确保进程终止
        interactive_process.terminate()
        if ref_process:
            ref_process.terminate()
            
        if result:
            return result

        # 如果没有分步评分，使用整体分数
        if not self.step_scores:
            self.total_score = step.get("score", test.meta["score"])
            self.max_score = step.get("score", test.meta["score"])

        return None

    def _handle_input_step(
        self,
        interaction_step: Dict[str, Any],
        interactive_process: InteractiveProcess,
        ref_process: Optional[InteractiveProcess],
        previous_step_type: str,
    ) -> None:
        """处理输入步骤"""
        echo = interaction_step.get("echo", True)
        wait_for_output = interaction_step.get("wait_for_output", True)

        if previous_step_type in ["signal", "sleep"]:
            time.sleep(0.2)

        # 发送输入到主进程
        interactive_process.send_input(
            interaction_step.get("content", ""),
            echo=echo,
            wait_for_output=wait_for_output,
        )

        # 如果有参考实现，也发送相同的输入
        if ref_process:
            ref_process.send_input(
                interaction_step.get("content", ""),
                echo=echo,
                wait_for_output=wait_for_output,
            )

    def _handle_close_step(
        self,
        interactive_process: InteractiveProcess,
        ref_process: Optional[InteractiveProcess],
    ) -> None:
        """处理关闭步骤"""
        interactive_process.close_input()
        if ref_process:
            ref_process.close_input()

    def _handle_wait_step(
        self,
        interaction_step: Dict[str, Any],
        interactive_process: InteractiveProcess,
        ref_process: Optional[InteractiveProcess],
        test: TestCase,
        step: Dict[str, Any],
        step_index: int,
        start_time: float,
        cmd: List[str],
        args: List[str],
    ) -> Optional[TestResult]:
        """处理等待步骤"""
        # 等待进程终止
        wait_timeout = interaction_step.get("timeout", 5.0)
        exit_code = interactive_process.wait(wait_timeout)

        if exit_code is None:  # 超时未终止
            if interaction_step.get("must_terminate", True):
                if ref_process:
                    ref_process.terminate()
                return TestResult(
                    success=False,
                    message=f"Process did not terminate within {wait_timeout}s after wait command",
                    time=time.perf_counter() - start_time,
                    score=self.total_score,
                    max_score=step.get("score", test.meta["score"]),
                    step_scores=self.step_scores,
                    error_details={
                        "step": step_index,
                        "step_name": step.get("name", step["command"]),
                        "error_message": "Process did not terminate within timeout",
                        "command": " ".join(cmd + args),
                    },
                )

        # 如果有参考实现，也等待它终止
        if ref_process:
            ref_process.wait(wait_timeout)

        return None

    def _handle_sleep_step(self, interaction_step: Dict[str, Any]) -> None:
        """处理睡眠步骤"""
        time.sleep(interaction_step.get("seconds", 1.0))

    def _handle_signal_step(
        self,
        interaction_step: Dict[str, Any],
        interactive_process: InteractiveProcess,
        ref_process: Optional[InteractiveProcess],
    ) -> None:
        """处理信号步骤"""
        interactive_process.send_signal(interaction_step.get("signal", "INT"))
        if ref_process:
            ref_process.send_signal(interaction_step.get("signal", "INT"))

    def _handle_check_step(
        self,
        step_i: int,
        interaction_step: Dict[str, Any],
        interactive_process: InteractiveProcess,
        ref_process: Optional[InteractiveProcess],
        test: TestCase,
        step: Dict[str, Any],
        step_index: int,
        start_time: float,
        cmd: List[str],
        args: List[str],
    ) -> Optional[TestResult]:
        """处理检查步骤"""
        # 获取主进程输出
        stdout, stderr = interactive_process.get_outputs()

        # 如果有参考实现，获取它的输出并比较
        if ref_process:
            ref_stdout, ref_stderr = ref_process.get_outputs()
            diff_result = self._compare_outputs(stdout, stderr, ref_stdout, ref_stderr)
            self.diff_results.append(diff_result)

        # 在no_check模式下，直接认为检查通过
        if self.no_check:
            step_success = True
            message = "Check skipped in no_check mode"
            step_score = interaction_step.get("score", 0)
        else:
            # 检查主进程输出
            step_success, message, step_score = self._check_interactive_output(
                interaction_step, stdout, stderr, test.path
            )

        step_max_score = interaction_step.get("score", 0)
        self.max_score += step_max_score

        if step_score is not None:
            self.total_score += step_score
            self.step_scores.append(
                (f"Interactive step {step_i}", step_score, step_max_score)
            )

        if not step_success and interaction_step.get("must_pass", True):
            interactive_process.terminate()
            if ref_process:
                ref_process.terminate()

                # 如果启用了比较模式，输出差异信息
                if self.console and not isinstance(self.console, type):
                    self._print_diff_results(self.diff_results, step_i)

            return TestResult(
                success=False,
                message=f"Interactive step {step_i} failed: {message}",
                time=time.perf_counter() - start_time,
                score=self.total_score,
                max_score=self.max_score
                if self.max_score > 0
                else step.get("score", test.meta["score"]),
                step_scores=self.step_scores,
                error_details={
                    "step": step_index,
                    "step_name": step.get("name", step["command"]),
                    "error_message": message,
                    "command": " ".join(cmd + args),
                },
            )

        # 检查完成后是否需要清除缓冲区
        if interaction_step.get("clear_buffer", False) and interactive_process:
            # 清除输出缓冲区
            with interactive_process.output_lock:
                interactive_process.stdout_buffer = io.StringIO()
                interactive_process.stderr_buffer = io.StringIO()
                interactive_process.stdout_data = ""
                interactive_process.stderr_data = ""
                if self.verbose:
                    print(f"Buffer cleared after check step {step_index}")

            # 如果有参考实现，也清除它的缓冲区
            if ref_process:
                with ref_process.output_lock:
                    ref_process.stdout_buffer = io.StringIO()
                    ref_process.stderr_buffer = io.StringIO()
                    ref_process.stdout_data = ""
                    ref_process.stderr_data = ""

        return None

    def _check_interactive_output(
        self, step: Dict[str, Any], stdout: str, stderr: str, test_dir: Path
    ) -> Tuple[bool, str, Optional[float]]:
        """检查交互式步骤的输出"""
        success, message, score = self.checker.check(
            step,
            stdout,
            stderr,
            0,
            test_dir,
        )
        return success, message, score

    def _compare_outputs(
        self, test_stdout: str, test_stderr: str, ref_stdout: str, ref_stderr: str
    ) -> Dict[str, Any]:
        """比较两个进程的输出，返回差异信息"""
        # 比较标准输出
        stdout_diff = list(
            difflib.unified_diff(
                ref_stdout.splitlines(),
                test_stdout.splitlines(),
                fromfile="reference.stdout",
                tofile="tested.stdout",
                lineterm="",
            )
        )

        # 比较标准错误
        stderr_diff = list(
            difflib.unified_diff(
                ref_stderr.splitlines(),
                test_stderr.splitlines(),
                fromfile="reference.stderr",
                tofile="tested.stderr",
                lineterm="",
            )
        )

        # 确定是否完全匹配
        stdout_match = len(stdout_diff) == 0
        stderr_match = len(stderr_diff) == 0
        match = stdout_match and stderr_match

        return {
            "match": match,
            "stdout_match": stdout_match,
            "stderr_match": stderr_match,
            "stdout_diff": stdout_diff,
            "stderr_diff": stderr_diff,
        }

    def _print_diff_results(
        self, diff_results: List[Dict[str, Any]], current_step: int
    ):
        """打印差异结果"""
        if not self.console or isinstance(self.console, type):
            return

        self.console.print(
            f"\n[bold red]Output differences found in step {current_step}:[/bold red]"
        )

        for i, diff in enumerate(diff_results, 1):
            if not diff["match"]:
                if not diff["stdout_match"]:
                    self.console.print("[cyan]Standard Output Diff:[/cyan]")
                    for line in diff["stdout_diff"]:
                        if line.startswith("+"):
                            self.console.print(f"[green]{line}[/green]")
                        elif line.startswith("-"):
                            self.console.print(f"[red]{line}[/red]")
                        else:
                            self.console.print(line)

                if not diff["stderr_match"]:
                    self.console.print("[cyan]Standard Error Diff:[/cyan]")
                    for line in diff["stderr_diff"]:
                        if line.startswith("+"):
                            self.console.print(f"[green]{line}[/green]")
                        elif line.startswith("-"):
                            self.console.print(f"[red]{line}[/red]")
                        else:
                            self.console.print(line)

            # 如果打印到当前步骤就停止
            if i >= current_step:
                break

    def _execute_single_step(
        self, test: TestCase, step: Dict[str, Any], step_index: int
    ) -> TestResult:
        if "mode" in step and step["mode"] == "interactive":
            return self._execute_interactive_mode(test, step, step_index)

        start_time = time.perf_counter()

        # 处理dry-run模式
        if self.dry_run:
            return self._handle_dry_run_step(test, step, step_index, start_time)

        # 执行命令
        try:
            process_result = self._run_command(test, step)

            # 显示详细输出（如果启用）
            self._display_verbose_output(step_index, step, process_result)

            # 检查结果
            return self._check_step_result(
                test, step, step_index, start_time, process_result
            )

        except subprocess.TimeoutExpired:
            return self._create_timeout_result(test, step, step_index, start_time)

    def _handle_dry_run_step(
        self, test: TestCase, step: Dict[str, Any], step_index: int, start_time: float
    ) -> TestResult:
        """处理dry-run模式下的步骤执行"""
        cmd = [self._resolve_path(step["command"], test.path)]
        args = [self._resolve_path(str(arg), test.path) for arg in step.get("args", [])]

        if self.console and not isinstance(self.console, type):
            self.console.print(f"\n[bold cyan]Step {step_index}:[/bold cyan]")
            if "name" in step:
                self.console.print(f"[bold]Name:[/bold] {step['name']}")
            self.console.print(f"[bold]Command:[/bold] {' '.join(cmd + args)}")
            if "stdin" in step:
                self.console.print(f"[bold]Input file:[/bold] {step['stdin']}")
            if "check" in step and not self.no_check:
                self.console.print("[bold]Checks:[/bold]")
                for check_type, check_value in step["check"].items():
                    if check_type == "files":
                        check_value = [
                            self._resolve_path(file, test.path) for file in check_value
                        ]
                    self.console.print(f"  - {check_type}: {check_value}")

        return TestResult(
            success=True,
            message="Dry run",
            time=time.perf_counter() - start_time,
            score=0,
            max_score=0,
        )

    def _run_command(
        self, test: TestCase, step: Dict[str, Any]
    ) -> subprocess.CompletedProcess:
        """执行命令并返回结果"""
        # 获取相对于测试目录的命令路径
        cmd = [self._resolve_path(step["command"], test.path, test.path)]
        args = [
            self._resolve_path(str(arg), test.path, test.path)
            for arg in step.get("args", [])
        ]

        return subprocess.run(
            cmd + args,
            cwd=test.path,
            input=self._get_stdin_data(test, step),
            capture_output=True,
            text=True,
            timeout=step.get("timeout", 5.0),
        )

    def _display_verbose_output(
        self,
        step_index: int,
        step: Dict[str, Any],
        process: subprocess.CompletedProcess,
    ) -> None:
        """显示详细的命令执行输出（如果启用了verbose模式）"""
        if not (self.verbose and self.console and not isinstance(self.console, type)):
            return

        cmd = [step["command"]]
        if "args" in step:
            cmd.extend(step.get("args", []))

        self.console.print(f"[bold cyan]Step {step_index} Output:[/bold cyan]")
        self.console.print("[bold]Command:[/bold]", " ".join(cmd))
        if process.stdout:
            self.console.print("[bold]Standard Output:[/bold]")
            self.console.print(process.stdout)
        if process.stderr:
            self.console.print("[bold]Standard Error:[/bold]")
            self.console.print(process.stderr)
        self.console.print(f"[bold]Return Code:[/bold] {process.returncode}\n")

    def _check_step_result(
        self,
        test: TestCase,
        step: Dict[str, Any],
        step_index: int,
        start_time: float,
        process: subprocess.CompletedProcess,
    ) -> TestResult:
        """检查步骤执行结果"""
        # 在no_check模式下，只要命令执行成功就认为通过
        if self.no_check:
            return self._create_success_result(
                test,
                step,
                step.get("score", test.meta["score"]) if process.returncode == 0 else 0,
                start_time,
            )

        # 如果有检查配置，执行检查
        if "check" in step:
            success, message, score = self.checker.check(
                step,
                process.stdout,
                process.stderr,
                process.returncode,
                test.path,
            )

            if not success:
                return self._create_failure_result(
                    test,
                    step,
                    step_index,
                    message,
                    start_time,
                    process.stdout,
                    process.stderr,
                    process.returncode,
                    process.stdout
                    if "expected_output" in step.get("check", {})
                    else "",
                )

        return self._create_success_result(test, step, score, start_time)

    def _resolve_relative_path(self, path: str, cwd: Path = os.getcwd()) -> str:
        result = path
        if isinstance(path, Path):
            result = str(path.resolve())

        try:
            result = str(path.relative_to(cwd, walk_up=True))
        except:
            try:
                result = str(path.relative_to(cwd))
            except:
                result = str(path)

        if len(result) > len(str(path)):
            result = str(path)

        return result

    def _resolve_path(
        self, path: str, test_dir: Path, cwd: Path = os.getcwd(), relative: bool = True
    ) -> str:
        if path.startswith("${") and path.endswith("}"):
            exec_name = path[2:-1]  # 去掉 ${ 和 }
            if exec_name in self.config.executables:
                path = self.config.executables[exec_name]

        build_dir = test_dir / "build"
        build_dir.mkdir(exist_ok=True)
        
        if relative:
            replacements = {
                "${test_dir}": self._resolve_relative_path(test_dir, cwd),
                "${common_dir}": self._resolve_relative_path(
                    self.config.paths["common_dir"], cwd
                ),
                "${root_dir}": self._resolve_relative_path(
                    self.config.project_root, cwd
                ),
                "${build_dir}": self._resolve_relative_path(build_dir, cwd),
            }
        else:
            replacements = {
                "${test_dir}": str(test_dir),
                "${common_dir}": str(self.config.paths["common_dir"]),
                "${root_dir}": str(self.config.project_root),
                "${build_dir}": str(build_dir),
            }

        for var, value in replacements.items():
            path = path.replace(var, value)

        return path

    def _get_stdin_data(self, test: TestCase, step: Dict[str, Any]) -> Optional[str]:
        if "stdin" not in step:
            return None

        stdin_file = test.path / step["stdin"]
        if not stdin_file.exists():
            raise FileNotFoundError(f"Input file {step['stdin']} not found")

        with open(stdin_file) as f:
            return f.read()

    def _create_timeout_result(
        self, test: TestCase, step: Dict[str, Any], step_index: int, start_time: float
    ) -> TestResult:
        error_message = f"Step {step_index} '{step.get('name', step['command'])}' timed out after {step.get('timeout', 5.0)}s"
        # 构造命令字符串
        cmd = [self._resolve_path(step["command"], test.path, os.getcwd())]
        if "args" in step:
            cmd.extend(
                [
                    self._resolve_path(str(arg), test.path, os.getcwd())
                    for arg in step.get("args", [])
                ]
            )
        command_str = " ".join(cmd)
        return TestResult(
            success=False,
            message=error_message,
            time=time.perf_counter() - start_time,
            score=0,
            max_score=step.get("score", test.meta["score"]),
            error_details={
                "step": step_index,
                "step_name": step.get("name", step["command"]),
                "error_message": error_message,
                "command": command_str,  # 添加实际运行的命令
            },
        )

    def _create_failure_result(
        self,
        test: TestCase,
        step: Dict[str, Any],
        step_index: int,
        message: str,
        start_time: float,
        stdout: str = "",
        stderr: str = "",
        return_code: Optional[int] = None,
        expected_output: str = "",
    ) -> TestResult:
        # 构造命令字符串
        cmd = [self._resolve_path(step["command"], test.path, os.getcwd())]
        if "args" in step:
            cmd.extend(
                [
                    self._resolve_path(str(arg), test.path, os.getcwd())
                    for arg in step.get("args", [])
                ]
            )
        command_str = " ".join(cmd)

        error_details = {
            "step": step_index,
            "step_name": step.get("name", step["command"]),
            "error_message": message,
            "command": command_str,  # 添加实际运行的命令
        }
        if stdout:
            error_details["stdout"] = stdout
        if stderr:
            error_details["stderr"] = stderr
        if return_code is not None:
            error_details["return_code"] = return_code
        if expected_output:
            error_details["expected_output"] = expected_output

        return TestResult(
            success=False,
            message=f"Step {step_index} '{step.get('name', step['command'])}' failed: {message}",
            time=time.perf_counter() - start_time,
            score=0,
            max_score=step.get("score", test.meta["score"]),
            error_details=error_details,
        )

    def _create_success_result(
        self,
        test: TestCase,
        step: Dict[str, Any],
        score: Optional[float],
        start_time: float,
    ) -> TestResult:
        step_score = score if score is not None else step.get("score", 0)
        return TestResult(
            success=True,
            message="Step completed successfully",
            time=time.perf_counter() - start_time,
            score=step_score,
            max_score=step.get("score", test.meta["score"]),
            step_scores=[
                (step.get("name", step["command"]), step_score, step.get("score", 0))
            ]
            if step.get("score", 0) > 0
            else None,
        )

    def _handle_dry_run_interactive(
        self,
        test: TestCase,
        step: Dict[str, Any],
        step_index: int,
        start_time: float,
        pwd: Path,
        cmd: List[str],
        args: List[str],
    ) -> TestResult:
        """处理交互式模式下的干运行"""
        if self.console and not isinstance(self.console, type):
            self.console.print(
                f"\n[bold cyan]Interactive Step {step_index}:[/bold cyan]"
            )
            if "name" in step:
                self.console.print(f"[bold]Name:[/bold] {step['name']}")
            self.console.print(f"[bold]Command:[/bold] {' '.join(cmd + args)}")
            self.console.print(f"[bold]Working Directory:[/bold] {pwd}")
            self.console.print(f"[bold]Timeout:[/bold] {step.get('timeout', 30.0)}s")

            # 如果启用了比较模式且有参考实现，打印参考实现的信息
            if self.compare and "reference" in step:
                ref_cmd = [
                    self._resolve_path(step["reference"]["command"], test.path, pwd)
                ]
                ref_args = [
                    self._resolve_path(str(arg), test.path, pwd)
                    for arg in step["reference"].get("args", [])
                ]
                self.console.print(
                    f"[bold]Reference:[/bold] {' '.join(ref_cmd + ref_args)}"
                )

            if "steps" in step:
                self.console.print("[bold]Interactive Steps:[/bold]")
                for i, interaction_step in enumerate(step["steps"], 1):
                    step_type = interaction_step.get("type", "")
                    self.console.print(f"  [bold]{i}. {step_type}[/bold]")

                    if step_type == "input":
                        self.console.print(
                            f"    Content: {interaction_step.get('content', '')}"
                        )
                        self.console.print(
                            f"    Echo: {interaction_step.get('echo', True)}"
                        )
                        self.console.print(
                            f"    Wait for output: {interaction_step.get('wait_for_output', True)}"
                        )
                    elif step_type == "wait":
                        self.console.print(
                            f"    Timeout: {interaction_step.get('timeout', 5.0)}s"
                        )
                        self.console.print(
                            f"    Must terminate: {interaction_step.get('must_terminate', True)}"
                        )
                    elif step_type == "sleep":
                        self.console.print(
                            f"    Seconds: {interaction_step.get('seconds', 1.0)}"
                        )
                    elif step_type == "signal":
                        self.console.print(
                            f"    Signal: {interaction_step.get('signal', 'INT')}"
                        )
                    elif step_type == "check":
                        self.console.print(
                            f"    Must pass: {interaction_step.get('must_pass', True)}"
                        )
                        if "score" in interaction_step:
                            self.console.print(
                                f"    Score: {interaction_step['score']}"
                            )
                        if "check" in interaction_step and not self.no_check:
                            self.console.print("    Checks:")
                            for check_type, check_value in interaction_step[
                                "check"
                            ].items():
                                if check_type == "files":
                                    check_value = [
                                        self._resolve_path(file, test.path)
                                        for file in check_value
                                    ]
                                self.console.print(
                                    f"      - {check_type}: {check_value}"
                                )

        return TestResult(
            success=True,
            message="Dry run of interactive test",
            time=time.perf_counter() - start_time,
            score=0,
            max_score=0,
        )

    def _create_interactive_processes(
        self,
        test: TestCase,
        step: Dict[str, Any],
        pwd: Path,
        cmd: List[str],
        args: List[str],
    ) -> Tuple[InteractiveProcess, Optional[InteractiveProcess]]:
        """创建交互式进程和可选的参考进程"""
        # 创建主测试进程
        interactive_process = InteractiveProcess(
            cmd + args,
            cwd=str(pwd),
            timeout=step.get("timeout", 30.0),
            stderr_to_stdout=step.get("stderr_to_stdout", False),
        )

        # 如果启用了比较模式且有参考实现，创建参考实现进程
        ref_process = None
        if self.compare and "reference" in step:
            ref_cmd = [self._resolve_path(step["reference"]["command"], test.path, pwd)]
            ref_args = [
                self._resolve_path(str(arg), test.path, pwd)
                for arg in step["reference"].get("args", [])
            ]
            ref_process = InteractiveProcess(
                ref_cmd + ref_args,
                cwd=str(pwd),
                timeout=step.get("timeout", 30.0),
                stderr_to_stdout=step.get("stderr_to_stdout", False),
            )

        return interactive_process, ref_process


class ResultFormatter(ABC):
    @abstractmethod
    def format_results(
        self,
        test_cases: List[TestCase],
        results: List[Dict[str, Any]],
        total_score: float,
        max_score: float,
    ) -> None:
        pass


class JsonFormatter(ResultFormatter):
    def format_results(
        self,
        test_cases: List[TestCase],
        results: List[Dict[str, Any]],
        total_score: float,
        max_score: float,
    ) -> None:
        json_result = {
            "total_score": round(total_score, 1),
            "max_score": round(max_score, 1),
            "percentage": round(total_score / max_score * 100, 1),
            "tests": results,
        }
        print(json.dumps(json_result, ensure_ascii=False))


class TableFormatter(ResultFormatter):
    def __init__(self, console: Console):
        self.console = console

    def format_results(
        self,
        test_cases: List[TestCase],
        results: List[Dict[str, Any]],
        total_score: float,
        max_score: float,
    ) -> None:
        self._format_rich_table(test_cases, results, total_score, max_score)

    def _format_rich_table(
        self,
        test_cases: List[TestCase],
        results: List[Dict[str, Any]],
        total_score: float,
        max_score: float,
    ) -> None:
        table = Table(show_header=True, header_style="bold")
        table.add_column("Test Case", style="cyan")
        table.add_column("Result", justify="center")
        table.add_column("Time", justify="right")
        table.add_column("Score", justify="right")
        table.add_column("Message")

        status_style = {
            "PASS": "[green]PASS[/green]",
            "PARTIAL": "[yellow]PARTIAL[/yellow]",
            "FAIL": "[red]FAIL[/red]",
        }

        for test, result in zip(test_cases, results):
            table.add_row(
                test.meta["name"],
                status_style[result["status"]],
                f"{result['time']:.2f}s",
                f"{result['score']:.1f}/{result['max_score']:.1f}",
                result["message"],
            )

        self.console.print(table)
        self._print_summary(total_score, max_score)

    def _format_basic_table(
        self,
        test_cases: List[TestCase],
        results: List[Dict[str, Any]],
        total_score: float,
        max_score: float,
    ) -> None:
        # 定义列宽
        col_widths = {
            "name": max(len(test.meta["name"]) for test in test_cases),
            "status": 8,  # PASS/PARTIAL/FAIL
            "time": 10,  # XX.XXs
            "score": 15,  # XX.X/XX.X
            "message": 40,
        }

        # 打印表头
        header = (
            f"{'Test Case':<{col_widths['name']}} "
            f"{'Result':<{col_widths['status']}} "
            f"{'Time':>{col_widths['time']}} "
            f"{'Score':>{col_widths['score']}} "
            f"{'Message':<{col_widths['message']}}"
        )
        self.console.print("-" * len(header))
        self.console.print(header)
        self.console.print("-" * len(header))

        # 打印每一行
        status_text = {
            "PASS": "PASS",
            "PARTIAL": "PARTIAL",
            "FAIL": "FAIL",
        }

        for test, result in zip(test_cases, results):
            row = (
                f"{test.meta['name']:<{col_widths['name']}} "
                f"{status_text[result['status']]:<{col_widths['status']}} "
                f"{result['time']:.2f}s".rjust(col_widths["time"])
                + f" {result['score']:.1f}/{result['max_score']}".rjust(
                    col_widths["score"]
                )
                + " "
                f"{result['message'][: col_widths['message']]:<{col_widths['message']}}"
            )
            self.console.print(row)

        self.console.print("-" * len(header))
        self._print_basic_summary(total_score, max_score)

    def _print_summary(self, total_score: float, max_score: float) -> None:
        summary = Panel(
            f"[bold]Total Score: {total_score:.1f}/{max_score:.1f} "
            f"({total_score / max_score * 100:.1f}%)[/bold]",
            border_style="green" if total_score == max_score else "yellow",
        )
        self.console.print()
        self.console.print(summary)

    def _print_basic_summary(self, total_score: float, max_score: float) -> None:
        self.console.print()
        self.console.print(
            f"Total Score: {total_score:.1f}/{max_score:.1f} "
            f"({total_score / max_score * 100:.1f}%)"
        )


class VSCodeConfigGenerator:
    """Generate and manage VS Code debug configurations"""

    def __init__(self, project_root: Path, config: Config):
        self.project_root = project_root
        self.config = config
        self.vscode_dir = project_root / ".vscode"
        self.launch_file = self.vscode_dir / "launch.json"
        self.tasks_file = self.vscode_dir / "tasks.json"

    def generate_configs(
        self, failed_steps: List[Tuple[TestCase, Dict[str, Any]]], merge: bool = True
    ) -> None:
        """Generate VS Code configurations for debugging a failed test step"""
        self.vscode_dir.mkdir(exist_ok=True)

        launch_config = []
        tasks_config = []

        for test_case, failed_step in failed_steps:
            launch_config.extend(self._generate_launch_config(test_case, failed_step))
            tasks_config.extend(self._generate_tasks_config(test_case))

        launch_config = {"version": "0.2.0", "configurations": launch_config}
        tasks_config = {"version": "2.0.0", "tasks": tasks_config}

        self._write_or_merge_json(
            self.launch_file, launch_config, "configurations", merge
        )

        self._write_or_merge_json(self.tasks_file, tasks_config, "tasks", merge)

    def _generate_launch_config(
        self, test_case: TestCase, failed_step: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Generate launch configuration based on debug type"""
        debug_type = (
            failed_step.get("debug", {}).get("type")
            or test_case.meta.get("debug", {}).get("type")
            or self.config.debug_config["default_type"]
        )

        cwd = str(self.config.project_root)
        program = self._resolve_path(
            failed_step["command"], test_case.path, self.config.project_root
        )
        args = [
            self._resolve_path(arg, test_case.path, self.config.project_root)
            for arg in failed_step.get("args", [])
        ]

        if debug_type == "cpp":
            configs = []
            base_name = f"Debug {test_case.meta['name']} - Step {failed_step.get('name', 'failed step')}"

            # Add GDB configuration
            configs.append(
                {
                    "name": f"{base_name} (GDB)",
                    "type": "cppdbg",
                    "request": "launch",
                    "program": program,
                    "args": args,
                    "stopOnEntry": True,
                    "cwd": cwd,
                    "environment": [],
                    "internalConsoleOptions": "neverOpen",
                    "MIMode": "gdb",
                    "setupCommands": [
                        {
                            "description": "Enable pretty-printing for gdb",
                            "text": "-enable-pretty-printing",
                            "ignoreFailures": True,
                        }
                    ],
                    "preLaunchTask": f"build-{test_case.path.name}",
                }
            )

            # Add LLDB configuration
            configs.append(
                {
                    "name": f"{base_name} (LLDB)",
                    "type": "lldb",
                    "request": "launch",
                    "program": program,
                    "args": args,
                    "cwd": cwd,
                    "internalConsoleOptions": "neverOpen",
                    "preLaunchTask": f"build-{test_case.path.name}",
                }
            )

            return configs
        elif debug_type == "python":
            return [
                {
                    "name": f"Debug {test_case.meta['name']} - Step {failed_step.get('name', 'failed step')}",
                    "type": "python",
                    "request": "launch",
                    "program": program,
                    "args": args,
                    "cwd": cwd,
                    "env": {},
                    "console": "integratedTerminal",
                    "justMyCode": False,
                    "preLaunchTask": f"build-{test_case.path.name}",
                }
            ]
        elif debug_type == "rust":
            base_name = f"Debug {test_case.meta['name']} - Step {failed_step.get('name', 'failed step')}"
            configs = []

            # Add CodeLLDB configuration for Rust
            configs.append({
                "name": f"{base_name} (CodeLLDB)",
                "type": "lldb",
                "request": "launch",
                "program": program,
                "args": args,
                "cwd": cwd,
                "sourceLanguages": ["rust"],
                "internalConsoleOptions": "neverOpen",
                "preLaunchTask": f"build-{test_case.path.name}",
                "env": {
                    "RUST_BACKTRACE": "1"
                }
            })

            # Add GDB configuration for Rust
            configs.append({
                "name": f"{base_name} (GDB)",
                "type": "cppdbg",
                "request": "launch",
                "program": program,
                "args": args,
                "stopOnEntry": True,
                "cwd": cwd,
                "environment": [
                    {
                        "name": "RUST_BACKTRACE",
                        "value": "1"
                    }
                ],
                "internalConsoleOptions": "neverOpen",
                "MIMode": "gdb",
                "setupCommands": [
                    {
                        "description": "Enable pretty-printing for gdb",
                        "text": "-enable-pretty-printing",
                        "ignoreFailures": True,
                    },
                    {
                        "description": "Enable Rust pretty-printing",
                        "text": "set language rust",
                        "ignoreFailures": True,
                    }
                ],
                "preLaunchTask": f"build-{test_case.path.name}",
            })

            return configs
        else:
            raise ValueError(f"Unsupported debug type: {debug_type}")

    def _generate_tasks_config(self, test_case: TestCase) -> Dict[str, Any]:
        """Generate tasks configuration for building the test case"""
        return [
            {
                "label": f"build-{test_case.path.name}",
                "type": "shell",
                "command": "python3",
                "args": ["grader.py", "--no-check", test_case.path.name],
                "group": {"kind": "build", "isDefault": True},
                "presentation": {"panel": "shared"},
                "options": {"env": {"DEBUG": "1"}},
            }
        ]

    def _write_or_merge_json(
        self,
        file_path: Path,
        new_config: Dict[str, Any],
        merge_key: str,
        should_merge: bool,
    ) -> None:
        """Write or merge JSON configuration file, overriding items with the same name."""
        # Add comment to mark auto-generated configurations
        if merge_key == "configurations":
            for config in new_config[merge_key]:
                config["name"] = f"{config['name']} [Auto-generated]"
                config["preLaunchTask"] = f"{config['preLaunchTask']} [Auto-generated]"
        elif merge_key == "tasks":
            for task in new_config[merge_key]:
                task["label"] = f"{task['label']} [Auto-generated]"

        if file_path.exists() and should_merge:
            try:
                with open(file_path) as f:
                    existing_config = json.load(f)

                # Merge configurations
                if merge_key in existing_config:
                    # Create a dictionary to map names to their items for quick lookup
                    existing_items_map = {
                        item[
                            "name" if merge_key == "configurations" else "label"
                        ].replace(" [Auto-generated]", ""): item
                        for item in existing_config[merge_key]
                    }

                    # Update existing items or add new items
                    for new_item in new_config[merge_key]:
                        item_key = new_item[
                            "name" if merge_key == "configurations" else "label"
                        ].replace(" [Auto-generated]", "")
                        existing_items_map[item_key] = new_item

                    # Rebuild the list from the updated map
                    existing_config[merge_key] = list(existing_items_map.values())
                else:
                    existing_config[merge_key] = new_config[merge_key]

                config_to_write = existing_config
            except json.JSONDecodeError:
                config_to_write = new_config
        else:
            config_to_write = new_config

        with open(file_path, "w") as f:
            json.dump(config_to_write, f, indent=4)

    def _resolve_relative_path(self, path: str, cwd: Path = os.getcwd()) -> str:
        result = path
        if isinstance(path, Path):
            result = str(path.resolve())

        try:
            result = str(path.relative_to(cwd, walk_up=True))
        except:
            try:
                result = str(path.relative_to(cwd))
            except:
                result = str(path)

        if len(result) > len(str(path)):
            result = str(path)

        return result

    def _resolve_path(self, path: str, test_dir: Path, cwd: Path = os.getcwd()) -> str:
        if path.startswith("${") and path.endswith("}"):
            exec_name = path[2:-1]
            if exec_name in self.config.executables:
                path = self.config.executables[exec_name]
                
        build_dir = test_dir / "build"
        build_dir.mkdir(exist_ok=True)

        replacements = {
            "${test_dir}": self._resolve_relative_path(test_dir, cwd),
            "${common_dir}": self._resolve_relative_path(
                self.config.paths["common_dir"], cwd
            ),
            "${root_dir}": self._resolve_relative_path(self.config.project_root, cwd),
            "${build_dir}": self._resolve_relative_path(build_dir, cwd),
        }

        for var, value in replacements.items():
            path = path.replace(var, value)

        return path


class Grader:
    def __init__(
        self,
        verbose=False,
        json_output=False,
        dry_run=False,
        no_check=False,
        generate_vscode=False,
        vscode_no_merge=False,
        compare=False,
    ):
        self.config = Config(Path.cwd())
        self.verbose = verbose
        self.json_output = json_output
        self.dry_run = dry_run
        self.no_check = no_check
        self.generate_vscode = generate_vscode
        self.vscode_no_merge = vscode_no_merge
        self.compare = compare
        self.console = Console(quiet=json_output)
        self.runner = TestRunner(
            self.config,
            self.console,
            verbose=self.verbose,
            dry_run=self.dry_run,
            no_check=self.no_check,
            compare=self.compare,
        )
        self.formatter = (
            JsonFormatter() if json_output else TableFormatter(self.console)
        )
        self.results: Dict[str, TestResult] = {}
        self.vscode_generator = VSCodeConfigGenerator(Path.cwd(), self.config)

    def _save_test_history(
        self,
        test_cases: List[TestCase],
        test_results: List[Dict[str, Any]],
        total_score: float,
        max_score: float,
    ) -> None:
        """保存测试历史到隐藏文件"""
        history_data = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "total_score": round(total_score, 1),
            "max_score": round(max_score, 1),
            "percentage": round(total_score / max_score * 100, 1)
            if max_score > 0
            else 0,
            "tests": [],
        }

        for test, result in zip(test_cases, test_results):
            test_data = {
                "name": test.meta["name"],
                "description": test.meta.get("description", ""),
                "path": str(test.path),
                "build_path": str(test.path / "build"),
                "score": result["score"],
                "max_score": result["max_score"],
                "status": result["status"],
                "time": result["time"],
                "message": result["message"],
                "step_scores": result["step_scores"],
            }

            # 如果测试失败，添加详细的错误信息
            if not result["success"] and result["error_details"]:
                error_details = result["error_details"][0]
                test_data["error_details"] = {
                    "step": error_details["step"],
                    "step_name": error_details["step_name"],
                    "error_message": error_details["error_message"],
                    "command": error_details.get("command", ""),  # 添加实际运行的命令
                }
                if "stdout" in error_details:
                    test_data["error_details"]["stdout"] = error_details["stdout"]
                if "stderr" in error_details:
                    test_data["error_details"]["stderr"] = error_details["stderr"]
                if "return_code" in error_details:
                    test_data["error_details"]["return_code"] = error_details[
                        "return_code"
                    ]

            history_data["tests"].append(test_data)

        try:
            # 读取现有历史记录（如果存在）
            history_file = Path(".test_history")
            if history_file.exists():
                with open(history_file, "r", encoding="utf-8") as f:
                    history = json.load(f)
                # 限制历史记录数量为最近10次
                history = history[-9:]
            else:
                history = []

            # 添加新的测试记录
            history.append(history_data)

            # 保存更新后的历史记录
            with open(history_file, "w", encoding="utf-8") as f:
                json.dump(history, f, ensure_ascii=False, indent=2)

        except Exception as e:
            if not self.json_output:
                self.console.print(
                    f"[yellow]Warning:[/yellow] Failed to save test history: {str(e)}"
                )

    def _print_debug_instructions(
        self, test_case: TestCase, failed_step: Dict[str, Any]
    ) -> None:
        """Print instructions for debugging the failed test case"""
        if self.json_output:
            return

        debug_type = (
            failed_step.get("debug", {}).get("type")
            or test_case.meta.get("debug", {}).get("type")
            or self.config.debug_config["default_type"]
        )

        self.console.print("\n[bold]Debug Instructions:[/bold]")
        self.console.print("1. VS Code configurations have been generated/updated:")
        self.console.print("   - .vscode/launch.json: Debug configurations")
        self.console.print("   - .vscode/tasks.json: Build tasks")
        self.console.print("\n2. To debug the failed step:")
        self.console.print("   a. Open VS Code in the project root directory")
        self.console.print("   b. Install required extensions:")
        if debug_type in ["gdb", "lldb"]:
            self.console.print("      - C/C++: ms-vscode.cpptools")
        elif debug_type == "python":
            self.console.print("      - Python: ms-python.python")
        elif debug_type == "rust":
            self.console.print("      - rust-analyzer: rust-lang.rust-analyzer")
            self.console.print("      - CodeLLDB: vadimcn.vscode-lldb")
            self.console.print("      - C/C++ (optional, for GDB): ms-vscode.cpptools")
        self.console.print("   c. Press F5 or use the Run and Debug view")
        self.console.print(
            "   d. Select the auto-generated configuration for this test"
        )
        self.console.print("\n3. Debug features available:")
        self.console.print("   - Set breakpoints (F9)")
        self.console.print("   - Step over (F10)")
        self.console.print("   - Step into (F11)")
        self.console.print("   - Step out (Shift+F11)")
        self.console.print("   - Continue (F5)")
        self.console.print("   - Inspect variables in the Variables view")
        self.console.print("   - Use Debug Console for expressions")
        if debug_type == "rust":
            self.console.print("\n4. Rust-specific debug features:")
            self.console.print("   - RUST_BACKTRACE=1 is enabled for better error messages")
            self.console.print("   - CodeLLDB provides native Rust debugging experience")
            self.console.print("   - GDB configuration is also available as a backup option")
        self.console.print(
            "\nNote: The test will be automatically rebuilt before debugging starts"
        )

    def _collect_failed_steps(
        self,
    ) -> List[Tuple[TestCase, Dict[str, Any], Dict[str, Any]]]:
        """收集所有失败的测试步骤"""
        failed_steps = []
        for test_name, result in self.results.items():
            if not result.success and result.error_details:
                test_case = next(
                    test
                    for test in self._load_test_cases()
                    if test.path.name == test_name
                )
                # 处理所有失败步骤的错误信息
                error_details_list = (
                    result.error_details
                    if isinstance(result.error_details, list)
                    else [result.error_details]
                )
                for error_details in error_details_list:
                    step_index = error_details["step"]
                    failed_step = test_case.run_steps[step_index - 1]
                    failed_steps.append((test_case, failed_step, error_details))
        return failed_steps

    def _generate_debug_configs(self) -> None:
        """为所有失败的测试步骤生成调试配置"""
        if not self.generate_vscode:
            return

        failed_steps = self._collect_failed_steps()
        if not failed_steps:
            return

        try:
            self.vscode_generator.generate_configs(
                [
                    (test_case, failed_step)
                    for test_case, failed_step, _ in failed_steps
                ],
                merge=not self.vscode_no_merge,
            )

            if not self.json_output and failed_steps:
                self._print_debug_instructions(failed_steps[0][0], failed_steps[0][1])
                if len(failed_steps) > 1:
                    self.console.print(
                        f"\n[yellow]Note:[/yellow] Debug configurations have been generated for {len(failed_steps)} failed steps."
                    )
        except Exception as e:
            if not self.json_output:
                self.console.print(
                    f"\n[red]Failed to generate VS Code configurations:[/red] {str(e)}"
                )

    def run_all_tests(
        self,
        specific_test: Optional[str] = None,
        prefix_match: bool = False,
        group: Optional[str] = None,
        specific_paths: Optional[List[Path]] = None,
    ):
        try:
            if not self._run_setup_steps():
                sys.exit(1)

            test_cases = self._load_test_cases(
                specific_test, prefix_match, group, specific_paths
            )
            if not self.json_output:
                if self.dry_run:
                    self.console.print(
                        "\n[bold]Dry-run mode enabled. Only showing commands.[/bold]\n"
                    )
                elif group and test_cases:
                    matched_group = next(
                        g
                        for g in self.config.groups
                        if g.lower().startswith(group.lower())
                    )
                    self.console.print(
                        f"\n[bold]Running {len(test_cases)} test cases in group {matched_group}...[/bold]\n"
                    )
                else:
                    self.console.print(
                        f"\n[bold]Running {len(test_cases)} test cases...[/bold]\n"
                    )

            total_score = 0
            max_score = 0
            test_results = []

            for test in test_cases:
                try:
                    result = self.runner.run_test(test)
                except Exception as e:
                    if not self.json_output:
                        self.console.print(
                            f"[red]Error:[/red] Grader script error while running test '{test.meta['name']}': {str(e)}"
                        )
                    else:
                        print(
                            f"Error: Grader script error while running test '{test.meta['name']}': {str(e)}",
                            file=sys.stderr,
                        )
                    sys.exit(1)

                self.results[test.path.name] = result
                result_dict = {
                    "name": test.meta["name"],
                    "success": result.success,
                    "status": result.status,
                    "time": round(result.time, 2),
                    "score": result.score,
                    "max_score": result.max_score,
                    "step_scores": result.step_scores,
                    "message": result.message,
                    "error_details": result.error_details,
                }
                test_results.append(result_dict)
                total_score += result.score
                max_score += result.max_score

            if not self.dry_run:
                self.formatter.format_results(
                    test_cases, test_results, total_score, max_score
                )

                self._save_test_history(
                    test_cases, test_results, total_score, max_score
                )

                # 在所有测试完成后生成调试配置
                self._generate_debug_configs()

            return total_score, max_score

        except Exception as e:
            print(e)
            print(traceback.format_exc())
            if not self.json_output:
                self.console.print(f"[red]Error:[/red] Grader script error: {str(e)}")
            else:
                print(f"Error: Grader script error: {str(e)}", file=sys.stderr)
            sys.exit(1)

    def _run_setup_steps(self) -> bool:
        if not self.config.setup_steps:
            return True

        if self.console and not isinstance(self.console, type):
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=self.console,
            ) as progress:
                total_steps = len(self.config.setup_steps)
                task = progress.add_task(
                    f"Running setup steps [0/{total_steps}]...",
                    total=total_steps,
                )

                for i, step in enumerate(self.config.setup_steps, 1):
                    step_name = step.get("message", "Setup step")
                    progress.update(
                        task,
                        description=f"Running setup steps [{i}/{total_steps}]: {step_name}",
                        completed=i - 1,
                    )

                    if not self._run_setup_step(step):
                        progress.update(task, completed=total_steps)
                        return False

                    progress.update(
                        task,
                        description=f"Running setup steps [{i}/{total_steps}]: {step_name}",
                        completed=i,
                    )
                return True
        else:
            for step in self.config.setup_steps:
                if not self._run_setup_step(step):
                    return False
            return True

    def _run_setup_step(self, step: Dict[str, Any]) -> bool:
        try:
            if step["type"] != "command":
                if not self.json_output:
                    self.console.print(
                        f"[red]Error:[/red] Unknown setup step type: {step['type']}"
                    )
                return False

            cmd = [step["command"]]
            if "args" in step:
                if isinstance(step["args"], list):
                    cmd.extend(step["args"])
            else:
                cmd.append(step["args"])

            process = subprocess.run(
                cmd,
                cwd=self.config.project_root,
                capture_output=True,
                text=True,
                timeout=step.get("timeout", 5.0),
            )

            if process.returncode != 0:
                if not self.json_output:
                    self.console.print("[red]Error:[/red] Command failed:")
                    self.console.print(process.stderr)
                return False

            return True

        except Exception as e:
            if not self.json_output:
                self.console.print(f"[red]Error:[/red] Command failed: {str(e)}")
            return False

    def _load_test_cases(
        self,
        specific_test: Optional[str] = None,
        prefix_match: bool = False,
        group: Optional[str] = None,
        specific_paths: Optional[List[Path]] = None,
    ) -> List[TestCase]:
        # 如果指定了具体的测试路径，直接加载这些测试
        if specific_paths:
            test_cases = []
            for test_path in specific_paths:
                if test_path.is_dir() and (test_path / "config.toml").exists():
                    test_cases.append(self._load_single_test(test_path))
            if not test_cases:
                if not self.json_output:
                    self.console.print(
                        "[red]Error:[/red] No valid test cases found in specified paths"
                    )
                else:
                    print(
                        "Error: No valid test cases found in specified paths",
                        file=sys.stderr,
                    )
                sys.exit(1)
            return test_cases

        # 如果指定了组，则从组配置中获取测试点列表
        if group:
            # 查找匹配的组名
            matching_groups = []
            for group_name in self.config.groups:
                if group_name.lower().startswith(group.lower()):
                    matching_groups.append(group_name)

            if not matching_groups:
                if not self.json_output:
                    self.console.print(
                        f"[red]Error:[/red] No group matching '{group}' found in config"
                    )
                else:
                    print(
                        f"Error: No group matching '{group}' found in config",
                        file=sys.stderr,
                    )
                sys.exit(1)
            elif len(matching_groups) > 1:
                if not self.json_output:
                    message = (
                        f"[yellow]Warning:[/yellow] Multiple groups match '{group}':"
                        if prefix_match and specific_test.isdigit()
                        else f"[yellow]Warning:[/yellow] Multiple test cases start with '{specific_test}':"
                    )
                    self.console.print(message)
                    for g in matching_groups:
                        self.console.print(f"  - {g}")
                    self.console.print("Please be more specific in your group name.")
                else:
                    message = (
                        f"Error: Multiple groups match '{group}'"
                        if prefix_match and specific_test.isdigit()
                        else f"Error: Multiple test cases start with '{specific_test}'"
                    )
                    print(message, file=sys.stderr)
                sys.exit(1)

            group = matching_groups[0]
            test_cases = []
            for test_id in self.config.groups[group]:
                cases = self._load_test_cases(test_id, True)
                test_cases.extend(cases)

            if not test_cases:
                if not self.json_output:
                    self.console.print(
                        f"[red]Error:[/red] No test cases found in group '{group}'"
                    )
                else:
                    print(
                        f"Error: No test cases found in group '{group}'",
                        file=sys.stderr,
                    )
                sys.exit(1)

            return test_cases

        if specific_test:
            # 获取所有匹配的测试目录
            matching_tests = []
            for test_dir in self.config.paths["cases_dir"].iterdir():
                if test_dir.is_dir() and (test_dir / "config.toml").exists():
                    if prefix_match and specific_test.isdigit():
                        # 使用数字前缀精确匹配模式
                        prefix_match = re.match(r"^(\d+)", test_dir.name)
                        if prefix_match and prefix_match.group(1) == specific_test:
                            matching_tests.append(test_dir)
                    else:
                        # 使用常规的开头匹配
                        if test_dir.name.lower().startswith(specific_test.lower()):
                            matching_tests.append(test_dir)

            if not matching_tests:
                if not self.json_output:
                    message = (
                        f"[red]Error:[/red] No test cases with prefix number '{specific_test}' found"
                        if prefix_match and specific_test.isdigit()
                        else f"[red]Error:[/red] No test cases starting with '{specific_test}' found"
                    )
                    self.console.print(message)
                else:
                    message = (
                        f"Error: No test cases with prefix number '{specific_test}' found"
                        if prefix_match and specific_test.isdigit()
                        else f"Error: No test cases starting with '{specific_test}' found"
                    )
                    print(message, file=sys.stderr)
                sys.exit(1)
            elif len(matching_tests) > 1:
                # 如果找到多个匹配项，列出所有匹配的测试用例
                if not self.json_output:
                    message = (
                        f"[yellow]Warning:[/yellow] Multiple test cases have prefix number '{specific_test}':"
                        if prefix_match and specific_test.isdigit()
                        else f"[yellow]Warning:[/yellow] Multiple test cases start with '{specific_test}':"
                    )
                    self.console.print(message)
                    for test_dir in matching_tests:
                        config = tomli.load(open(test_dir / "config.toml", "rb"))
                        self.console.print(
                            f"  - {test_dir.name}: {config['meta']['name']}"
                        )
                    self.console.print(
                        "Please be more specific in your test case name."
                    )
                else:
                    message = (
                        f"Error: Multiple test cases have prefix number '{specific_test}'"
                        if prefix_match and specific_test.isdigit()
                        else f"Error: Multiple test cases start with '{specific_test}'"
                    )
                    print(message, file=sys.stderr)
                sys.exit(1)

            return [self._load_single_test(matching_tests[0])]

        if not self.config.paths["cases_dir"].exists():
            if not self.json_output:
                self.console.print("[red]Error:[/red] tests/cases directory not found")
            else:
                print("Error: tests/cases directory not found", file=sys.stderr)
            sys.exit(1)

        def get_sort_key(path: Path) -> tuple:
            # 尝试从文件夹名称中提取数字前缀
            match = re.match(r"(\d+)", path.name)
            if match:
                # 如果有数字前缀，返回 (0, 数字值, 文件夹名) 元组
                # 0 表示优先级最高
                return (0, int(match.group(1)), path.name)
            else:
                # 如果没有数字前缀，返回 (1, 0, 文件夹名) 元组
                # 1 表示优先级较低，这些文件夹会按字母顺序排在有数字前缀的文件夹后面
                return (1, 0, path.name)

        test_cases = []
        # 使用自定义排序函数
        for test_dir in sorted(
            self.config.paths["cases_dir"].iterdir(), key=get_sort_key
        ):
            if test_dir.is_dir() and (test_dir / "config.toml").exists():
                test_cases.append(self._load_single_test(test_dir))

        if not test_cases:
            if not self.json_output:
                self.console.print(
                    "[red]Error:[/red] No test cases found in tests/cases/"
                )
            else:
                print("Error: No test cases found in tests/cases/", file=sys.stderr)
            sys.exit(1)

        return test_cases

    def _load_single_test(self, test_path: Path) -> TestCase:
        try:
            with open(test_path / "config.toml", "rb") as f:
                config = tomli.load(f)

            if "meta" not in config:
                raise ValueError("Missing 'meta' section in config")
            if "name" not in config["meta"]:
                raise ValueError("Missing 'name' in meta section")
            if "score" not in config["meta"]:
                raise ValueError("Missing 'score' in meta section")
            if "run" not in config:
                raise ValueError("Missing 'run' section in config")

            return TestCase(
                path=test_path, meta=config["meta"], run_steps=config["run"]
            )
        except Exception as e:
            if not self.json_output:
                self.console.print(
                    f"[red]Error:[/red] Failed to load test '{test_path.name}': {str(e)}"
                )
            else:
                print(
                    f"Error: Failed to load test '{test_path.name}': {str(e)}",
                    file=sys.stderr,
                )
            sys.exit(1)


def get_current_shell() -> str:
    """
    获取当前用户使用的shell类型
    优先检查父进程，然后检查SHELL环境变量
    返回值: 'bash', 'zsh', 'fish' 等
    """
    # 方法1: 检查父进程
    try:
        parent_pid = os.getppid()
        with open(f"/proc/{parent_pid}/comm", "r") as f:
            shell = f.read().strip()
            if shell in ["bash", "zsh", "fish"]:
                return shell
    except:
        pass

    # 方法2: 通过SHELL环境变量
    shell_path = os.environ.get("SHELL", "")
    if shell_path:
        shell = os.path.basename(shell_path)
        if shell in ["bash", "zsh", "fish"]:
            return shell

    # 默认返回bash
    return "bash"

def get_last_failed_tests(history_file: Path | str) -> List[Any]:
    """Get the last failed tests from the history file

    Args:
        history_file (Path): The path to the history file

    Returns:
        List[Any]: The last failed tests
    """
    if isinstance(history_file, str):
        history_file = Path(history_file)

    if not history_file.exists():
        return []
    
    with open(history_file, "r", encoding="utf-8") as f:
        history = json.load(f)
        if not history:
            return []
        
        last_result = history[-1]
        return filter(lambda x: x["status"] != "PASS", last_result["tests"])

def main():
    parser = argparse.ArgumentParser(description="Grade student submissions")
    parser.add_argument(
        "-j", "--json", action="store_true", help="Output results in JSON format"
    )
    parser.add_argument(
        "-w",
        "--write-result",
        action="store_true",
        help="Write percentage score to .autograder_result file",
    )
    parser.add_argument(
        "-p",
        "--prefix",
        action="store_true",
        help="Use number prefix exact matching mode for test case selection",
    )
    parser.add_argument(
        "-g",
        "--group",
        help="Run all test cases in the specified group",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Show detailed output of each test step",
    )
    parser.add_argument(
        "-l",
        "--get-last-failed",
        action="store_true",
        help="Read last test result and output command to set TEST_BUILD environment variable",
    )
    parser.add_argument(
        "-f",
        "--rerun-failed",
        action="store_true",
        help="Only run the test cases that failed in the last run",
    )
    parser.add_argument(
        "--shell",
        choices=["bash", "zsh", "fish"],
        help="Manually specify shell type for environment variable commands",
    )
    parser.add_argument(
        "-d",
        "--dry-run",
        action="store_true",
        help="Only show commands that would be executed (only works with single test case)",
    )
    parser.add_argument(
        "-n",
        "--no-check",
        action="store_true",
        help="Run test cases without performing checks specified in config",
    )
    parser.add_argument(
        "--vscode",
        action="store_true",
        help="Generate VS Code debug configurations for failed test cases",
    )
    parser.add_argument(
        "--vscode-no-merge",
        action="store_true",
        help="Overwrite existing VS Code configurations instead of merging",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Enable comparison mode to show diff with reference implementation",
    )
    parser.add_argument("test", nargs="?", help="Specific test to run")
    args = parser.parse_args()

    try:
        if args.get_last_failed:
            try:
                failed_tests = get_last_failed_tests(".test_history")
                if not failed_tests:
                    print("No failed test found in last run", file=sys.stderr)
                    sys.exit(1)
                
                first_failed_test = failed_tests[0]
                shell_type = args.shell or get_current_shell()

                # 根据不同shell类型生成相应的命令
                if shell_type == "fish":
                    print(f"set -x TEST_BUILD {first_failed_test['build_path']}")
                else:  # bash 或 zsh
                    print(f"export TEST_BUILD={first_failed_test['build_path']}")
                    
                sys.exit(0)
                    
            except Exception as e:
                print(f"Error reading test history: {str(e)}", file=sys.stderr)
                sys.exit(1)

        # 检查dry-run模式是否与单个测试点一起使用
        if args.dry_run and not args.test:
            print(
                "Error: --dry-run can only be used with a single test case",
                file=sys.stderr,
            )
            sys.exit(1)

        grader = Grader(
            json_output=args.json,
            dry_run=args.dry_run,
            no_check=args.no_check,
            verbose=args.verbose,
            generate_vscode=args.vscode,
            vscode_no_merge=args.vscode_no_merge,
            compare=args.compare,
        )
        
        if args.rerun_failed:
            try:
                failed_tests = get_last_failed_tests(".test_history")
                if not failed_tests:
                    print("No failed test found in last run", file=sys.stderr)
                    sys.exit(1)
                
                failed_paths = [Path(test["path"]) for test in failed_tests]
                total_score, max_score = grader.run_all_tests(specific_paths=failed_paths)

            except Exception as e:
                print(f"Error reading test history: {str(e)}", file=sys.stderr)
                sys.exit(1)
        
        else:
            total_score, max_score = grader.run_all_tests(
                args.test, prefix_match=args.prefix, group=args.group
            )

        # 如果是dry-run模式，直接退出
        if args.dry_run:
            sys.exit(0)

        percentage = (total_score / max_score * 100) if max_score > 0 else 0

        # 如果需要写入结果文件
        if args.write_result:
            with open(".autograder_result", "w") as f:
                f.write(f"{percentage:.2f}")

        # 如果有测试点失败，输出提示信息
        if total_score < max_score and not args.json and grader.config.debug_config.get("show_test_build_hint", True):
            shell_type = args.shell or get_current_shell()

            grader.console.print(
                "\n[bold yellow]To set TEST_BUILD environment variable to the failed test case's build directory:[/bold yellow]"
            )

            if shell_type == "fish":
                grader.console.print(
                    "$ [bold green]python3 grader.py -l | source[/bold green]"
                )
            else:
                grader.console.print(
                    '$ [bold green]eval "$(python3 grader.py -l)"[/bold green]'
                )

        # 只要不是0分就通过
        sys.exit(0 if percentage > 0 else 1)
    except subprocess.CalledProcessError as e:
        print(
            f"Error: Command execution failed with return code {e.returncode}",
            file=sys.stderr,
        )
        sys.exit(1)
    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
