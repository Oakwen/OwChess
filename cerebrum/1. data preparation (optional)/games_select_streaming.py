##############################################################################
## NAME: games_select_streaming.py                                          ##
## AUTHOR: Enhanced for large PGN files (>200GB) with error logging        ##
## LICENSE: MIT                                                             ##
##############################################################################

##############################################################################
## PURPOSE:                                                                 ##
## Select games from multiple pgn files (streaming for large files)         ##
## Store results in ./games.pgn, errors in ./error.pgn                     ##
##############################################################################

import os
import re
import random
import sys
import time
import json
from pathlib import Path
from typing import Set, List, Generator, Optional, Tuple, Dict, Any
import psutil

# from src.games_select import MAX_PLIES

# 配置参数
# MAX_PLIES = 127
MAX_PLIES = 160
SEED = 21052014
BUFFER_SIZE = 1024 * 1024 * 50  # 50MB缓冲区
MAX_GAME_SIZE = 1024 * 1024 * 10  # 10MB单游戏最大大小
RESERVOIR_SIZE = 100000  # 蓄水池抽样大小
LOG_INTERVAL = 10000  # 每处理10000个游戏记录一次
CHECKPOINT_INTERVAL = 50000  # 每处理50000个游戏保存一次检查点

# 设置随机种子
random.seed(SEED)


class PGNMoveParser:
    """PGN棋步解析器，用于计算步数"""

    @staticmethod
    def calculate_plycount(game_text: str) -> int:
        """
        从PGN棋步字符串计算总步数(plies)
        返回: plycount (半着数)
        """
        # 找到棋步部分（通常在第一个空行之后）
        lines = game_text.strip().split("\n")

        # 跳过所有标签行（以[开头）
        move_lines = []
        in_moves_section = False

        for line in lines:
            line = line.strip()
            if not line:
                continue

            if line.startswith("["):
                # 标签行，继续跳过
                continue
            else:
                # 第一个非标签行，开始棋步部分
                in_moves_section = True

            if in_moves_section:
                move_lines.append(line)

        if not move_lines:
            return 0

        # 合并所有棋步行
        moves_text = " ".join(move_lines)

        # 移除注释和变体
        moves_text = PGNMoveParser._remove_comments_and_variations(moves_text)

        # 移除结果标记
        moves_text = PGNMoveParser._remove_result_markers(moves_text)

        # 计算步数
        return PGNMoveParser._count_plies(moves_text)

    @staticmethod
    def _remove_comments_and_variations(moves_text: str) -> str:
        """移除PGN中的注释和变体"""
        # 移除花括号注释 {}
        while True:
            start = moves_text.find("{")
            if start == -1:
                break
            end = moves_text.find("}", start)
            if end == -1:
                # 不完整的注释，移除剩余部分
                moves_text = moves_text[:start]
                break
            moves_text = moves_text[:start] + " " + moves_text[end + 1 :]

        # 移除括号注释 ()
        while True:
            start = moves_text.find("(")
            if start == -1:
                break
            end = moves_text.find(")", start)
            if end == -1:
                # 不完整的注释，移除剩余部分
                moves_text = moves_text[:start]
                break
            moves_text = moves_text[:start] + " " + moves_text[end + 1 :]

        # 移除分号注释 ; 到行尾
        lines = moves_text.split("\n")
        cleaned_lines = []
        for line in lines:
            semicolon_pos = line.find(";")
            if semicolon_pos != -1:
                line = line[:semicolon_pos]
            cleaned_lines.append(line)

        return " ".join(cleaned_lines)

    @staticmethod
    def _remove_result_markers(moves_text: str) -> str:
        """移除结果标记"""
        # 移除常见的结果标记
        result_patterns = [
            r"\s*1-0\s*$",
            r"\s*0-1\s*$",
            r"\s*1/2-1/2\s*$",
            r"\s*\*\s*$",  # 未完成的对局
        ]

        for pattern in result_patterns:
            moves_text = re.sub(pattern, " ", moves_text)

        return moves_text.strip()

    @staticmethod
    def _count_plies(moves_text: str) -> int:
        """计算棋步字符串中的半着数"""
        if not moves_text.strip():
            return 0

        # 分割为令牌
        tokens = moves_text.split()
        ply_count = 0

        i = 0
        while i < len(tokens):
            token = tokens[i]

            # 跳过数字加点（如 "1.", "2." 等）
            if re.match(r"^\d+\.+$", token):
                i += 1
                continue

            # 检查是否是有效的棋步
            if PGNMoveParser._is_valid_move_token(token):
                ply_count += 1

            i += 1

        return ply_count

    @staticmethod
    def _is_valid_move_token(token: str) -> bool:
        """检查是否是有效的棋步令牌"""
        # 移除数字加点前缀（如 "1.e4" -> "e4"）
        if "." in token:
            parts = token.split(".")
            if len(parts) > 1 and parts[-1]:
                token = parts[-1]

        # 常见的无效令牌
        invalid_patterns = [
            r"^\.+$",  # 只有点
            r"^\d+$",  # 只有数字
        ]

        for pattern in invalid_patterns:
            if re.match(pattern, token):
                return False

        # 有效的棋步通常包含字母或特殊走法
        valid_patterns = [
            r"^[KQRBN]?[a-h]?[1-8]?x?[a-h][1-8](?:=?[QRBN])?[+#]?$",  # 标准走法
            r"^O-O(-O)?[+#]?$",  # 王车易位
            r"^[a-h]x[a-h][1-8](?:=?[QRBN])?[+#]?$",  # 吃过路兵
        ]

        for pattern in valid_patterns:
            if re.match(pattern, token, re.IGNORECASE):
                return True

        # 对于不符合严格模式的令牌，检查是否包含棋步特征
        if any(c.isalpha() for c in token) and len(token) <= 10:
            return True

        return False

    @staticmethod
    def extract_plycount_from_game(game_text: str) -> Optional[int]:
        """
        从游戏文本中提取plycount，优先使用标签，没有则计算
        返回: plycount或None
        """
        # 首先尝试从PlyCount标签提取
        plycount_match = re.search(r'\[PlyCount\s+"(\d+)"\]', game_text)
        if plycount_match:
            try:
                return int(plycount_match.group(1))
            except (ValueError, AttributeError):
                pass

        # 如果没有PlyCount标签，尝试计算
        try:
            return PGNMoveParser.calculate_plycount(game_text)
        except Exception as e:
            # 计算失败
            return None


class ErrorLogger:
    """错误/无效日志记录器，用于记录和保存错误/无效游戏"""

    def __init__(self, error_file_path="error.pgn", stats_file_path="error_stats.json"):
        self.error_file_path = Path(error_file_path)
        self.stats_file_path = Path(stats_file_path)
        self.error_file = None
        self.error_stats = {
            "total_errors": 0,
            "error_types": {
                "missing_result": 0,
                "missing_plycount": 0,
                "invalid_result": 0,
                "plycount_too_high": 0,
                "invalid_plycount": 0,
                "other": 0,
            },
            "error_examples": {
                "missing_result": [],
                "missing_plycount": [],
                "invalid_result": [],
                "plycount_too_high": [],
                "invalid_plycount": [],
                "other": [],
            },
        }

    def open(self):
        """打开错误/无效文件"""
        self.error_file = open(
            self.error_file_path, "w", encoding="utf-8", errors="replace"
        )

    def close(self):
        """关闭错误/无效文件并保存统计信息"""
        if self.error_file:
            self.error_file.close()
            self.error_file = None

        # 保存错误/无效统计
        with open(self.stats_file_path, "w", encoding="utf-8") as f:
            json.dump(self.error_stats, f, indent=2, ensure_ascii=False)

    def log_error(self, game_text: str, error_type: str, error_info: str = ""):
        """记录错误/无效游戏"""
        self.error_stats["total_errors"] += 1

        # 更新错误/无效类型统计
        if error_type in self.error_stats["error_types"]:
            self.error_stats["error_types"][error_type] += 1
        else:
            self.error_stats["error_types"]["other"] += 1

        # 记录错误/无效示例（最多保存10个示例）
        if error_type in self.error_stats["error_examples"]:
            examples = self.error_stats["error_examples"][error_type]
            if len(examples) < 10:
                # 提取前几行作为示例
                lines = game_text.split("\n")[:5]
                example = {
                    "preview": "\n".join(lines),
                    "error_info": error_info,
                    "timestamp": time.time(),
                }
                examples.append(example)

        # 将错误/无效游戏写入文件
        if self.error_file:
            self.error_file.write(f"# ERROR TYPE: {error_type}\n")
            self.error_file.write(f"# ERROR INFO: {error_info}\n")
            self.error_file.write(
                f"# TIMESTAMP: {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
            )
            self.error_file.write(game_text)
            if not game_text.endswith("\n\n"):
                self.error_file.write("\n\n")

    def get_stats(self) -> Dict[str, Any]:
        """获取错误/无效统计信息"""
        return self.error_stats.copy()


class PGNGameStreamer:
    """流式读取PGN文件的类"""

    def __init__(self, pgn_dir="./pgn"):
        self.pgn_dir = Path(pgn_dir)
        self.files: List[Path] = []
        self._find_pgn_files()

    def _find_pgn_files(self):
        """查找所有PGN文件（跨平台兼容，避免重复）"""
        if not self.pgn_dir.exists():
            print(f"错误: 目录 {self.pgn_dir} 不存在")
            sys.exit(1)

        # 使用集合来避免重复（特别是Windows系统）
        seen_files: Set[str] = set()
        self.files = []

        # 查找所有.pgn文件（不区分大小写）
        for file_path in self.pgn_dir.rglob("*"):
            if file_path.is_file():
                # 检查扩展名（不区分大小写）
                ext = file_path.suffix.lower()
                if ext == ".pgn":
                    # 对于Windows，使用绝对路径的小写形式作为唯一标识
                    file_key = str(file_path.resolve()).lower()
                    if file_key not in seen_files:
                        seen_files.add(file_key)
                        self.files.append(file_path)

        # 如果上面的方法找不到文件，尝试第二种方法
        if not self.files:
            # 方法2: 使用glob并手动去重
            all_files = []
            all_files.extend(self.pgn_dir.glob("**/*.pgn"))

            # 去重
            for file_path in all_files:
                file_key = str(file_path.resolve()).lower()
                if file_key not in seen_files:
                    seen_files.add(file_key)
                    self.files.append(file_path)

        # 按路径排序以保证一致性
        self.files.sort(key=lambda x: str(x).lower())

        if not self.files:
            print(f"错误: 在 {self.pgn_dir} 中未找到PGN文件")
            sys.exit(1)

        print(f"找到 {len(self.files)} 个唯一的PGN文件")
        if len(self.files) < 10:  # 如果文件不多，显示文件名
            for i, f in enumerate(self.files[:10], 1):
                print(f"  {i}. {f.name} ({f.stat().st_size / (1024 * 1024):.1f} MB)")

    def stream_games_simple(self) -> Generator[str, None, None]:
        """简化版本的流式读取，适合标准格式的PGN文件"""
        total_files = len(self.files)

        for idx, pgn_file in enumerate(self.files, 1):
            file_size = pgn_file.stat().st_size
            print(
                f"[{idx}/{total_files}] 处理文件: {pgn_file.name} ({file_size / (1024 * 1024 * 1024):.2f} GB)"
            )

            try:
                with open(pgn_file, "r", encoding="utf-8", errors="replace") as f:
                    game_lines = []
                    line_count = 0
                    game_count = 0

                    for line in f:
                        line_count += 1

                        # 检测新游戏开始
                        if line.startswith("[Event ") and game_lines:
                            game_text = "".join(game_lines)
                            game_count += 1

                            # 重置游戏行，开始新游戏
                            game_lines = [line]

                            if len(game_text) > 10:  # 确保不是空游戏
                                yield game_text
                        else:
                            game_lines.append(line)

                        # 定期报告进度
                        if line_count % 1000000 == 0:
                            print(
                                f"  已读取 {line_count:,} 行，找到 {game_count} 个游戏"
                            )

                    # 处理最后一个游戏
                    if game_lines:
                        game_text = "".join(game_lines)
                        if len(game_text) > 10:
                            game_count += 1
                            yield game_text

                    print(f"  文件完成: 共 {line_count:,} 行，{game_count} 个游戏")

            except UnicodeDecodeError:
                # 如果UTF-8失败，尝试其他编码
                print(f"  UTF-8解码失败，尝试其他编码...")
                try:
                    with open(pgn_file, "r", encoding="latin-1", errors="replace") as f:
                        game_lines = []
                        line_count = 0
                        game_count = 0

                        for line in f:
                            line_count += 1

                            if line.startswith("[Event ") and game_lines:
                                game_text = "".join(game_lines)
                                game_count += 1
                                game_lines = [line]

                                if len(game_text) > 10:
                                    yield game_text
                            else:
                                game_lines.append(line)

                        if game_lines:
                            game_text = "".join(game_lines)
                            if len(game_text) > 10:
                                game_count += 1
                                yield game_text

                        print(
                            f"  文件完成(使用latin-1): 共 {line_count:,} 行，{game_count} 个游戏"
                        )

                except Exception as e:
                    print(f"  处理文件时出错: {e}")
                    continue

            except Exception as e:
                print(f"处理文件 {pgn_file} 时出错: {e}")
                continue


class ReservoirSampler:
    """蓄水池抽样类，用于随机选择和棋"""

    def __init__(self, reservoir_size=RESERVOIR_SIZE):
        self.reservoir_size = reservoir_size
        self.reservoir: List[str] = []
        self.total_draws = 0
        self._reservoir_full = False

    def add_draw(self, game_text: str):
        """添加和棋到蓄水池"""
        self.total_draws += 1

        if not self._reservoir_full:
            # 蓄水池未满，直接添加
            self.reservoir.append(game_text)
            if len(self.reservoir) >= self.reservoir_size:
                self._reservoir_full = True
        else:
            # 蓄水池已满，随机替换
            r = random.randint(0, self.total_draws - 1)
            if r < self.reservoir_size:
                self.reservoir[r] = game_text

    def get_samples(self, num_samples: int) -> List[str]:
        """从蓄水池中获取指定数量的样本"""
        if num_samples <= 0:
            return []

        if num_samples >= len(self.reservoir):
            # 如果需要所有样本，直接返回（已经是随机样本）
            return self.reservoir.copy()
        else:
            # 随机选择指定数量的样本
            return random.sample(self.reservoir, num_samples)

    def clear(self):
        """清空蓄水池（节省内存）"""
        self.reservoir.clear()
        self.total_draws = 0
        self._reservoir_full = False


class CheckpointManager:
    """断点续传管理器"""

    def __init__(self, checkpoint_dir="checkpoints"):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)
        self.state_file = self.checkpoint_dir / "state.json"

    def save_state(
        self,
        total_games: int,
        mates_count: int,
        errors_count: int,
        current_file: str,
        computed_plycounts: int = 0,
    ):
        """保存处理状态"""
        state = {
            "total_games": total_games,
            "mates_count": mates_count,
            "errors_count": errors_count,
            "current_file": current_file,
            "computed_plycounts": computed_plycounts,
            "timestamp": time.time(),
            "seed": SEED,
            "max_plies": MAX_PLIES,
        }

        with open(self.state_file, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2)

        print(f"检查点已保存: {self.state_file}")

    def load_state(self):
        """加载处理状态"""
        if not self.state_file.exists():
            return None

        try:
            with open(self.state_file, "r", encoding="utf-8") as f:
                state = json.load(f)

            # 验证关键参数是否匹配
            if state.get("seed") != SEED or state.get("max_plies") != MAX_PLIES:
                print("警告: 检查点参数与当前设置不匹配，将重新开始")
                return None

            return state
        except Exception as e:
            print(f"加载检查点失败: {e}")
            return None

    def cleanup(self):
        """清理检查点文件"""
        if self.state_file.exists():
            self.state_file.unlink()


class GameProcessor:
    """游戏处理器，整合流式读取和抽样"""

    def __init__(self, max_plies=MAX_PLIES):
        self.max_plies = max_plies
        self.streamer = PGNGameStreamer()
        self.reservoir_sampler = ReservoirSampler()
        self.checkpoint = CheckpointManager()
        self.error_logger = ErrorLogger()

        # 统计信息
        self.total_games = 0
        self.mates_count = 0
        self.errors_count = 0
        self.computed_plycounts = 0
        self.start_time = None

        # 断点续传状态
        self.resume_state = None
        self.current_file_idx = 0

    def parse_game_info(
        self, game_text: str
    ) -> Tuple[Optional[str], Optional[int], Optional[str]]:
        """
        解析游戏信息（结果和步数）
        自动处理缺失PlyCount标签的情况
        返回: (result, plycount, error_type)
        """
        result = None
        plycount = None
        error_type = None

        # 查找Result标签
        result_match = re.search(r'\[Result\s+"([^"]+)"\]', game_text)
        if result_match:
            result = result_match.group(1)
        else:
            error_type = "missing_result"

        # 尝试获取plycount
        plycount = PGNMoveParser.extract_plycount_from_game(game_text)

        if plycount is None:
            # 计算失败
            if error_type is None:
                error_type = "missing_plycount"
            self.computed_plycounts += 1

        return result, plycount, error_type

    def validate_game(
        self, game_text: str
    ) -> Tuple[bool, Optional[str], Optional[int], Optional[str]]:
        """
        检查游戏是否有效
        返回: (is_valid, result, plycount, error_type)
        """
        result, plycount, error_type = self.parse_game_info(game_text)

        # 检查结果
        if result is None:
            return False, None, None, "missing_result"

        if result not in ["1-0", "0-1", "1/2-1/2"]:
            return False, result, plycount, "invalid_result"

        # 检查步数
        if plycount is None:
            return False, result, None, "missing_plycount"

        try:
            if int(plycount) > self.max_plies:
                return False, result, plycount, "plycount_too_high"
        except (ValueError, TypeError):
            return False, result, plycount, "invalid_plycount"

        return True, result, plycount, None

    def process_games(self):
        """主处理函数"""
        print("大文件PGN游戏筛选器 (流式处理版本 + 错误/无效记录)")
        print("=" * 60)
        print(f"配置: MAX_PLIES={MAX_PLIES}, RESERVOIR_SIZE={RESERVOIR_SIZE:,}")
        print(f"输入目录: {self.streamer.pgn_dir}")
        print(f"输出文件: games.pgn")
        print(f"错误/无效文件: error.pgn")
        print()

        self.start_time = time.time()

        # 打开错误/无效日志文件
        self.error_logger.open()

        # 尝试从检查点恢复
        self._try_resume()

        print("开始处理游戏...")
        print()

        try:
            # 第一遍：统计胜负局，收集和棋到蓄水池
            self._process_first_pass()

            # 第二遍：选择和棋
            self._process_second_pass()

            # 输出最终统计
            self._print_final_stats()

            # 清理检查点
            self.checkpoint.cleanup()

        except KeyboardInterrupt:
            print("\n\n处理被用户中断")
            self._save_checkpoint()
            print("检查点已保存，下次运行将从中断处恢复")

        except Exception as e:
            print(f"\n处理过程中发生错误: {e}")
            import traceback

            traceback.print_exc()
            self._save_checkpoint()
            print("检查点已保存")

        finally:
            # 关闭错误/无效日志文件
            self.error_logger.close()

    def _try_resume(self):
        """尝试从检查点恢复"""
        state = self.checkpoint.load_state()
        if state:
            print(f"找到检查点: {self.checkpoint.state_file}")
            print(f"  总游戏: {state['total_games']:,}")
            print(f"  胜负局: {state['mates_count']:,}")
            print(f"  错误/无效: {state['errors_count']:,}")
            print(f"  计算步数: {state.get('computed_plycounts', 0):,}")
            print(f"  当前文件: {state['current_file']}")

            choice = input("是否从检查点恢复处理？(y/n): ").strip().lower()
            if choice == "y":
                self.resume_state = state
                self.total_games = state["total_games"]
                self.mates_count = state["mates_count"]
                self.errors_count = state["errors_count"]
                self.computed_plycounts = state.get("computed_plycounts", 0)

                # 查找当前文件索引
                current_file = state["current_file"]
                for idx, file_path in enumerate(self.streamer.files):
                    if file_path.name == current_file:
                        self.current_file_idx = idx
                        break

                print(f"从文件 {current_file} 恢复处理")
            else:
                print("重新开始处理...")
                self.checkpoint.cleanup()

    def _process_first_pass(self):
        """第一遍处理：筛选游戏"""
        print("=" * 60)
        print("第一遍：筛选游戏")
        print("=" * 60)

        # 确定输出文件模式
        output_mode = "w" if self.mates_count == 0 else "a"

        # 从断点处开始处理文件
        for file_idx in range(self.current_file_idx, len(self.streamer.files)):
            pgn_file = self.streamer.files[file_idx]
            print(
                f"\n处理文件 [{file_idx + 1}/{len(self.streamer.files)}]: {pgn_file.name}"
            )

            # 记录当前处理文件
            self.current_file_idx = file_idx

            # 使用简单流式读取
            for game_text in self.streamer.stream_games_simple():
                self.total_games += 1

                # 检查游戏有效性
                valid, result, plycount, error_type = self.validate_game(game_text)

                if not valid:
                    self.errors_count += 1

                    # 记录无效游戏
                    error_info = f"Result: {result}, PlyCount: {plycount}"
                    self.error_logger.log_error(game_text, error_type, error_info)
                    continue

                # 分类处理
                if result == "1/2-1/2":
                    # 和棋：添加到蓄水池
                    self.reservoir_sampler.add_draw(game_text)
                else:
                    # 胜负局：直接写入文件
                    self.mates_count += 1
                    self._write_game_to_file(game_text, output_mode, "games.pgn")
                    output_mode = "a"  # 后续追加

                # 定期输出进度
                if self.total_games % LOG_INTERVAL == 0:
                    self._log_progress()

                # 定期保存检查点
                if self.total_games % CHECKPOINT_INTERVAL == 0:
                    self._save_checkpoint()

            # 文件处理完成，保存检查点
            self._save_checkpoint()

        print("\n" + "=" * 60)
        print("第一遍完成")
        print("=" * 60)
        print(f"总计游戏: {self.total_games:,}")
        print(f"胜负局: {self.mates_count:,}")
        print(f"和棋: {self.reservoir_sampler.total_draws:,}")
        print(f"错误/无效: {self.errors_count:,}")
        print(f"计算步数: {self.computed_plycounts:,}")
        print(f"处理时间: {time.time() - self.start_time:.1f} 秒")

    def _process_second_pass(self):
        """第二遍处理：选择和棋"""
        print("\n" + "=" * 60)
        print("第二遍：选择和棋")
        print("=" * 60)

        if self.mates_count == 0:
            print("没有胜负局，不选择和棋")
            return

        # 计算需要选择和棋的数量
        draws_needed = min(
            int(self.mates_count / 2), self.reservoir_sampler.total_draws
        )

        if draws_needed == 0:
            print("无需选择和棋")
            return

        print(f"胜负局: {self.mates_count:,}")
        print(f"需要选择和棋: {draws_needed:,} 个")
        print(f"蓄水池中可选和棋: {self.reservoir_sampler.total_draws:,}")
        print(f"蓄水池大小: {len(self.reservoir_sampler.reservoir):,}")

        # 从蓄水池中抽样
        print("正在选择和棋...")
        selected_draws = self.reservoir_sampler.get_samples(draws_needed)

        # 写入选中的和棋
        print(f"写入 {len(selected_draws):,} 个和棋...")
        for i, game_text in enumerate(selected_draws, 1):
            self._write_game_to_file(game_text, "a", "games.pgn")

            if i % 1000 == 0:
                print(f"  已写入 {i:,} 个和棋")

        print(f"和棋选择完成: 已写入 {len(selected_draws):,} 个和棋")

        # 清理蓄水池以释放内存
        self.reservoir_sampler.clear()

    def _write_game_to_file(
        self, game_text: str, mode: str = "a", filename: str = "games.pgn"
    ):
        """将游戏写入文件"""
        with open(filename, mode, encoding="utf-8", errors="replace") as f:
            f.write(game_text)
            if not game_text.endswith("\n\n"):
                f.write("\n\n")

    def _log_progress(self):
        """记录处理进度"""
        elapsed = time.time() - self.start_time
        if elapsed == 0:
            elapsed = 0.001

        games_per_sec = self.total_games / elapsed

        # 内存使用
        process = psutil.Process()
        mem_mb = process.memory_info().rss / 1024 / 1024

        print(
            f"[进度] 游戏: {self.total_games:,} | "
            f"速度: {games_per_sec:.1f}/秒 | "
            f"时间: {elapsed:.0f}秒 | "
            f"内存: {mem_mb:.1f}MB | "
            f"胜负: {self.mates_count:,} | "
            f"和棋: {self.reservoir_sampler.total_draws:,} | "
            f"错误/无效: {self.errors_count:,} | "
            f"计算步数: {self.computed_plycounts:,}"
        )

    def _save_checkpoint(self):
        """保存检查点"""
        if self.current_file_idx < len(self.streamer.files):
            current_file = self.streamer.files[self.current_file_idx].name
        else:
            current_file = "COMPLETED"

        self.checkpoint.save_state(
            total_games=self.total_games,
            mates_count=self.mates_count,
            errors_count=self.errors_count,
            current_file=current_file,
            computed_plycounts=self.computed_plycounts,
        )

    def _print_final_stats(self):
        """打印最终统计信息"""
        elapsed = time.time() - self.start_time

        # 计算时间格式
        hours = int(elapsed // 3600)
        minutes = int((elapsed % 3600) // 60)
        seconds = int(elapsed % 60)

        # 检查输出文件大小
        games_size = 0
        if os.path.exists("games.pgn"):
            games_size = os.path.getsize("games.pgn")

        errors_size = 0
        if os.path.exists("error.pgn"):
            errors_size = os.path.getsize("error.pgn")

        # 获取错误/无效统计
        error_stats = self.error_logger.get_stats()

        print("\n" + "=" * 60)
        print("处理完成!")
        print("=" * 60)
        print(f"总计处理游戏: {self.total_games:,}")
        print(f"胜负局数量: {self.mates_count:,}")
        print(f"和棋总数: {self.reservoir_sampler.total_draws:,}")
        print(
            f"选择和棋数量: {min(int(self.mates_count / 2), self.reservoir_sampler.total_draws):,}"
        )
        print(f"错误/无效游戏: {self.errors_count:,}")
        print(f"计算步数的游戏: {self.computed_plycounts:,}")
        print(f"输出文件: games.pgn ({games_size / (1024 * 1024):.2f} MB)")
        print(f"错误/无效文件: error.pgn ({errors_size / (1024 * 1024):.2f} MB)")
        print(f"错误/无效统计: error_stats.json")
        print(f"总耗时: {hours:02d}:{minutes:02d}:{seconds:02d}")
        print(
            f"平均速度: {self.total_games / elapsed:.1f} 游戏/秒"
            if elapsed > 0
            else "速度: N/A"
        )

        # 质量统计
        if self.total_games > 0:
            valid_games = self.mates_count + self.reservoir_sampler.total_draws
            valid_rate = (valid_games / self.total_games) * 100
            computed_rate = (
                (self.computed_plycounts / self.total_games) * 100
                if self.computed_plycounts > 0
                else 0
            )

            print(f"\n质量统计:")
            print(f"  有效游戏率: {valid_rate:.1f}%")
            print(f"  计算步数率: {computed_rate:.1f}%")

            if self.reservoir_sampler.total_draws > 0:
                mate_draw_ratio = self.mates_count / self.reservoir_sampler.total_draws
                print(f"  胜负:和棋比例: {mate_draw_ratio:.2f}:1")

        # 错误/无效类型统计
        if error_stats["total_errors"] > 0:
            print(f"\n错误/无效类型统计:")
            for error_type, count in error_stats["error_types"].items():
                if count > 0:
                    percentage = (count / error_stats["total_errors"]) * 100
                    print(f"  {error_type}: {count:,} ({percentage:.1f}%)")


def test_plycount_calculation():
    """测试步数计算功能"""
    test_games = [
        # 简单游戏
        """[Event "Test Game"]
[Site "?"]
[Date "2023.01.01"]
[Round "?"]
[White "Player1"]
[Black "Player2"]
[Result "1-0"]

1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 4. Ba4 Nf6 5. O-O Be7 6. Re1 b5 7. Bb3 d6 8. c3 O-O 9. h3 Nb8 10. d4 Nbd7 11. c4 c6 12. cxb5 axb5 13. Nc3 Bb7 14. Bg5 b4 15. Nb1 h6 16. Bh4 c5 17. dxe5 Nxe4 18. Bxe7 Qxe7 19. exd6 Qf6 20. Nbd2 Nxd6 21. Nc4 Nxc4 22. Bxc4 Nb6 23. Ne5 Rae8 24. Bxf7+ Rxf7 25. Nxf7 Rxe1+ 26. Qxe1 Kxf7 27. Qe3 Qg5 28. Qxg5 hxg5 29. b3 Ke6 30. a3 Kd6 31. axb4 cxb4 32. Ra5 Nd5 33. f3 Bc8 34. Kf2 Bf5 35. Ra7 g6 36. Ra6+ Kc5 37. Ke1 Nf4 38. g3 Nxh3 39. Kd2 Kb5 40. Rd6 Kc5 41. Ra6 Nf2 42. g4 Bd3 43. Re6 1-0""",
        # 带有PlyCount标签的游戏
        """[Event "Test Game 2"]
[Site "?"]
[Date "2023.01.02"]
[Round "?"]
[White "Player1"]
[Black "Player2"]
[Result "1/2-1/2"]
[PlyCount "60"]

1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 4. Ba4 Nf6 5. O-O Be7 6. Re1 b5 7. Bb3 d6 8. c3 O-O 9. h3 Nb8 10. d4 Nbd7 11. c4 c6 12. cxb5 axb5 13. Nc3 Bb7 14. Bg5 b4 15. Nb1 h6 16. Bh4 c5 17. dxe5 Nxe4 18. Bxe7 Qxe7 19. exd6 Qf6 20. Nbd2 Nxd6 21. Nc4 Nxc4 22. Bxc4 Nb6 23. Ne5 Rae8 24. Bxf7+ Rxf7 25. Nxf7 Rxe1+ 26. Qxe1 Kxf7 27. Qe3 Qg5 28. Qxg5 hxg5 29. b3 Ke6 30. a3 Kd6 1/2-1/2""",
        # 没有PlyCount标签的游戏
        """[Event "Test Game 3"]
[Site "?"]
[Date "2023.01.03"]
[Round "?"]
[White "Player1"]
[Black "Player2"]
[Result "0-1"]

1. e4 c5 2. Nf3 d6 3. d4 cxd4 4. Nxd4 Nf6 5. Nc3 a6 6. Bg5 e6 7. f4 Qb6 8. Qd2 Qxb2 9. Rb1 Qa3 10. e5 dxe5 11. fxe5 Nfd7 12. Bc4 Qa5 13. Rb3 Bb4 14. Qf2 O-O 15. Bxe6 fxe6 16. Qh4 h6 17. Qg4 Kh7 18. Nxe6 Qa4 19. Qh5 Qc6 20. Bf6 gxf6 21. Qxh6+ Kg8 22. Qh8+ Kf7 23. Qh5+ Kg8 24. Qh8+ Kf7 25. Qh5+ Ke7 26. Qf7+ Kd8 27. Qxf6+ Kc7 28. Qf7+ Kb8 29. Nxd8 Nc6 30. Nxc6+ bxc6 31. Qxc7# 0-1""",
        # 带有注释的游戏
        """[Event "Test Game 4"]
[Site "?"]
[Date "2023.01.04"]
[Round "?"]
[White "Player1"]
[Black "Player2"]
[Result "1-0"]

1. e4 {This is a comment} e5 2. Nf3 Nc6 {Another comment} 3. Bb5 {The Ruy Lopez} a6 4. Ba4 Nf6 5. O-O Be7 6. Re1 b5 7. Bb3 d6 8. c3 O-O 9. h3 (9. d4 Bg4 10. d5) 9... Nb8 10. d4 Nbd7 11. c4 c6 12. cxb5 axb5 13. Nc3 Bb7 14. Bg5 b4 15. Nb1 h6 16. Bh4 c5 17. dxe5 Nxe4 18. Bxe7 Qxe7 19. exd6 Qf6 20. Nbd2 Nxd6 21. Nc4 Nxc4 22. Bxc4 Nb6 23. Ne5 Rae8 24. Bxf7+ Rxf7 25. Nxf7 Rxe1+ 26. Qxe1 Kxf7 27. Qe3 Qg5 28. Qxg5 hxg5 29. b3 Ke6 30. a3 Kd6 31. axb4 cxb4 32. Ra5 Nd5 33. f3 Bc8 34. Kf2 Bf5 35. Ra7 g6 36. Ra6+ Kc5 37. Ke1 Nf4 38. g3 Nxh3 39. Kd2 Kb5 40. Rd6 Kc5 41. Ra6 Nf2 42. g4 Bd3 43. Re6 1-0""",
    ]

    print("测试步数计算功能...")
    print("=" * 40)

    for i, game_text in enumerate(test_games, 1):
        plycount = PGNMoveParser.extract_plycount_from_game(game_text)
        print(f"游戏 {i}: plycount = {plycount}")

        # 验证计算
        if plycount is not None:
            calculated = PGNMoveParser.calculate_plycount(game_text)
            print(f"  计算值: {calculated}")

        print()


def main():
    """主函数"""
    # 显示系统信息
    print("PGN游戏筛选器 (支持自动计算步数和错误/无效记录)")
    print("=" * 60)
    print(f"Python版本: {sys.version}")
    print(f"平台: {sys.platform}")
    print(f"内存总量: {psutil.virtual_memory().total / (1024**3):.1f} GB")
    print()

    # 测试步数计算功能（可选）
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        test_plycount_calculation()
        return

    # 处理游戏
    processor = GameProcessor()
    processor.process_games()


if __name__ == "__main__":
    main()
