"""成对连续块有放回自举：比较 2 成分与 3 成分 FastICA 的留样相对残差增量。

口径与 backend/src/domain 完全一致（同一 FastICA 参数、秩阈值与相对残差公式）：
- 校准段、留样段分别在各自内部有放回抽取连续块，拼接回原段长度；
- 每轮同一个重抽样同时喂给 2 成分与 3 成分（成对设计）；
- 只用重抽后的校准块拟合，留样块只用于评分；
- 任一侧秩不足或未收敛，整轮记失败，不补抽、不占额外轮次；
- 固定种子、逐轮独立 RNG，中断后续跑只补缺轮，结果与一次跑完一致；
- 状态按轮号合并并加文件锁，并发/重试不会重复计轮；
- 全部轮次结束后才输出成对增量的 95% 分位区间与失败数；有效轮不足两轮不报区间。
"""

from __future__ import annotations

import argparse
import csv
import fcntl
import json
import os
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# 复用仓库内的领域实现，保证算法口径不漂移。
BACKEND_DIR = Path(__file__).resolve().parents[1] / "backend"
sys.path.insert(0, str(BACKEND_DIR))

from sklearn.exceptions import ConvergenceWarning  # noqa: E402

from src.domain.diagnostics import rms  # noqa: E402
from src.domain.schema import CALIBRATION, MIC_KEYS  # noqa: E402
from src.domain.separation import separate  # noqa: E402

DEFAULT_ROUNDS = 1000
DEFAULT_BLOCK = 10
DEFAULT_MAX_ITER = 600
DEFAULT_SEED = 20261126
COMPONENT_PAIR = (2, 3)
STATE_NAME = "bootstrap_state.json"
REPORT_NAME = "bootstrap_report.json"

# 失败原因分类（与整轮失败的两类硬规则对应，另保留数值异常兜底）。
FAIL_RANK = "rank_deficient"
FAIL_CONVERGENCE = "not_converged"
FAIL_NUMERIC = "numeric_error"


@dataclass(frozen=True)
class BootstrapConfig:
    rounds: int
    block_size: int
    seed: int
    max_iter: int
    n_calibration: int
    n_holdout: int

    def as_dict(self) -> dict:
        # 注意：rounds 是运行目标而非自举口径，不写入状态配置，
        # 这样中断后续跑扩大目标轮数不会触发配置冲突；其余参数变化仍被拒绝。
        return {
            "block_size": self.block_size,
            "seed": self.seed,
            "max_iter": self.max_iter,
            "n_calibration": self.n_calibration,
            "n_holdout": self.n_holdout,
            "component_pair": list(COMPONENT_PAIR),
        }


def load_microphones(csv_path: Path) -> tuple[np.ndarray, int]:
    """读取四麦 CSV，按采样点排序，返回观测矩阵和校准点数。"""
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    if not rows:
        raise ValueError("CSV 为空，无法自举。")
    missing = [key for key in MIC_KEYS if key not in rows[0]]
    if missing:
        raise ValueError(f"CSV 缺少麦通道列：{', '.join(missing)}。")
    if "split" not in rows[0] or "sample_index" not in rows[0]:
        raise ValueError("CSV 缺少 split 或 sample_index 列。")
    rows.sort(key=lambda row: int(row["sample_index"]))
    matrix = np.asarray(
        [[float(row[key]) for key in MIC_KEYS] for row in rows], dtype=float
    )
    if not np.isfinite(matrix).all():
        raise ValueError("CSV 存在非有限数值，请先修复采样。")
    calibration_count = sum(row["split"].strip() == CALIBRATION for row in rows)
    if not (0 < calibration_count < len(rows)):
        raise ValueError("校准段或留样段为空。")
    return matrix, calibration_count


def block_starts(rng: np.random.Generator, length: int, block_size: int) -> np.ndarray:
    """有放回抽取连续块起点，块数向上取整后截断回原长度。"""
    n_blocks = int(np.ceil(length / block_size))
    return rng.integers(0, length - block_size + 1, size=n_blocks)


def resample_segments(
    microphones: np.ndarray,
    calibration_count: int,
    block_size: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """两段各自在段内做连续块有放回自举，拼接成等长的新矩阵。"""
    pieces = []
    for offset, length in ((0, calibration_count),
                           (calibration_count, len(microphones) - calibration_count)):
        starts = block_starts(rng, length, block_size)
        indices = np.concatenate(
            [np.arange(start, start + block_size) for start in starts]
        )[:length]
        pieces.append(microphones[offset + indices])
    return np.vstack(pieces)


def holdout_relative_residual(
    resampled: np.ndarray, calibration_count: int, n_components: int, max_iter: int
) -> tuple[float, str | None, int]:
    """拟合一次并返回（留样相对残差%，失败原因，迭代次数）。

    失败原因非空即表示该侧未成功：秩不足、未收敛或出现非有限数值。
    """
    try:
        with warnings.catch_warnings():
            # 未收敛时 sklearn 只发 ConvergenceWarning，这里按规则转成失败。
            warnings.simplefilter("error", ConvergenceWarning)
            separation = separate(
                resampled, calibration_count, n_components, max_iter
            )
    except ValueError as error:
        message = str(error)
        if "有效秩" in message:
            return float("nan"), FAIL_RANK, 0
        return float("nan"), FAIL_NUMERIC, 0
    except ConvergenceWarning:
        return float("nan"), FAIL_CONVERGENCE, max_iter
    except (FloatingPointError, RuntimeError, OverflowError):
        return float("nan"), FAIL_NUMERIC, 0

    if not np.isfinite(separation.residual).all():
        return float("nan"), FAIL_NUMERIC, separation.iterations

    calibration_scale = float(
        rms(resampled[:calibration_count] - separation.mean)
    )
    holdout_error = float(rms(separation.residual[calibration_count:]))
    return 100.0 * holdout_error / calibration_scale, None, separation.iterations


def run_one_round(
    round_index: int,
    microphones: np.ndarray,
    calibration_count: int,
    config: BootstrapConfig,
) -> dict:
    """一轮成对自举：同一个重抽样分别拟合 2、3 成分。"""
    # 每轮一条独立派生 RNG，轮号固定则抽块序列固定，续跑与一次跑完同分布同实现。
    rng = np.random.default_rng(
        np.random.SeedSequence([config.seed, round_index]).entropy
    )
    resampled = resample_segments(
        microphones, calibration_count, config.block_size, rng
    )
    results = {}
    for n_components in COMPONENT_PAIR:
        residual, reason, iterations = holdout_relative_residual(
            resampled, calibration_count, n_components, config.max_iter
        )
        results[n_components] = {
            "holdout_residual_pct": None if reason else residual,
            "fail_reason": reason,
            "iterations": iterations,
        }

    failures = {
        str(k): value["fail_reason"]
        for k, value in results.items()
        if value["fail_reason"] is not None
    }
    # 只有两侧都拿到残差才算有效；只算出一侧仍按缺轮/失败处理。
    success = not failures
    return {
        "round": round_index,
        "status": "ok" if success else "failed",
        "fail_reasons": failures,
        "residual_2_pct": results[2]["holdout_residual_pct"],
        "residual_3_pct": results[3]["holdout_residual_pct"],
        "delta_pct": (
            results[2]["holdout_residual_pct"] - results[3]["holdout_residual_pct"]
            if success
            else None
        ),
        "iterations_2": results[2]["iterations"],
        "iterations_3": results[3]["iterations"],
    }


def load_state(state_path: Path) -> dict:
    if not state_path.exists():
        return {"config": None, "rounds": {}}
    with state_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    data.setdefault("rounds", {})
    return data


def merge_rounds(state_path: Path, config: BootstrapConfig, batch: list[dict]) -> dict:
    """加文件锁的读-改-写：按轮号去重，并发进程/重试不会重复计轮。"""
    state_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = state_path.with_suffix(".lock")
    with lock_path.open("a+") as lock_handle:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        state = load_state(state_path)
        if state["config"] is None:
            state["config"] = config.as_dict()
        elif state["config"] != config.as_dict():
            raise ValueError(
                "状态文件的自举配置与当前参数不一致；如要换参数请使用 --reset 或另选 --state-dir。"
            )
        for record in batch:
            key = str(record["round"])
            # 已定论的轮次保持原值，任何重试都不覆盖、不重复计数。
            state["rounds"].setdefault(key, record)
        atomic_write_json(state_path, state)
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
    return state


def atomic_write_json(path: Path, payload: dict) -> None:
    # 临时名带 PID 与随机后缀，避免并发进程互相删除对方的临时文件。
    temporary = path.with_name(
        f"{path.name}.tmp.{os.getpid()}.{np.random.randint(1_000_000)}"
    )
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def missing_rounds(state: dict, total: int) -> list[int]:
    decided = {int(key) for key in state.get("rounds", {})}
    return [index for index in range(total) if index not in decided]


def percentile_interval(deltas: np.ndarray) -> dict:
    if len(deltas) < 2:
        return {"reported": False, "reason": "有效轮不足两轮，不报区间。"}
    lower, median, upper = np.percentile(deltas, [2.5, 50.0, 97.5])
    return {
        "reported": True,
        "method": "成对自举增量的 2.5%/97.5% 分位（百分位法 95% 区间）",
        "lower_pct": float(lower),
        "median_pct": float(median),
        "upper_pct": float(upper),
        "mean_pct": float(np.mean(deltas)),
    }


def build_report(state: dict, config: BootstrapConfig) -> dict:
    records = [state["rounds"][str(i)] for i in range(config.rounds)
               if str(i) in state["rounds"]]
    finished = len(records)
    valid = [record for record in records if record["status"] == "ok"]
    failed = [record for record in records if record["status"] != "ok"]
    deltas = np.asarray([record["delta_pct"] for record in valid], dtype=float)

    reason_counts: dict[str, int] = {}
    for record in failed:
        for reason in record["fail_reasons"].values():
            reason_counts[reason] = reason_counts.get(reason, 0) + 1

    complete = finished >= config.rounds
    report = {
        "complete": complete,
        "config": config.as_dict(),
        "rounds_total": config.rounds,
        "rounds_decided": finished,
        "rounds_pending": config.rounds - finished,
        "valid_rounds": len(valid),
        "failed_rounds": len(failed),
        "fail_reason_counts": reason_counts,
        # 规则：未全部跑完或有效轮不足两轮时，区间字段不给出。
        "delta_interval": percentile_interval(deltas) if complete else {
            "reported": False,
            "reason": "自举尚未全部跑完，暂不报区间。",
        },
    }
    if complete and len(valid) >= 2:
        report["paired_delta_pct"] = [float(value) for value in deltas]
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="2 成分相对 3 成分的留样相对残差增量：成对连续块自举。"
    )
    parser.add_argument("--csv", type=Path, required=True, help="四麦 CSV 路径")
    parser.add_argument("--state-dir", type=Path, default=Path(__file__).parent / "state",
                        help="断点状态与报告目录（默认 analysis/state）")
    parser.add_argument("--rounds", type=int, default=DEFAULT_ROUNDS, help="自举轮数（默认 1000）")
    parser.add_argument("--block-size", type=int, default=DEFAULT_BLOCK,
                        help="连续块采样点数，两段共用（默认 10 点）")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="固定主种子（默认 20261126）")
    parser.add_argument("--max-iter", type=int, default=DEFAULT_MAX_ITER, help="FastICA 迭代上限（默认 600）")
    parser.add_argument("--workers", type=int, default=1, help="并发工作进程数（默认 1）")
    parser.add_argument("--flush-every", type=int, default=20, help="每多少轮落盘一次（默认 20）")
    parser.add_argument("--reset", action="store_true", help="开跑前清空同目录旧状态")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.rounds < 1:
        raise SystemExit("--rounds 必须 >= 1。")
    if args.block_size < 1:
        raise SystemExit("--block-size 必须 >= 1。")
    if args.workers < 1:
        raise SystemExit("--workers 必须 >= 1。")
    if args.flush_every < 1:
        raise SystemExit("--flush-every 必须 >= 1。")


def main() -> int:
    args = parse_args()
    validate_args(args)
    microphones, calibration_count = load_microphones(args.csv)
    holdout_count = len(microphones) - calibration_count
    if args.block_size > min(calibration_count, holdout_count):
        raise SystemExit(
            f"--block-size={args.block_size} 超过最短段长度 "
            f"(校准 {calibration_count} / 留样 {holdout_count})。"
        )

    config = BootstrapConfig(
        rounds=args.rounds,
        block_size=args.block_size,
        seed=args.seed,
        max_iter=args.max_iter,
        n_calibration=calibration_count,
        n_holdout=holdout_count,
    )
    state_dir = args.state_dir
    state_dir.mkdir(parents=True, exist_ok=True)
    state_path = state_dir / STATE_NAME
    report_path = state_dir / REPORT_NAME

    if args.reset and state_path.exists():
        state_path.unlink()

    state = load_state(state_path)
    if state["config"] is not None and state["config"] != config.as_dict():
        raise SystemExit(
            "旧状态配置与当前参数不一致。使用 --reset 清除，或换一个 --state-dir。"
        )

    pending = missing_rounds(state, config.rounds)
    if args.workers == 1:
        batch: list[dict] = []
        for order, round_index in enumerate(pending, 1):
            record = run_one_round(round_index, microphones, calibration_count, config)
            batch.append(record)
            if len(batch) >= args.flush_every or order == len(pending):
                state = merge_rounds(state_path, config, batch)
                batch.clear()
                print(f"已定论 {len(state['rounds'])}/{config.rounds} 轮", flush=True)
    else:
        # 进程级并发：任务只取缺轮，落盘按轮号去重，重试不产生重复轮。
        from concurrent.futures import ProcessPoolExecutor, as_completed

        batch = []
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(
                    run_one_round, round_index, microphones, calibration_count, config
                ): round_index
                for round_index in pending
            }
            for order, future in enumerate(as_completed(futures), 1):
                batch.append(future.result())
                if len(batch) >= args.flush_every or order == len(pending):
                    state = merge_rounds(state_path, config, batch)
                    batch.clear()
                    print(f"已定论 {len(state['rounds'])}/{config.rounds} 轮", flush=True)

    state = load_state(state_path)
    report = build_report(state, config)
    atomic_write_json(report_path, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
