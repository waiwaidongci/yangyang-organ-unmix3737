"""自举工具的规则测试。

直接运行：
  /workspace/.venv/bin/python -I analysis/tests/test_bootstrap.py
"""

import csv
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ANALYSIS_DIR = Path(__file__).resolve().parents[1]
WORKSPACE_DIR = ANALYSIS_DIR.parent
sys.path.insert(0, str(ANALYSIS_DIR))
sys.path.insert(0, str(WORKSPACE_DIR / "backend"))

from src.domain.fixtures import demo_rows  # noqa: E402
from src.domain.schema import MIC_KEYS  # noqa: E402

import bootstrap_ica as boot  # noqa: E402

PYTHON = str(WORKSPACE_DIR / ".venv" / "bin" / "python")
SCRIPT = str(ANALYSIS_DIR / "bootstrap_ica.py")


def write_demo_csv(path: Path) -> Path:
    import csv

    rows = demo_rows()
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return path


def write_rank2_csv(path: Path, n_cal: int = 160, n_hold: int = 80) -> Path:
    """有效秩为 2：前两路是独立信号，后两路恒定，任何行重排后秩仍为 2。

    3 成分必然触发 separate() 的秩不足错误；2 成分可正常拟合。
    """
    rng = np.random.default_rng(7)
    total = n_cal + n_hold
    signals = np.column_stack((
        rng.laplace(size=total),
        np.sin(2 * np.pi * 31 * np.arange(total) / 4000.0),
    ))
    mixing = rng.normal(size=(2, 2))
    head = signals @ mixing.T
    head = head / np.max(np.abs(head)) * 0.5
    microphones = np.column_stack((
        head,
        np.full((total, 1), 0.01),
        np.full((total, 1), -0.02),
    ))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["id", "sample_index", "sample_rate_hz", "split", *MIC_KEYS])
        for index, sample in enumerate(microphones):
            split = "calibration" if index < n_cal else "holdout"
            writer.writerow([
                f"R2-{index:04d}", index, 4000, split, *[f"{v:.8f}" for v in sample]
            ])
    return path


def run_cli(csv_path: Path, state_dir: Path, *extra: str) -> dict:
    completed = subprocess.run(
        [PYTHON, "-I", SCRIPT, "--csv", str(csv_path),
         "--state-dir", str(state_dir), *extra],
        check=True, capture_output=True, text=True,
    )
    return json.loads(completed.stdout[completed.stdout.index("{"):])


class BootstrapRulesTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = Path(tempfile.mkdtemp(prefix="bootstrap-test-"))
        self.csv_path = write_demo_csv(self.temp / "demo.csv")

    def tearDown(self) -> None:
        shutil.rmtree(self.temp, ignore_errors=True)

    def test_determinism_and_paired_delta(self) -> None:
        first = run_cli(self.csv_path, self.temp / "run_a",
                        "--rounds", "40", "--block-size", "10", "--seed", "123")
        second = run_cli(self.csv_path, self.temp / "run_b",
                         "--rounds", "40", "--block-size", "10", "--seed", "123")
        state_a = json.loads((self.temp / "run_a" / boot.STATE_NAME).read_text())
        state_b = json.loads((self.temp / "run_b" / boot.STATE_NAME).read_text())
        self.assertEqual(state_a["rounds"], state_b["rounds"])

        records = list(state_a["rounds"].values())
        self.assertEqual(len(records), 40)
        for record in records:
            if record["status"] == "ok":
                self.assertAlmostEqual(
                    record["delta_pct"],
                    record["residual_2_pct"] - record["residual_3_pct"],
                    places=12,
                )
                self.assertTrue(np.isfinite(record["residual_2_pct"]))
                self.assertTrue(np.isfinite(record["residual_3_pct"]))
        interval = first["delta_interval"]
        self.assertTrue(interval["reported"])
        self.assertLess(interval["lower_pct"], interval["upper_pct"])

    def test_resume_matches_one_shot(self) -> None:
        # 先跑 15 轮模拟中断，再续跑到 40 轮；两轮参数必须完全一致。
        run_cli(self.csv_path, self.temp / "resume",
                "--rounds", "15", "--block-size", "10", "--seed", "99")
        resumed_report = run_cli(self.csv_path, self.temp / "resume",
                                 "--rounds", "40", "--block-size", "10", "--seed", "99")
        self.assertEqual(resumed_report["rounds_decided"], 40)
        self.assertEqual(resumed_report["rounds_pending"], 0)

        run_cli(self.csv_path, self.temp / "oneshot",
                "--rounds", "40", "--block-size", "10", "--seed", "99")
        resumed = json.loads((self.temp / "resume" / boot.STATE_NAME).read_text())
        one_shot = json.loads((self.temp / "oneshot" / boot.STATE_NAME).read_text())
        self.assertEqual(resumed["rounds"], one_shot["rounds"])

    def test_rank_deficient_rounds_fail_without_extra_draws(self) -> None:
        rank2_csv = write_rank2_csv(self.temp / "rank2.csv")
        # 有效秩恒为 2：块自举的任意行重排都无法恢复，3 成分每轮都秩不足，
        # 2 成分正常 → 每轮单侧失败、整轮记失败、不补抽、不增加轮次。
        strict_report = run_cli(rank2_csv, self.temp / "rank2-strict",
                                "--rounds", "12", "--block-size", "1")
        self.assertEqual(strict_report["rounds_total"], 12)
        self.assertEqual(strict_report["rounds_decided"], 12)
        self.assertEqual(strict_report["valid_rounds"], 0)
        self.assertEqual(strict_report["failed_rounds"], 12)
        self.assertEqual(
            strict_report["fail_reason_counts"].get(boot.FAIL_RANK), 12
        )
        self.assertFalse(strict_report["delta_interval"]["reported"])

        # 换块长结论不变：失败轮照常定论，有效+失败恒等于总轮数。
        loose_report = run_cli(rank2_csv, self.temp / "rank2-loose",
                               "--rounds", "12", "--block-size", "8")
        self.assertEqual(loose_report["rounds_decided"], 12)
        self.assertEqual(loose_report["failed_rounds"], 12)
        self.assertEqual(
            loose_report["fail_reason_counts"].get(boot.FAIL_RANK), 12
        )

    def test_non_convergence_is_failure(self) -> None:
        report = run_cli(self.csv_path, self.temp / "noiter",
                         "--rounds", "6", "--max-iter", "1")
        self.assertEqual(report["failed_rounds"], 6)
        self.assertIn(boot.FAIL_CONVERGENCE, report["fail_reason_counts"])
        self.assertFalse(report["delta_interval"]["reported"])

    def test_single_valid_round_reports_no_interval(self) -> None:
        report = run_cli(self.csv_path, self.temp / "one",
                         "--rounds", "1", "--block-size", "10")
        self.assertEqual(report["valid_rounds"], 1)
        self.assertFalse(report["delta_interval"]["reported"])
        self.assertIn("不足两轮", report["delta_interval"]["reason"])

    def test_incomplete_state_withholds_interval(self) -> None:
        run_cli(self.csv_path, self.temp / "partial",
                "--rounds", "10", "--seed", "5")
        state = boot.load_state(self.temp / "partial" / boot.STATE_NAME)
        config = boot.BootstrapConfig(
            rounds=20, block_size=boot.DEFAULT_BLOCK, seed=5,
            max_iter=boot.DEFAULT_MAX_ITER,
            n_calibration=160, n_holdout=80,
        )
        report = boot.build_report(state, config)
        self.assertFalse(report["complete"])
        self.assertFalse(report["delta_interval"]["reported"])

    def test_rank_failure_at_function_level(self) -> None:
        # 有效秩 2：两列信号 + 两列恒定，3 成分被 separate() 的秩阈值直接拒绝，
        # 2 成分可拟合；该相关性对任意行重排都成立，自举后秩不变。
        rng = np.random.default_rng(11)
        signals = rng.normal(size=(120, 2))
        base = np.column_stack((
            signals,
            np.full((120, 1), 0.05),
            np.full((120, 1), -0.03),
        ))
        centered = base[:80] - base[:80].mean(axis=0)
        singular = np.linalg.svd(centered, compute_uv=False)
        floor = max(1e-8 * singular[0], 1e-10 * np.sqrt(80))
        self.assertEqual(int((singular > floor).sum()), 2)

        residual3, reason3, _ = boot.holdout_relative_residual(
            base, calibration_count=80, n_components=3, max_iter=600
        )
        self.assertEqual(reason3, boot.FAIL_RANK)
        self.assertFalse(np.isfinite(residual3))
        residual2, reason2, _ = boot.holdout_relative_residual(
            base, calibration_count=80, n_components=2, max_iter=600
        )
        self.assertIsNone(reason2)
        self.assertTrue(np.isfinite(residual2))

        config = boot.BootstrapConfig(
            rounds=4, block_size=10, seed=3, max_iter=600,
            n_calibration=80, n_holdout=40,
        )
        for round_index in range(4):
            record = boot.run_one_round(round_index, base, 80, config)
            self.assertEqual(record["status"], "failed")
            self.assertEqual(record["fail_reasons"]["3"], boot.FAIL_RANK)
            self.assertIsNone(record["delta_pct"])  # 只算出一侧也不给增量

    def test_concurrent_runs_do_not_double_count(self) -> None:
        state_dir = self.temp / "concurrent"
        args = [
            PYTHON, "-I", SCRIPT, "--csv", str(self.csv_path),
            "--state-dir", str(state_dir),
            "--rounds", "60", "--block-size", "10", "--seed", "777",
            "--workers", "1",
        ]
        processes = [subprocess.Popen(args) for _ in range(3)]
        codes = [process.wait() for process in processes]
        self.assertEqual(codes, [0, 0, 0])
        state = json.loads((state_dir / boot.STATE_NAME).read_text())
        self.assertEqual(len(state["rounds"]), 60)
        report = json.loads((state_dir / boot.REPORT_NAME).read_text())
        self.assertEqual(report["rounds_decided"], 60)
        self.assertTrue(report["delta_interval"]["reported"])

        # 并发结果与单进程一次跑完逐轮一致。
        run_cli(self.csv_path, self.temp / "serial",
                "--rounds", "60", "--block-size", "10", "--seed", "777")
        serial = json.loads((self.temp / "serial" / boot.STATE_NAME).read_text())
        self.assertEqual(state["rounds"], serial["rounds"])

    def test_resample_keeps_segment_lengths_and_boundary(self) -> None:
        matrix, cal = boot.load_microphones(self.csv_path)
        rng = np.random.default_rng(42)
        resampled = boot.resample_segments(matrix, cal, 10, rng)
        self.assertEqual(resampled.shape, matrix.shape)
        # 重抽只在各自段内取索引，校准块不会混入留样点。
        cal_rows = {tuple(row) for row in matrix[:cal]}
        hold_rows = {tuple(row) for row in matrix[cal:]}
        for row in resampled[:cal]:
            self.assertIn(tuple(row), cal_rows)
        for row in resampled[cal:]:
            self.assertIn(tuple(row), hold_rows)
        self.assertTrue(hold_rows.isdisjoint(cal_rows))


if __name__ == "__main__":
    unittest.main(verbosity=2)
