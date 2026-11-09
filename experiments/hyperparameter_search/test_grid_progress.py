from pathlib import Path

import numpy as np
import pytest

from experiments.hyperparameter_search import grid_progress


def test_eta_excludes_cached_work_and_reports_stalls(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    clock = [100.0]
    monkeypatch.setattr(grid_progress, "perf_counter", lambda: clock[0])
    status = np.lib.format.open_memmap(
        tmp_path / "status.npy", mode="w+", dtype="uint8", shape=(4, 3)
    )
    status[:] = 0
    status[0] = 1
    status.flush()
    progress = grid_progress.GridProgress(tmp_path, enabled=False)
    try:
        progress.refresh()
        assert progress.completed_configs == 1
        assert progress.remaining_seconds is None
        status[1] = [1, 2, 1]
        status.flush()
        clock[0] += 10
        progress.refresh()
        assert progress.completed_configs == 2
        assert progress.completed_folds == 6
        assert progress.remaining_seconds == pytest.approx(20)
        clock[0] += 10
        progress.refresh()
        assert progress.remaining_seconds == pytest.approx(40)
        status[:] = 1
        status.flush()
        clock[0] += 10
        progress.refresh()
        assert progress.complete
        assert progress.remaining_seconds == 0
    finally:
        progress.close()
        status._mmap.close()


def test_monitor_waits_for_worker_cache(tmp_path: Path) -> None:
    progress = grid_progress.GridProgress(tmp_path, enabled=False)
    try:
        progress.refresh()
        assert not progress.complete
        assert progress.remaining_seconds is None
    finally:
        progress.close()


def test_long_eta_is_not_wrapped_at_24_hours() -> None:
    assert grid_progress._duration(1000 * 3600) == "1000h 00m"
