from __future__ import annotations

import argparse
from pathlib import Path
from time import perf_counter, sleep
from typing import Optional

import numpy as np
from tqdm.auto import tqdm


def _duration(seconds: Optional[float]) -> str:
    if seconds is None:
        return "?"
    minutes, seconds = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h {minutes:02d}m" if hours else f"{minutes}m {seconds:02d}s"


class GridProgress:
    """Display committed Grid evaluations and ETA from newly completed CV folds."""

    def __init__(self, directory: Path, position: int = 1, enabled: bool = True) -> None:
        self.directory = directory
        self.position = position
        self.completed_configs = 0
        self.completed_folds = 0
        self.remaining_seconds: Optional[float] = None
        self.complete = False
        self._status: Optional[np.memmap] = None
        self._baseline = 0
        self._started = self._changed = perf_counter()
        self._last_sample = -float("inf")
        self.bar = tqdm(
            total=0,
            desc=f"Grid {directory.name}",
            position=position,
            leave=False,
            dynamic_ncols=True,
            disable=not enabled,
            bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{postfix}]",
        )

    def refresh(self) -> None:
        now = perf_counter()
        if now - self._last_sample < 1:
            return
        self._last_sample = now
        if self._status is None:
            path = self.directory / "status.npy"
            if not path.exists():
                self.bar.set_postfix_str("starting worker / preparing cache")
                return
            try:
                status = np.load(path, mmap_mode="r", allow_pickle=False)
            except (OSError, ValueError):
                self.bar.set_postfix_str("preparing cache")
                return
            self._status = status
            self._baseline = int(np.count_nonzero(status))
            self.completed_folds = self._baseline
            self._started = self._changed = now
        status = self._status
        folds = int(np.count_nonzero(status))
        self.completed_configs = int(np.count_nonzero(np.all(status != 0, axis=1)))
        if folds != self.completed_folds:
            self._changed = now
        self.completed_folds = folds
        self.complete = folds == status.size
        elapsed = now - self._started
        rate = (folds - self._baseline) / elapsed if elapsed > 0 else 0.0
        self.remaining_seconds = (status.size - folds) / rate if rate > 0 else None
        self.bar.total = len(status)
        self.bar.n = self.completed_configs
        details = (
            "CV complete"
            if self.complete
            else f"{folds}/{status.size} folds; {rate:.2f} folds/s; ETA {_duration(self.remaining_seconds)}; idle {_duration(now - self._changed)}"
        )
        self.bar.set_postfix_str(details)

    def close(self) -> None:
        if self._status is not None:
            getattr(self._status, "_mmap").close()
        self.bar.close()


def watch(output: Path) -> None:
    active: dict[Path, GridProgress] = {}
    finished: set[Path] = set()
    print(f"Watching Grid caches in {output / 'cache'}; Ctrl+C stops only the monitor.")
    try:
        while True:
            for directory in sorted((output / "cache").glob("*")):
                if directory.is_dir() and directory not in active and directory not in finished:
                    positions = {progress.position for progress in active.values()}
                    position = next(i for i in range(1, len(active) + 2) if i not in positions)
                    active[directory] = GridProgress(directory, position)
            for directory, progress in list(active.items()):
                progress.refresh()
                if progress.complete:
                    tqdm.write(
                        f"Grid {directory.name}: CV complete ({progress.completed_configs} configurations)"
                    )
                    progress.close()
                    del active[directory]
                    finished.add(directory)
            sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        for progress in active.values():
            progress.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Watch Grid CV progress without changing the experiment."
    )
    parser.add_argument(
        "--output", type=Path, default=Path(__file__).resolve().parent / "output" / "experiment"
    )
    watch(parser.parse_args().output)
