"""
Training metrics: TensorBoard scalars + optional CSV episode log.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict

from torch.utils.tensorboard import SummaryWriter


class TrainLogger:
    def __init__(self, log_dir: str | Path, csv_name: str = "episodes.csv"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._writer = SummaryWriter(str(self.log_dir))
        self._csv_path = self.log_dir / csv_name
        self._csv_header_written = self._csv_path.is_file()

    def close(self) -> None:
        self._writer.close()

    def log_scalars(self, prefix: str, scalars: Dict[str, float], step: int) -> None:
        for k, v in scalars.items():
            self._writer.add_scalar(f"{prefix}/{k}", float(v), step)

    def log_episode_csv(self, row: Dict[str, Any]) -> None:
        mode = "a" if self._csv_header_written else "w"
        with open(self._csv_path, mode, newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(row.keys()))
            if not self._csv_header_written:
                w.writeheader()
                self._csv_header_written = True
            w.writerow({k: row[k] for k in row})
