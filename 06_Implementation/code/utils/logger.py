"""实验日志: CSV metrics + checkpoint 管理 + 终端输出."""

import csv
import json
import time
from pathlib import Path

import numpy as np
import torch


class ExperimentLogger:
    """Training logger: CSV metrics, best-model tracking, terminal output.

    Args:
        log_dir: directory for logs/ subfolder.
        exp_name: experiment name, creates {log_dir}/{exp_name}/.
        save_best_only: only keep best checkpoint on disk.
    """

    def __init__(self, log_dir: str, exp_name: str, save_best_only: bool = True):
        self.log_dir = Path(log_dir) / exp_name
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.ckpt_dir = self.log_dir / "checkpoints"
        self.ckpt_dir.mkdir(exist_ok=True)
        self.save_best_only = save_best_only

        self.csv_path = self.log_dir / "metrics.csv"
        self.csv_file = open(self.csv_path, "w", newline="")
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_header_written = False

        self.best_metric = -float("inf")
        self.best_epoch = -1
        self.start_time = time.time()

    def write_row(self, epoch: int, metrics: dict):
        """Write one row of metrics."""
        row = {"epoch": epoch}
        row.update(metrics)
        if not self.csv_header_written:
            self.csv_writer.writerow(row.keys())
            self.csv_header_written = True
        self.csv_writer.writerow(row.values())
        self.csv_file.flush()

    def save_checkpoint(self, model: torch.nn.Module, epoch: int,
                         val_metric: float, mode: str = "max",
                         optimizer=None, scheduler=None, extra: dict = None):
        """Save checkpoint; track best. Returns True if new best."""
        if mode == "max":
            is_best = val_metric > self.best_metric
        else:
            is_best = val_metric < self.best_metric

        if is_best:
            self.best_metric = val_metric
            self.best_epoch = epoch

        def _cpu_state(sd):
            return {k: v.cpu() for k, v in sd.items()}

        ckpt = {
            "epoch": epoch,
            "model_state_dict": _cpu_state(model.state_dict()),
            "best_metric": self.best_metric,
            "best_epoch": self.best_epoch,
            "extra": extra or {},
        }
        if optimizer:
            ckpt["optimizer_state_dict"] = _cpu_state(optimizer.state_dict())
        if scheduler:
            ckpt["scheduler_state_dict"] = _cpu_state(scheduler.state_dict())

        # Always save best
        torch.save(ckpt, self.ckpt_dir / "best.pt")

        if not self.save_best_only:
            torch.save(ckpt, self.ckpt_dir / f"epoch_{epoch:03d}.pt")

        return is_best

    def save_predictions(self, preds: np.ndarray, labels: np.ndarray,
                           filename: str = "predictions.csv"):
        """Save predictions vs labels."""
        path = self.log_dir / filename
        np.savetxt(path, np.stack([labels, preds], axis=1), fmt="%d",
                    delimiter=",", header="label,pred", comments="")

    def save_summary(self, metrics: dict, filename: str = "summary.json"):
        """Save final evaluation summary."""
        # Convert numpy types
        clean = {}
        for k, v in metrics.items():
            if isinstance(v, (np.floating, np.integer)):
                clean[k] = float(v)
            elif isinstance(v, np.ndarray):
                clean[k] = v.tolist()
            else:
                clean[k] = v
        with open(self.log_dir / filename, "w") as f:
            json.dump(clean, f, indent=2)

    def elapsed(self) -> str:
        mins = (time.time() - self.start_time) / 60
        if mins < 60:
            return f"{mins:.1f}m"
        return f"{mins / 60:.1f}h"

    def close(self):
        self.csv_file.close()
