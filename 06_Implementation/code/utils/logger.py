"""实验日志: CSV/JSONL metrics + run_info + checkpoint 管理 + 终端输出."""

import csv
import json
import os
import subprocess
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch


def _get_git_commit() -> str:
    """获取当前 git commit hash, 失败返回 'unknown'."""
    try:
        r = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                          capture_output=True, text=True, timeout=5)
        return r.stdout.strip() if r.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


def _get_git_branch() -> str:
    """获取当前 git branch, 失败返回 'unknown'."""
    try:
        r = subprocess.run(["git", "branch", "--show-current"],
                          capture_output=True, text=True, timeout=5)
        return r.stdout.strip() if r.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


class ExperimentLogger:
    """Training logger with git-friendly monitoring data.

    每个实验目录结构:
        logs/{exp_name}/
        ├── run_info.json      # 一次性写入 — 配置 + 系统信息 + git commit
        ├── metrics.jsonl      # 每 epoch 追加一行 JSON (git diff 友好)
        ├── metrics.csv        # 传统 CSV (含 loss 分量)
        ├── checkpoints/
        │   ├── best.pt
        │   └── latest.pt
        └── summary.json       # 最终评估

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

        # CSV 日志
        self.csv_path = self.log_dir / "metrics.csv"
        self.csv_file = open(self.csv_path, "w", newline="", encoding='utf-8')
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_header_written = False

        # JSONL 日志 (每 epoch 一行, git diff 友好)
        self.jsonl_path = self.log_dir / "metrics.jsonl"
        self.jsonl_file = open(self.jsonl_path, "w", encoding='utf-8')

        self.best_metric = -float("inf")
        self.best_epoch = -1
        self.start_time = time.time()

    # ---- Run info --------------------------------------------------------

    def save_run_info(self, cfg: dict, device: str, model_params: int,
                      trainable_params: int, dataset_size: int,
                      num_classes: int, extra: dict = None):
        """保存实验元数据 (启动时调用一次)."""
        import torch as _torch

        info = {
            "exp_name": cfg["experiment"]["name"],
            "seed": cfg["experiment"]["seed"],
            "started_at": datetime.now().isoformat(),
            "git_commit": _get_git_commit(),
            "git_branch": _get_git_branch(),
            "device": str(device),
            "model": {
                "name": cfg["model"]["name"],
                "variant": cfg["model"].get("variant", "unknown"),
                "total_params": model_params,
                "trainable_params": trainable_params,
            },
            "data": {
                "dataset_size": dataset_size,
                "num_classes": num_classes,
                "class_names": cfg["data"].get("class_names", []),
                "image_size": cfg["data"]["image_size"],
            },
            "training": {
                "batch_size": cfg["training"]["batch_size"],
                "epochs": cfg["training"]["epochs"],
                "early_stop_patience": cfg["training"]["early_stop_patience"],
                "early_stop_metric": cfg["training"]["early_stop_metric"],
                "grad_clip_norm": cfg["training"]["grad_clip_norm"],
            },
            "optimizer": cfg["optimizer"],
            "scheduler": cfg["scheduler"],
            "loss": {k: v for k, v in cfg["loss"].items() if k != "schedule"},
            "loss_schedule": cfg["loss"].get("schedule", {}),
            "model_cfg": cfg["model"],
            "system": {
                "pytorch_version": _torch.__version__,
                "cuda_available": _torch.cuda.is_available(),
                "cuda_version": _torch.version.cuda if _torch.cuda.is_available() else None,
                "gpu_name": _torch.cuda.get_device_name(0) if _torch.cuda.is_available() else None,
                "gpu_count": _torch.cuda.device_count() if _torch.cuda.is_available() else 0,
            },
        }
        if extra:
            info["extra"] = extra

        with open(self.log_dir / "run_info.json", "w", encoding='utf-8') as f:
            json.dump(info, f, indent=2, ensure_ascii=False)

    # ---- Metrics logging -------------------------------------------------

    def write_row(self, epoch: int, metrics: dict, loss_comps: dict = None):
        """写入一行 metrics 到 CSV 和 JSONL.

        Args:
            epoch: 当前 epoch 编号.
            metrics: validation metrics dict.
            loss_comps: loss 分量 dict, e.g. {"L_CE": 1.2, "L_EDL": 0.0, ...}
        """
        row = {"epoch": epoch}
        row.update(metrics)
        if loss_comps:
            row.update(loss_comps)

        # CSV
        if not self.csv_header_written:
            self.csv_writer.writerow(row.keys())
            self.csv_header_written = True
        self.csv_writer.writerow([self._clean_value(v) for v in row.values()])
        self.csv_file.flush()

        # JSONL (每行一条完整 JSON, git diff 只显示新增行)
        json_row = {k: self._clean_value(v) for k, v in row.items()}
        self.jsonl_file.write(json.dumps(json_row, ensure_ascii=False) + "\n")
        self.jsonl_file.flush()

    @staticmethod
    def _clean_value(v):
        """转换 numpy/torch 类型为原生 Python 类型."""
        if isinstance(v, (np.floating, np.integer)):
            return float(v) if isinstance(v, np.floating) else int(v)
        if isinstance(v, torch.Tensor):
            return v.item()
        if isinstance(v, np.ndarray):
            return v.tolist()
        return v

    # ---- Checkpointing ---------------------------------------------------

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
            if isinstance(sd, torch.Tensor):
                return sd.cpu()
            if isinstance(sd, dict):
                return {k: _cpu_state(v) for k, v in sd.items()}
            if isinstance(sd, list):
                return [_cpu_state(v) for v in sd]
            return sd

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

    def save_latest(self, model, epoch, optimizer=None, scheduler=None,
                     rng_state=None, extra=None):
        """Save latest checkpoint for crash recovery (always written)."""
        def _cpu_state(sd):
            if isinstance(sd, torch.Tensor):
                return sd.cpu()
            if isinstance(sd, dict):
                return {k: _cpu_state(v) for k, v in sd.items()}
            if isinstance(sd, list):
                return [_cpu_state(v) for v in sd]
            return sd

        ckpt = {
            "epoch": epoch,
            "model_state_dict": _cpu_state(model.state_dict()),
            "best_metric": self.best_metric,
            "best_epoch": self.best_epoch,
        }
        if optimizer:
            ckpt["optimizer_state_dict"] = _cpu_state(optimizer.state_dict())
        if scheduler:
            ckpt["scheduler_state_dict"] = _cpu_state(scheduler.state_dict())
        if rng_state:
            ckpt["rng_state"] = rng_state
        if extra:
            ckpt["extra"] = extra
        torch.save(ckpt, self.ckpt_dir / "latest.pt")

    def save_summary(self, metrics: dict, filename: str = "summary.json"):
        """Save final evaluation summary."""
        clean = {}
        for k, v in metrics.items():
            if isinstance(v, (np.floating, np.integer)):
                clean[k] = float(v)
            elif isinstance(v, np.ndarray):
                clean[k] = v.tolist()
            else:
                clean[k] = v
        with open(self.log_dir / filename, "w", encoding='utf-8') as f:
            json.dump(clean, f, indent=2)

    def elapsed(self) -> str:
        mins = (time.time() - self.start_time) / 60
        if mins < 60:
            return f"{mins:.1f}m"
        return f"{mins / 60:.1f}h"

    def close(self):
        self.csv_file.close()
        self.jsonl_file.close()
