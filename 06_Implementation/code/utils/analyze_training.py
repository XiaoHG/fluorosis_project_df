#!/usr/bin/env python3
"""训练数据分析 — 读取 run_info.json + metrics.jsonl 生成诊断报告.

用法:
    python code/utils/analyze_training.py logs/e4_full_symmamba_fold0
    python code/utils/analyze_training.py logs/              # 分析所有实验
    python code/utils/analyze_training.py logs/ --json       # JSON 输出 (供 AI 消费)
"""

import argparse
import json
import sys
from pathlib import Path


def load_experiment(log_dir: Path) -> dict:
    """加载单个实验的所有数据."""
    data = {"dir": str(log_dir), "name": log_dir.name}

    info_path = log_dir / "run_info.json"
    if info_path.exists():
        with open(info_path, encoding='utf-8') as f:
            data["run_info"] = json.load(f)

    metrics_path = log_dir / "metrics.jsonl"
    epochs = []
    if metrics_path.exists():
        with open(metrics_path, encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    epochs.append(json.loads(line))
    data["epochs"] = epochs

    summary_path = log_dir / "summary.json"
    if summary_path.exists():
        with open(summary_path, encoding='utf-8') as f:
            data["summary"] = json.load(f)

    return data


def analyze_experiment(exp: dict) -> dict:
    """分析单个实验: 趋势, 异常, 建议."""
    epochs = exp["epochs"]
    info = exp.get("run_info", {})
    n = len(epochs)

    report = {
        "exp_name": exp["name"],
        "total_epochs": n,
        "status": "running",
        "issues": [],
        "metrics_summary": {},
    }

    if n == 0:
        report["status"] = "no_data"
        report["issues"].append("no epoch data")
        return report

    if info:
        report["model"] = info.get("model", {}).get("name", "unknown")
        report["batch_size"] = info.get("training", {}).get("batch_size", "?")
        report["device"] = info.get("device", "?")
        report["git_commit"] = info.get("git_commit", "?")

    first = epochs[0]
    last = epochs[-1]

    def _get(ep, key, default=None):
        return ep.get(key, default)

    best_qwk = max(epochs, key=lambda e: e.get("qwk", -1))
    best_acc = max(epochs, key=lambda e: e.get("accuracy", -1))

    report["metrics_summary"] = {
        "first_epoch": {
            "loss": round(_get(first, "train_loss", 0), 4),
            "qwk": round(_get(first, "qwk", 0), 4),
            "acc": round(_get(first, "accuracy", 0), 4),
            "f1": round(_get(first, "macro_f1", 0), 4),
        },
        "last_epoch": {
            "epoch": n - 1,
            "loss": round(_get(last, "train_loss", 0), 4),
            "qwk": round(_get(last, "qwk", 0), 4),
            "acc": round(_get(last, "accuracy", 0), 4),
            "f1": round(_get(last, "macro_f1", 0), 4),
        },
        "best_qwk": {
            "epoch": best_qwk["epoch"],
            "qwk": round(best_qwk["qwk"], 4),
            "loss": round(_get(best_qwk, "train_loss", 0), 4),
        },
        "best_acc": {
            "epoch": best_acc["epoch"],
            "acc": round(best_acc["accuracy"], 4),
        },
    }

    # Loss 趋势
    if n >= 3:
        recent_loss = [e.get("train_loss", 0) for e in epochs[-3:]]
        early_loss = [e.get("train_loss", 0) for e in epochs[:min(3, n)]]
        avg_recent = sum(recent_loss) / len(recent_loss)
        avg_early = sum(early_loss) / len(early_loss)
        report["loss_trend"] = {
            "early_avg": round(avg_early, 4),
            "recent_avg": round(avg_recent, 4),
            "change_pct": round((avg_recent - avg_early) / (avg_early + 1e-8) * 100, 1),
        }

    # QWK 趋势
    if n >= 5:
        recent_qwk = [e.get("qwk", 0) for e in epochs[-5:]]
        if recent_qwk[-1] > recent_qwk[0] + 0.01:
            report["qwk_trend"] = "improving"
        elif abs(recent_qwk[-1] - recent_qwk[0]) < 0.01:
            report["qwk_trend"] = "stagnant"
        else:
            report["qwk_trend"] = "declining"

    # 异常检测
    if n >= 5 and "loss_trend" in report:
        lt = report["loss_trend"]
        if lt["change_pct"] > 10:
            report["issues"].append(f"loss rising {lt['change_pct']:.0f}% — lr may be too high or overfitting")
        elif lt["change_pct"] > -5 and n > 10:
            report["issues"].append(f"loss plateau ({lt['change_pct']:.0f}%) — may be stuck in local minimum")

    if best_qwk.get("qwk", 0) < 0:
        report["issues"].append("QWK negative — model worse than random, check labels/loss")

    if n >= 10:
        mid = n // 2
        first_half_qwk = max(e.get("qwk", 0) for e in epochs[:mid])
        second_half_qwk = max(e.get("qwk", 0) for e in epochs[mid:])
        if first_half_qwk > second_half_qwk + 0.1:
            report["issues"].append(f"QWK declining ({first_half_qwk:.3f} -> {second_half_qwk:.3f}) — possible overfitting")

    for e in epochs:
        if "train_loss" in e and (e["train_loss"] != e["train_loss"]):  # NaN
            report["issues"].append(f"epoch {e['epoch']}: loss=NaN — gradient explosion!")
            report["status"] = "crashed"
            break

    # GPU 利用率
    if n > 0 and "gpu_mem_gb" in epochs[0]:
        avg_mem = sum(e.get("gpu_mem_gb", 0) for e in epochs) / n
        peak_mem = max(e.get("gpu_peak_gb", 0) for e in epochs)
        report["gpu"] = {"avg_mem_gb": round(avg_mem, 1), "peak_mem_gb": round(peak_mem, 1)}
        if avg_mem < 0.5:
            report["issues"].append(f"GPU underutilized ({avg_mem:.2f}GB avg) — DataLoader may be bottleneck")

    # 学习率
    if n > 0 and "lr_backbone" in epochs[0]:
        report["lr"] = {"start": epochs[0]["lr_backbone"], "end": round(epochs[-1]["lr_backbone"], 8)}

    # 综合判断
    recs = []
    if not report["issues"]:
        if n < 10:
            recs.append("early stage, continue training")
        elif report.get("qwk_trend") == "improving":
            recs.append("QWK still improving, continue")
            report["status"] = "healthy"
        else:
            recs.append("QWK not improving, consider reducing lr or adding regularization")
    else:
        for issue in report["issues"]:
            if "overfitting" in issue:
                recs.append("increase dropout / weight_decay / data augmentation")
            if "plateau" in issue:
                recs.append("reduce lr 10x or try warm restart")
            if "rising" in issue:
                recs.append("check lr is not too high, verify warmup and grad clipping")
            if "underutilized" in issue:
                recs.append("check num_workers, pin_memory, consider data caching")
            if "NaN" in issue:
                recs.append("gradient explosion! reduce lr, increase grad_clip_norm, check data")
    report["recommendations"] = recs

    # 早停预测
    if info.get("training", {}).get("early_stop_patience", 0) > 0:
        best_ep = best_qwk["epoch"]
        es_ep = info["training"]["early_stop_patience"]
        eps_since_best = n - 1 - best_ep
        if eps_since_best >= es_ep:
            report["status"] = "early_stopped"
            report["notes"] = f"QWK no improvement for {eps_since_best} epochs since epoch {best_ep}"

    return report


def analyze_all(logs_dir: Path) -> list[dict]:
    """分析所有实验目录."""
    reports = []
    for subdir in sorted(logs_dir.iterdir()):
        if subdir.is_dir() and (subdir / "run_info.json").exists():
            exp = load_experiment(subdir)
            reports.append(analyze_experiment(exp))
    return reports


def print_report_text(report: dict):
    """人类可读报告."""
    print(f"\n{'='*60}")
    print(f"Experiment: {report['exp_name']}")
    print(f"Status: {report.get('status', 'unknown')}")
    if report.get("model"):
        print(f"Model: {report['model']} | batch: {report.get('batch_size', '?')} | device: {report.get('device', '?')}")
    if report.get("git_commit"):
        print(f"Commit: {report['git_commit']}")
    print(f"Total Epochs: {report['total_epochs']}")

    ms = report.get("metrics_summary", {})
    if ms:
        print(f"\n--- Metrics Summary ---")
        f = ms.get("first_epoch", {})
        print(f"  Initial:  loss={f.get('loss','?')}  QWK={f.get('qwk','?')}  Acc={f.get('acc','?')}  F1={f.get('f1','?')}")
        b = ms.get("best_qwk", {})
        print(f"  Best QWK: epoch={b.get('epoch','?')}  QWK={b.get('qwk','?')}  loss={b.get('loss','?')}")
        l = ms.get("last_epoch", {})
        print(f"  Latest:   epoch={l.get('epoch','?')}  QWK={l.get('qwk','?')}  Acc={l.get('acc','?')}  F1={l.get('f1','?')}")

    lt = report.get("loss_trend", {})
    if lt:
        print(f"\nLoss Trend: {lt['early_avg']:.4f} -> {lt['recent_avg']:.4f} ({lt['change_pct']:+.1f}%)")

    qt = report.get("qwk_trend")
    if qt:
        print(f"QWK Trend: {qt}")

    lr = report.get("lr")
    if lr:
        print(f"LR: {lr['start']} -> {lr['end']}")

    gpu = report.get("gpu")
    if gpu:
        print(f"GPU: avg={gpu['avg_mem_gb']}GB peak={gpu['peak_mem_gb']}GB")

    issues = report.get("issues", [])
    if issues:
        print(f"\n--- Issues ({len(issues)}) ---")
        for i in issues:
            print(f"  ! {i}")
    else:
        print(f"\n--- No Issues ---")

    recs = report.get("recommendations", [])
    if recs:
        print(f"\n--- Recommendations ---")
        for r in recs:
            print(f"  * {r}")

    if report.get("notes"):
        print(f"\nNote: {report['notes']}")


def main():
    parser = argparse.ArgumentParser(description="Analyze training data")
    parser.add_argument("path", help="path to logs dir or specific experiment dir")
    parser.add_argument("--json", action="store_true", help="JSON output for AI consumption")
    args = parser.parse_args()

    path = Path(args.path)
    if not path.exists():
        print(f"Error: {path} does not exist", file=sys.stderr)
        sys.exit(1)

    if (path / "run_info.json").exists():
        exp = load_experiment(path)
        reports = [analyze_experiment(exp)]
    else:
        reports = analyze_all(path)

    if not reports:
        print("No experiment data found (need run_info.json)", file=sys.stderr)
        sys.exit(1)

    if args.json:
        print(json.dumps(reports, indent=2, ensure_ascii=False))
    else:
        for r in reports:
            print_report_text(r)
        print(f"\n{'='*60}")
        print(f"Total: {len(reports)} experiment(s)")


if __name__ == "__main__":
    main()
