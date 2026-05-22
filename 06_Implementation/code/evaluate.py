"""评估脚本: 加载最佳模型, 计算全量指标并保存."""

import argparse
from pathlib import Path

import numpy as np
import torch
import yaml

from data.dataset import FluorosisDataset, load_split_indices, _SubsetWithTransform
from data.augmentations import get_val_transform
from utils.metrics import compute_all_metrics, calibrate_threshold
from models.symmamba import SymMamba
from models.baselines import create_baseline


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def build_model(cfg: dict) -> torch.nn.Module:
    m = cfg["model"]
    if m["name"] == "symmamba":
        return SymMamba(
            num_classes=cfg["data"]["num_classes"],
            embed_dim=m["backbone"]["embed_dim"],
            depths=m["backbone"]["depths"],
            d_state=m["backbone"]["d_state"],
            d_conv=m["backbone"]["d_conv"],
            expand=m["backbone"]["expand"],
            dropout=m["backbone"]["dropout"],
            edl_hidden=m["edl_head"]["hidden_dim"],
            edl_dropout=m["edl_head"]["dropout"],
        )
    elif m["name"] in ("resnet50", "resnet50_coral", "vit"):
        edl_cfg = m.get("edl_head", {})
        kwargs = {"num_classes": cfg["data"]["num_classes"]}
        if m["name"] == "vit":
            kwargs["pretrained"] = m["backbone"].get("pretrained", False)
            kwargs["img_size"] = m["backbone"].get("img_size", 224)
        if m["name"] != "resnet50_coral":
            kwargs["edl_hidden"] = edl_cfg.get("hidden_dim", 256)
            kwargs["edl_dropout"] = edl_cfg.get("dropout", 0.3)
        return create_baseline(m["name"], **kwargs)
    raise ValueError(f"Unknown model: {m['name']}")


@torch.no_grad()
def evaluate_model(model, loader, device):
    model.eval()
    all_preds, all_labels, all_alpha, all_u = [], [], [], []
    for x, y in loader:
        x = x.to(device)
        out = model(x)
        all_preds.append(out["pred"].cpu())
        all_labels.append(y)
        if "alpha" in out:
            all_alpha.append(out["alpha"].cpu())
        if "u" in out:
            all_u.append(out["u"].cpu())

    preds = torch.cat(all_preds).numpy()
    labels = torch.cat(all_labels).numpy()
    alpha = torch.cat(all_alpha).numpy() if all_alpha else np.zeros((len(preds), 4))
    u = torch.cat(all_u).numpy() if all_u else np.zeros(len(preds))

    return preds, labels, alpha, u


def main():
    parser = argparse.ArgumentParser(description="氟斑牙模型评估")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split_file", default="split_indices.json")
    parser.add_argument("--data_root", default="")
    parser.add_argument("--output_dir", default="evaluation_results")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.data_root:
        cfg["data"]["root"] = args.data_root

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    full_ds = FluorosisDataset(cfg["data"]["root"])
    splits = load_split_indices(args.split_file)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_fold_metrics = []

    for split in splits:
        fold = split["fold"]
        print(f"\n===== Evaluating Fold {fold} =====")

        val_tf = get_val_transform(tuple(cfg["data"]["image_size"]))
        val_ds = _SubsetWithTransform(full_ds, split["val"], val_tf)
        val_loader = torch.utils.data.DataLoader(
            val_ds, batch_size=cfg["training"]["batch_size"],
            shuffle=False, num_workers=2)

        model = build_model(cfg).to(device)
        ckpt = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"Loaded checkpoint from epoch {ckpt.get('epoch', '?')}")

        preds, labels, alpha, u = evaluate_model(model, val_loader, device)

        cal = calibrate_threshold(labels, preds, u)
        metrics = compute_all_metrics(labels, preds, alpha, u, cal["theta"])
        metrics["fold"] = fold
        metrics["theta"] = cal["theta"]
        metrics["retention"] = cal["retention"]
        all_fold_metrics.append(metrics)

        np.savetxt(out_dir / f"fold{fold}_preds.csv",
                    np.stack([labels, preds], axis=1),
                    fmt="%d", delimiter=",", header="label,pred", comments="")
        print(f"Fold {fold}: QWK={metrics['qwk']:.4f} SDR={metrics['sdr']:.4f}")

    print("\n===== Cross-Fold Summary =====")
    keys = ["qwk", "sdr", "macro_f1", "ece", "ur_acc", "retention"]
    for k in keys:
        vals = [m[k] for m in all_fold_metrics if k in m]
        if vals:
            print(f"{k:>12s}: {np.mean(vals):.4f} +/- {np.std(vals):.4f}")


if __name__ == "__main__":
    main()
