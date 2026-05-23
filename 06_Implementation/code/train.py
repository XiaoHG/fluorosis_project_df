"""SymMamba 训练脚本: 5-fold CV, CutMix, 不确定性校准, 早停."""

import argparse
import copy
import random
from pathlib import Path

import numpy as np
import torch
import torch.multiprocessing as mp
import torch.nn as nn
import yaml

from data.dataset import FluorosisDataset, generate_split_indices, create_dataloaders
from data.augmentations import get_train_transform, get_val_transform, get_cutmix
from models.symmamba import SymMamba
from models.baselines import create_baseline
from models.losses import compute_total_loss
from utils.metrics import compute_all_metrics, calibrate_threshold
from utils.logger import ExperimentLogger


def deep_merge(base: dict, override: dict) -> dict:
    merged = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(merged.get(k), dict):
            merged[k] = deep_merge(merged[k], v)
        else:
            merged[k] = v
    return merged


def load_config(config_path: str, profile: str = None) -> dict:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    if profile and profile in cfg.get("device_profiles", {}):
        prof = cfg["device_profiles"][profile]
        cfg["device"].update(prof)
        cfg["training"]["batch_size"] = prof["batch_size"]
    return cfg


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_model(cfg: dict) -> nn.Module:
    m = cfg["model"]
    if m["name"] == "symmamba":
        return SymMamba(
            in_channels=cfg["data"].get("in_channels", 3),
            num_classes=cfg["data"]["num_classes"],
            embed_dim=m["backbone"]["embed_dim"],
            depths=m["backbone"]["depths"],
            d_state=m["backbone"]["d_state"],
            d_conv=m["backbone"]["d_conv"],
            expand=m["backbone"]["expand"],
            dropout=m["backbone"]["dropout"],
            edl_hidden=m["edl_head"]["hidden_dim"],
            edl_dropout=m["edl_head"]["dropout"],
            use_sym_head=True,
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


def train_epoch(model, loader, optimizer, loss_cfg, device, cutmix_fn=None, epoch=0):
    model.train()
    total_loss = 0.0
    n_batches = len(loader)
    for i, (x, y) in enumerate(loader):
        x, y = x.to(device), y.to(device)

        if cutmix_fn is not None and torch.rand(1).item() < 0.5:
            x, y_mixed = cutmix_fn(x, y)
        else:
            y_mixed = None

        optimizer.zero_grad()
        out = model(x)

        if y_mixed is not None:
            y_target = y_mixed
        else:
            y_target = y

        alpha = out["alpha"]
        z = out.get("features", out.get("logits", None))
        if z is None:
            z = alpha

        loss, comps = compute_total_loss(alpha, y_target, z, loss_cfg, epoch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        del out, loss, alpha, z

        mem = torch.cuda.memory_allocated(device) / 1024**3
        print(f"  batch {i+1}/{n_batches} loss={total_loss/(i+1):.4f} GPU={mem:.1f}G", flush=True)

    return total_loss / n_batches


@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    all_preds, all_labels, all_alpha, all_u = [], [], [], []
    all_s_sym = []
    for x, y in loader:
        x = x.to(device)
        out = model(x)
        all_preds.append(out["pred"].cpu())
        all_labels.append(y)
        all_alpha.append(out["alpha"].cpu())
        all_u.append(out["u"].cpu())
        if "s_sym" in out:
            all_s_sym.append(out["s_sym"].cpu())

    preds = torch.cat(all_preds).numpy()
    labels = torch.cat(all_labels).numpy()
    alpha = torch.cat(all_alpha).numpy()
    u = torch.cat(all_u).numpy()

    cal = calibrate_threshold(labels, preds, u)
    metrics = compute_all_metrics(labels, preds, alpha, u, cal["theta"])
    metrics["val_loss"] = 0.0
    return metrics, cal


def main():
    parser = argparse.ArgumentParser(description="SymMamba 氟斑牙分级训练")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--profile", default="rtx_pro_6000")
    parser.add_argument("--fold", type=int, default=-1, help="单折训练, -1=全部5折")
    parser.add_argument("--resume", default="")
    parser.add_argument("--data_root", default="")
    parser.add_argument("--exp_name", default="")
    args = parser.parse_args()

    mp.set_start_method('spawn', force=True)

    cfg = load_config(args.config, args.profile)
    if args.data_root:
        cfg["data"]["root"] = args.data_root
    set_seed(cfg["experiment"]["seed"])

    device_str = cfg["device"].get("device", "cuda")
    if device_str == "cuda" and torch.cuda.is_available():
        device = torch.device(f"cuda:{cfg['device']['gpu_id']}")
    elif device_str == "mps" and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    image_size = tuple(cfg["data"]["image_size"])
    print(f"Loading dataset from {cfg['data']['root']}...")
    full_ds = FluorosisDataset(cfg["data"]["root"])
    print(f"  {len(full_ds)} images in {len(full_ds.CLASS_NAMES)} classes")

    split_path = cfg["cv"].get("split_file", "split_indices.json")
    if Path(split_path).exists():
        from data.dataset import load_split_indices
        splits = load_split_indices(split_path)
    else:
        splits = generate_split_indices(full_ds, cfg["cv"]["n_folds"],
                                         cfg["experiment"]["seed"], split_path)
    folds_to_run = [args.fold] if args.fold >= 0 else list(range(len(splits)))

    for fold_idx in folds_to_run:
        split = splits[fold_idx]
        exp_name = args.exp_name or cfg["experiment"]["name"]
        fold_name = f"{exp_name}_fold{fold_idx}"
        logger = ExperimentLogger(cfg["logging"]["log_dir"], fold_name,
                                   cfg["logging"]["save_best_only"])
        print(f"\n===== Fold {fold_idx}/{len(splits)} =====")
        print(f"  Train: {len(split['train'])} samples, Val: {len(split['val'])} samples")

        print(f"  Preparing DataLoader (spawn, {cfg['device']['num_workers']} workers)...")
        train_tf = get_train_transform(image_size)
        val_tf = get_val_transform(image_size)
        train_loader, val_loader = create_dataloaders(
            full_ds, split, cfg["training"]["batch_size"],
            train_tf, val_tf, cfg["device"]["num_workers"], cfg["device"]["pin_memory"])
        cutmix_fn = get_cutmix() if cfg.get("augmentation", {}).get("use_cutmix", False) else None
        print(f"  DataLoader ready ({len(train_loader)} batches/epoch)")

        print(f"  Building model...")
        model = build_model(cfg).to(device)
        n_params = sum(p.numel() for p in model.parameters())
        n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  Model: {n_params:,} params ({n_trainable:,} trainable)")

        head_params = []
        backbone_params = []
        for name, p in model.named_parameters():
            if "edl_head" in name or "head" in name:
                head_params.append(p)
            else:
                backbone_params.append(p)

        opt_cfg = cfg["optimizer"]
        optimizer = torch.optim.AdamW([
            {"params": backbone_params, "lr": opt_cfg["lr_backbone"]},
            {"params": head_params, "lr": opt_cfg["lr_head"]},
        ], weight_decay=opt_cfg["weight_decay"], betas=opt_cfg["betas"])

        sch_cfg = cfg["scheduler"]
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cfg["training"]["epochs"] - sch_cfg["warmup_epochs"],
            eta_min=sch_cfg["min_lr"])

        def warmup_lr(epoch, warmup_epochs, base_lrs):
            if epoch < warmup_epochs:
                scale = (epoch + 1) / warmup_epochs
                for i, pg in enumerate(optimizer.param_groups):
                    pg["lr"] = base_lrs[i] * scale

        base_lrs = [pg["lr"] for pg in optimizer.param_groups]

        start_epoch = 0
        best_metric = -float("inf")
        patience_counter = 0
        es_patience = cfg["training"]["early_stop_patience"]

        if args.resume:
            ckpt = torch.load(args.resume, map_location=device)
            model.load_state_dict(ckpt["model_state_dict"])
            optimizer.load_state_dict(ckpt.get("optimizer_state_dict", {}))
            start_epoch = ckpt["epoch"] + 1
            best_metric = ckpt.get("best_metric", -float("inf"))
            print(f"Resumed from epoch {start_epoch}")

        n_epochs = cfg["training"]["epochs"]
        print(f"  Training {n_epochs} epochs (early stop patience={es_patience})...")
        torch.cuda.reset_peak_memory_stats(device)

        for epoch in range(start_epoch, n_epochs):
            warmup_lr(epoch, sch_cfg["warmup_epochs"], base_lrs)

            train_loss = train_epoch(model, train_loader, optimizer,
                                      cfg["loss"], device, cutmix_fn, epoch)

            if epoch >= sch_cfg["warmup_epochs"]:
                scheduler.step()

            val_metrics, cal = validate(model, val_loader, device)
            val_metrics["train_loss"] = train_loss

            is_best = logger.save_checkpoint(
                model, epoch, val_metrics["qwk"],
                mode=cfg["training"]["early_stop_mode"],
                optimizer=optimizer, scheduler=scheduler,
                extra={"theta": cal["theta"]})

            logger.write_row(epoch, val_metrics)

            torch.cuda.empty_cache()
            mem = torch.cuda.memory_allocated(device) / 1024**3
            peak = torch.cuda.max_memory_allocated(device) / 1024**3

            pf = cfg["logging"].get("print_freq", 1)
            if (epoch + 1) % pf == 0 or epoch == 0 or is_best:
                print(f"Epoch {epoch:3d} | loss: {train_loss:.4f} | "
                      f"QWK: {val_metrics['qwk']:.4f} | SDR: {val_metrics.get('sdr', 0):.4f} | "
                      f"theta*: {cal['theta']:.2f} | GPU: {mem:.1f}/{peak:.1f}G | "
                      f"{logger.elapsed()} | {'* BEST' if is_best else ''}")

            if is_best:
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= es_patience:
                print(f"Early stopping at epoch {epoch}")
                break

        best_ckpt = torch.load(logger.ckpt_dir / "best.pt", map_location=device)
        model.load_state_dict(best_ckpt["model_state_dict"])
        val_metrics, _ = validate(model, val_loader, device)
        logger.save_summary(val_metrics)
        print(f"Fold {fold_idx} final | QWK={val_metrics['qwk']:.4f} SDR={val_metrics.get('sdr',0):.4f} theta*={best_ckpt['extra'].get('theta',0.5):.2f}")

        logger.close()


if __name__ == "__main__":
    main()
