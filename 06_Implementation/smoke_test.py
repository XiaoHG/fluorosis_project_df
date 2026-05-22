"""冒烟测试: 端到端验证训练管线 (使用小输入加速CPU测试).

测试项:
1. 数据集加载 + 5-fold split
2. DataLoader + 增强管线
3. SymMamba 前向传播 (small input)
4. 损失计算 (EDL + ordinal + contrastive + boundary)
5. 训练步 (forward -> loss -> backward -> step)
6. 验证 + 指标 + 校准
7. Checkpoint 保存/加载
8. Baseline 模型
"""

import sys, os, tempfile
import numpy as np
import torch

os.chdir("/Volumes/KINGSTON/fluorosis_project/fluorosis_project_df/06_Implementation")
sys.path.insert(0, "code")

DATA_ROOT = "/Volumes/KINGSTON/fluorosis_project/fluorosis_project_df/dataset_df/dental_fluorosis/images"
DEVICE = torch.device("cpu")
SMALL_SIZE = (64, 32)  # tiny size for fast CPU smoke test
passed = 0; failed = 0

def check(name, condition):
    global passed, failed
    if condition:
        passed += 1; print(f"  [PASS] {name}")
    else:
        failed += 1; print(f"  [FAIL] {name}")

print("=" * 60)
print(f"SymMamba Smoke Test | Device: {DEVICE} | PyTorch {torch.__version__}")
print(f"Using small input size: {SMALL_SIZE}")
print("=" * 60)

# ---- 1. Dataset ----
print("\n1. Dataset & Split")
from data.dataset import FluorosisDataset, generate_split_indices

ds = FluorosisDataset(DATA_ROOT)
check("200 samples", len(ds) == 200)
check("Class names", ds.CLASS_NAMES == ["normal", "mild", "moderate", "severe"])
img, label = ds[0]
check("PIL (512,256)", hasattr(img, 'size') and img.size == (512, 256))
check("Label 0-3", 0 <= label <= 3)
splits = generate_split_indices(ds, 5, seed=42)
check("5 folds", len(splits) == 5)
check("Train ~160", len(splits[0]["train"]) >= 155)
check("Val ~40", len(splits[0]["val"]) >= 35)

# ---- 2. DataLoader ----
print("\n2. DataLoader & Augmentations")
from data.augmentations import get_train_transform, get_val_transform
from data.dataset import create_dataloaders

train_tf = get_train_transform(SMALL_SIZE)
val_tf = get_val_transform(SMALL_SIZE)
train_loader, val_loader = create_dataloaders(
    ds, splits[0], 2, train_tf, val_tf, num_workers=0, pin_memory=False)

bx, by = next(iter(train_loader))
bs = bx.shape[0]
check(f"Train batch [2,3,{SMALL_SIZE[0]},{SMALL_SIZE[1]}]",
      bx.shape == (bs, 3, SMALL_SIZE[0], SMALL_SIZE[1]))
check("Images normalized", bx.mean().abs() < 2)

bx_v, by_v = next(iter(val_loader))
check("Val batch loaded", bx_v.shape[0] >= 1)

# ---- 3. SymMamba Forward ----
print("\n3. SymMamba Forward (small input)")
from models.symmamba import SymMamba

model = SymMamba(num_classes=4, dropout=0.0).to(DEVICE)
n = sum(p.numel() for p in model.parameters())
check(f"Params ~4M ({n/1e6:.2f}M)", 3.5e6 <= n <= 5e6)

model.train()
x = bx.to(DEVICE)
out = model(x)
for k in ["alpha", "belief", "u", "pred", "features", "s_sym"]:
    check(f"Key '{k}'", k in out)
check("alpha shape", out["alpha"].shape == (bs, 4))
check("u shape", out["u"].shape == (bs,))
check("features dim", out["features"].shape[-1] == 768)
check("alpha >= 1", bool((out["alpha"] >= 1).all()))
check("u in (0,1]", bool(((out["u"] > 0) & (out["u"] <= 1)).all()))

# ---- 4. Loss ----
print("\n4. Loss Functions")
from models.losses import compute_total_loss

loss_cfg = {"edl": {"weight": 1.0}, "ordinal": {"weight": 0.1},
            "contrastive": {"weight": 0.05, "temperature": 0.07},
            "boundary": {"weight": 0.01}}
total, comps = compute_total_loss(out["alpha"], by, out["features"], loss_cfg)
check("Loss finite", bool(torch.isfinite(total)))
check("L_EDL > 0", comps["L_EDL"] > 0)
check("L_ord >= 0", comps["L_ord"] >= 0)

# ---- 5. Training Step ----
print("\n5. Training Step")
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
optimizer.zero_grad()
out2 = model(x)
loss, _ = compute_total_loss(out2["alpha"], by, out2["features"], loss_cfg)
loss.backward()
gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
optimizer.step()
check("Backward+step OK", True)
check("Grad norm finite", bool(torch.isfinite(gn)))
check("Grad norm > 0", gn > 0)

# ---- 6. Validation (1 batch) ----
print("\n6. Validation & Metrics (1 batch)")
from utils.metrics import compute_qwk, compute_sdr, calibrate_threshold, compute_all_metrics

model.eval()
with torch.no_grad():
    out_v = model(bx_v.to(DEVICE))
    preds_v = out_v["pred"].numpy()
    labels_v = by_v.numpy()
    u_v = out_v["u"].numpy()

qwk = compute_qwk(labels_v, preds_v)
check("QWK in [-1,1]", -1 <= qwk <= 1)
cal = calibrate_threshold(labels_v, preds_v, u_v)
check("theta in (0,1]", 0 < cal["theta"] <= 1)
sdr, ret = compute_sdr(labels_v, preds_v, u_v, cal["theta"])
check("SDR >= 0", sdr >= 0)

# Run full val for metrics
all_preds, all_labels, all_u = [], [], []
with torch.no_grad():
    for bx_v_batch, by_v_batch in val_loader:
        out_v_batch = model(bx_v_batch.to(DEVICE))
        all_preds.append(out_v_batch["pred"])
        all_labels.append(by_v_batch)
        all_u.append(out_v_batch["u"])
preds = torch.cat(all_preds).numpy()
labels = torch.cat(all_labels).numpy()
u_arr = torch.cat(all_u).numpy()
met = compute_all_metrics(labels, preds, u=u_arr)
check("Full val metrics", "qwk" in met)
print(f"  Untrained QWK={met['qwk']:.4f} SDR={met.get('sdr',0):.4f} (expected near random)")

# ---- 7. Checkpoint ----
print("\n7. Checkpoint IO")
from utils.logger import ExperimentLogger

with tempfile.TemporaryDirectory() as tmpdir:
    logger = ExperimentLogger(tmpdir, "smoke")
    logger.write_row(0, {"train_loss": 1.5, "val_qwk": 0.6})
    logger.save_checkpoint(model, 0, 0.6, mode="max", optimizer=optimizer,
                           extra={"theta": 0.5})
    logger.save_summary(met)
    logger.close()
    ckpt = torch.load(os.path.join(tmpdir, "smoke", "checkpoints", "best.pt"),
                       map_location=DEVICE, weights_only=False)
    check("Has state_dict", "model_state_dict" in ckpt)
    check("theta=0.5", ckpt["extra"]["theta"] == 0.5)
    check("CSV+JSON exist", os.path.exists(os.path.join(tmpdir, "smoke", "metrics.csv")))

# Roundtrip
model2 = SymMamba(num_classes=4, dropout=0.0).to(DEVICE)
model2.load_state_dict(ckpt["model_state_dict"])
check("Roundtrip load OK", True)

# ---- 8. Baselines ----
print("\n8. Baseline (ResNet50, cached)")
from models.baselines import create_baseline

b1 = create_baseline("resnet50", num_classes=4, use_edl=True,
                      edl_hidden=128, edl_dropout=0.0).to(DEVICE)
b1_out = b1(torch.randn(2, 3, 256, 512).to(DEVICE))
check("B1 alpha+pred", "alpha" in b1_out and "pred" in b1_out)

# ---- Summary ----
print("\n" + "=" * 60)
print(f"RESULTS: {passed} passed, {failed} failed  ({passed+failed} checks)")
if failed == 0:
    print("SMOKE TEST: ALL PASSED")
else:
    print(f"SMOKE TEST: {failed} FAILURES")
    sys.exit(1)
print("=" * 60)
