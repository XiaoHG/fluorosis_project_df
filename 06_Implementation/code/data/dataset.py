"""氟斑牙数据集加载, 支持 5-fold CV 和标签映射."""

import json
from pathlib import Path
from PIL import Image
from sklearn.model_selection import StratifiedKFold

import torch
from torch.utils.data import Dataset, DataLoader


class FluorosisDataset(Dataset):
    """从 {root}/{class_name}/*.png 加载氟斑牙图像.

    Args:
        root: 数据集根目录, 包含 normal/mild/moderate/severe 子文件夹.
        transform: torchvision transform.
    """

    CLASS_NAMES = ["normal", "mild", "moderate", "severe"]

    def __init__(self, root: str, transform=None):
        self.root = Path(root)
        self.transform = transform
        self.samples = []

        for label, cls_name in enumerate(self.CLASS_NAMES):
            cls_dir = self.root / cls_name
            if not cls_dir.is_dir():
                raise FileNotFoundError(f"类别目录不存在: {cls_dir}")
            for img_path in sorted(cls_dir.glob("*.png")):
                if img_path.name.startswith("._"):
                    continue
                self.samples.append((str(img_path), label))

        if len(self.samples) == 0:
            raise RuntimeError(f"在 {root} 下未找到任何 PNG 图像")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label

    @property
    def labels(self):
        return [s[1] for s in self.samples]


def generate_split_indices(dataset: FluorosisDataset, n_folds: int = 5,
                           seed: int = 42, save_path: str = None) -> list[dict]:
    """生成分层 K-fold 索引, 保存为 JSON.

    Returns:
        [{"fold": 0, "train": [...], "val": [...], ...}, ...]
    """
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    labels = dataset.labels

    splits = []
    for fold, (train_idx, val_idx) in enumerate(skf.split(range(len(dataset)), labels)):
        splits.append({
            "fold": fold,
            "train": [int(i) for i in train_idx],
            "val": [int(i) for i in val_idx],
        })

    if save_path:
        with open(save_path, "w") as f:
            json.dump(splits, f, indent=2)

    return splits


def load_split_indices(path: str) -> list[dict]:
    with open(path) as f:
        return json.load(f)


class _SubsetWithTransform(Dataset):
    """Subset with dedicated transform (avoids transform arg duplication)."""
    def __init__(self, dataset: Dataset, indices: list[int], transform=None):
        self.dataset = dataset
        self.indices = indices
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        img_path, label = self.dataset.samples[self.indices[idx]]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label


def create_dataloaders(dataset: FluorosisDataset, split: dict, batch_size: int,
                       train_transform, val_transform,
                       num_workers: int = 4, pin_memory: bool = True):
    """为指定 fold 创建 train/val DataLoader."""
    train_ds = _SubsetWithTransform(dataset, split["train"], train_transform)
    val_ds = _SubsetWithTransform(dataset, split["val"], val_transform)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=pin_memory,
                              drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=pin_memory)
    return train_loader, val_loader
