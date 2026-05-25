"""评价指标: QWK, SDR, Macro F1, ECE, UR-Acc."""

import numpy as np
from numpy.typing import NDArray


def _qwk_weight_matrix(K: int) -> NDArray:
    i = np.arange(K)
    return (i[:, None] - i[None, :]) ** 2 / (K - 1) ** 2


def compute_qwk(y_true: NDArray, y_pred: NDArray) -> float:
    """Quadratic Weighted Kappa.

    Args:
        y_true: [N] integer labels.
        y_pred: [N] integer predictions.
    Returns: QWK in [-1, 1].
    """
    K = max(y_true.max(), y_pred.max()) + 1
    W = _qwk_weight_matrix(K)
    O = np.zeros((K, K))
    for t, p in zip(y_true, y_pred):
        O[t, p] += 1
    E_rows = O.sum(axis=1, keepdims=True)
    E_cols = O.sum(axis=0, keepdims=True)
    E = (E_rows @ E_cols) / O.sum()
    num = (W * O).sum()
    den = (W * E).sum()
    return 1.0 - num / den if den > 0 else 0.0


def compute_sdr(y_true: NDArray, y_pred: NDArray, u: NDArray,
                theta: float) -> tuple[float, float]:
    """Severe Detection Rate at uncertainty threshold theta.

    Returns: (SDR, retention_rate)
    """
    severe_idx = 3
    confident = u <= theta
    if confident.sum() == 0:
        return 0.0, 0.0
    mask = confident & (y_true == severe_idx)
    tp = (y_pred[mask] == severe_idx).sum()
    fn = ((~confident) & (y_true == severe_idx)).sum() + \
         (confident & (y_true == severe_idx) & (y_pred != severe_idx)).sum()
    total_severe = tp + fn
    sdr = tp / total_severe if total_severe > 0 else 0.0
    retention = confident.mean()
    return float(sdr), float(retention)


def calibrate_threshold(y_true: NDArray, y_pred: NDArray, u: NDArray,
                         theta_range=(0.1, 0.9), step=0.05,
                         sdr_target=0.95) -> dict:
    """Calibrate theta: max retention s.t. SDR >= target.

    Returns: {"theta": float, "sdr": float, "retention": float}
    """
    best = {"theta": 1.0, "sdr": 0.0, "retention": 0.0}
    for th in np.arange(theta_range[0], theta_range[1] + step * 0.5, step):
        sdr, ret = compute_sdr(y_true, y_pred, u, th)
        if sdr >= sdr_target and ret > best["retention"]:
            best = {"theta": float(th), "sdr": sdr, "retention": ret}
    return best


def compute_macro_f1(y_true: NDArray, y_pred: NDArray) -> float:
    """Macro-averaged F1 across all classes."""
    K = max(y_true.max(), y_pred.max()) + 1
    f1s = []
    for k in range(K):
        tp = ((y_pred == k) & (y_true == k)).sum()
        fp = ((y_pred == k) & (y_true != k)).sum()
        fn = ((y_pred != k) & (y_true == k)).sum()
        p = tp / (tp + fp) if tp + fp > 0 else 0.0
        r = tp / (tp + fn) if tp + fn > 0 else 0.0
        f1 = 2 * p * r / (p + r) if p + r > 0 else 0.0
        f1s.append(f1)
    return float(np.mean(f1s))


def compute_ece(y_true: NDArray, probs: NDArray, n_bins=10) -> float:
    """Expected Calibration Error.

    Args:
        y_true: [N] integer labels.
        probs: [N, K] predicted probabilities.
        n_bins: number of bins.
    """
    conf = probs.max(axis=-1)
    pred = probs.argmax(axis=-1)
    acc = (pred == y_true).astype(np.float32)
    ece = 0.0
    N = len(y_true)
    bin_edges = np.linspace(0, 1, n_bins + 1)
    for i in range(n_bins):
        mask = (conf >= bin_edges[i]) & (conf < bin_edges[i + 1])
        if mask.sum() == 0:
            continue
        bin_acc = acc[mask].mean()
        bin_conf = conf[mask].mean()
        ece += (mask.sum() / N) * abs(bin_acc - bin_conf)
    return float(ece)


def compute_ur_accuracy(y_true: NDArray, y_pred: NDArray, u: NDArray,
                         theta: float) -> tuple[float, float]:
    """Uncertainty-Referred Accuracy at threshold theta.

    Returns: (UR-Acc, retention_rate)
    """
    mask = u <= theta
    if mask.sum() == 0:
        return 0.0, 0.0
    acc = (y_pred[mask] == y_true[mask]).mean()
    return float(acc), float(mask.mean())


def compute_all_metrics(y_true: NDArray, y_pred: NDArray,
                         alpha: NDArray = None, u: NDArray = None,
                         theta: float = 0.5) -> dict:
    """Compute all evaluation metrics.

    Args:
        y_true: [N] integer labels.
        y_pred: [N] integer predictions.
        alpha: [N, K] Dirichlet parameters (for ECE via belief).
        u: [N] uncertainty values.
        theta: uncertainty threshold (default fallback, use calibrate_threshold first).

    Returns: dict with qwk, macro_f1, ece, sdr, retention, ur_acc.
    """
    results = {}
    results["qwk"] = compute_qwk(y_true, y_pred)
    results["accuracy"] = float((y_pred == y_true).mean())
    results["macro_f1"] = compute_macro_f1(y_true, y_pred)

    if u is not None:
        sdr, retention = compute_sdr(y_true, y_pred, u, theta)
        ur_acc, _ = compute_ur_accuracy(y_true, y_pred, u, theta)
        results["sdr"] = sdr
        results["retention"] = retention
        results["ur_acc"] = ur_acc

    if alpha is not None:
        S = alpha.sum(axis=-1, keepdims=True)
        probs = (alpha - 1) / S
        results["ece"] = compute_ece(y_true, probs)
    else:
        results["ece"] = 0.0

    return results
