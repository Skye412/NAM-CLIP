import os
import random
from typing import Dict, List, Optional

import numpy as np
try:
    from scipy import ndimage
    from scipy.ndimage import gaussian_filter
except ImportError:
    ndimage = None
    gaussian_filter = None
import torch
import torch.nn.functional as F

try:
    from sklearn.metrics import auc, average_precision_score, precision_recall_curve, roc_auc_score
except ImportError:
    def auc(x, y):
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        order = np.argsort(x, kind="mergesort")
        return float(np.trapz(y[order], x[order]))

    def precision_recall_curve(y_true, y_score):
        y_true = np.asarray(y_true, dtype=np.int64)
        y_score = np.asarray(y_score, dtype=np.float64)
        desc_idx = np.argsort(y_score, kind="mergesort")[::-1]
        y_score = y_score[desc_idx]
        y_true = y_true[desc_idx]
        distinct_value_indices = np.where(np.diff(y_score))[0]
        threshold_idxs = np.r_[distinct_value_indices, y_true.size - 1]
        tps = np.cumsum(y_true)[threshold_idxs].astype(np.float64)
        fps = (1 + threshold_idxs - tps).astype(np.float64)
        precision = tps / np.maximum(tps + fps, 1e-12)
        recall = tps / max(tps[-1], 1.0)
        sl = slice(None, None, -1)
        return np.r_[precision[sl], 1.0], np.r_[recall[sl], 0.0], y_score[threshold_idxs][sl]

    def average_precision_score(y_true, y_score):
        precision, recall, _ = precision_recall_curve(y_true, y_score)
        return float(-np.sum(np.diff(recall) * precision[:-1]))

    def roc_auc_score(y_true, y_score):
        y_true = np.asarray(y_true, dtype=np.int64)
        y_score = np.asarray(y_score, dtype=np.float64)
        desc_idx = np.argsort(y_score, kind="mergesort")[::-1]
        y_score = y_score[desc_idx]
        y_true = y_true[desc_idx]
        distinct_value_indices = np.where(np.diff(y_score))[0]
        threshold_idxs = np.r_[distinct_value_indices, y_true.size - 1]
        tps = np.cumsum(y_true)[threshold_idxs].astype(np.float64)
        fps = (1 + threshold_idxs - tps).astype(np.float64)
        tps = np.r_[0.0, tps]
        fps = np.r_[0.0, fps]
        tpr = tps / max(tps[-1], 1.0)
        fpr = fps / max(fps[-1], 1.0)
        return float(np.trapz(tpr, fpr))

DEFECT_CLASSES = ["Crack", "Spalling", "Corrosion", "Efflorescence"]


def setup_seed(seed: int = 111):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def norm01_np(arr: np.ndarray) -> np.ndarray:
    arr = arr.astype(np.float32, copy=False)
    mn, mx = float(arr.min()), float(arr.max())
    if mx <= mn + 1e-12:
        return np.zeros_like(arr, dtype=np.float32)
    return (arr - mn) / (mx - mn)


def fast_aupro(masks: np.ndarray, amaps: np.ndarray, num_th: int = 40, max_fpr: float = 0.3) -> float:
    if masks.sum() == 0:
        return 0.0
    min_th, max_th = float(amaps.min()), float(amaps.max())
    if max_th <= min_th + 1e-12:
        return 0.0

    thresholds = np.linspace(min_th, max_th, num_th, endpoint=False, dtype=np.float32)
    bins = np.concatenate([[-np.inf], np.nextafter(thresholds, np.inf), [np.inf]])

    comp_pros_list = []
    for i in range(len(masks)):
        if masks[i].sum() == 0:
            continue
        labeled, _ = ndimage.label(masks[i] > 0.5)
        objs = ndimage.find_objects(labeled)
        for comp_id, slc in enumerate(objs, start=1):
            if slc is None:
                continue
            sub_mask = (labeled[slc] == comp_id)
            area = float(sub_mask.sum())
            if area > 0:
                comp_vals = amaps[i][slc][sub_mask]
                c_hist, _ = np.histogram(comp_vals, bins=bins)
                c_tp = np.cumsum(c_hist[::-1])[::-1][1:].astype(np.float64)
                comp_pros_list.append(c_tp / area)

    if not comp_pros_list:
        return 0.0

    neg_vals = amaps[masks <= 0.5]
    total_neg = float(len(neg_vals))
    if total_neg == 0:
        return 0.0

    neg_hist, _ = np.histogram(neg_vals, bins=bins)
    fp_arr = np.cumsum(neg_hist[::-1])[::-1][1:].astype(np.float64)
    fpr_arr = fp_arr / total_neg
    mean_pros = np.mean(np.stack(comp_pros_list, axis=0), axis=0)

    valid = (fpr_arr <= max_fpr)
    fprs = fpr_arr[valid]
    pros = mean_pros[valid]

    if len(fprs) < 2:
        return 0.0
    order = np.argsort(fprs)
    fprs = fprs[order]
    pros = pros[order]
    fpr_span = float(fprs.max() - fprs.min())
    if fpr_span <= 1e-12:
        return float(pros.mean())
    fpr_norm = (fprs - fprs.min()) / fpr_span
    return float(auc(fpr_norm, pros))


# ----------------------------------------------------------------------------
# Rank-invariant AUPRO (standard MVTec definition)
# ----------------------------------------------------------------------------
# `fast_aupro` above places its 40 thresholds uniformly in SCORE space and rescales the
# retained FPR span to [0, 1]. Its value therefore depends on how peaked the score
# histogram is, not only on the ranking: a sharp map reaches FPR <= 0.3 within the first
# one or two thresholds, so the integral covers only a narrow low-FPR slice. A monotone
# transform of the same map changes the result. The functions below place thresholds at
# quantiles of the negative scores, so FPR is sampled evenly on [0, max_fpr], and
# normalise the integral by max_fpr. `fast_aupro` is kept only to reproduce old tables.
def _trapz(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    return float(np.sum(np.diff(x) * (y[1:] + y[:-1]) * 0.5))


def _neg_quantile_bins(neg_vals: np.ndarray, max_fpr: float = 0.3, num_th: int = 300,
                       max_sample: int = 2_000_000, seed: int = 0):
    """Histogram edges at quantiles of the negative scores, plus the exact FPR at each edge."""
    if neg_vals.size > max_sample:
        rng = np.random.RandomState(seed)
        sample = neg_vals[rng.randint(0, neg_vals.size, size=max_sample)]
    else:
        sample = neg_vals
    targets = np.linspace(0.0, min(1.0, max_fpr * 1.1), num_th + 1)[1:]
    ths = np.unique(np.quantile(sample.astype(np.float64), 1.0 - targets).astype(np.float32))
    bins = np.concatenate([[-np.inf], ths.astype(np.float64), [np.inf]])
    neg_hist, _ = np.histogram(neg_vals, bins=bins)
    # number of negatives with score >= ths[k]  ==  sum(neg_hist[k + 1:])
    fpr = np.cumsum(neg_hist[::-1])[::-1][1:].astype(np.float64) / float(neg_vals.size)
    return bins, fpr


def _pro_curves(comp_masks: np.ndarray, amaps: np.ndarray, bins: np.ndarray):
    """Per-connected-component overlap at every threshold. Returns ([n_comp, n_th], areas)."""
    curves, areas = [], []
    for i in range(len(comp_masks)):
        if comp_masks[i].sum() == 0:
            continue
        labeled, _ = ndimage.label(comp_masks[i] > 0.5)
        for comp_id, slc in enumerate(ndimage.find_objects(labeled), start=1):
            if slc is None:
                continue
            sub = (labeled[slc] == comp_id)
            area = float(sub.sum())
            if area <= 0:
                continue
            c_hist, _ = np.histogram(amaps[i][slc][sub], bins=bins)
            curves.append(np.cumsum(c_hist[::-1])[::-1][1:].astype(np.float64) / area)
            areas.append(area)
    if not curves:
        return None, np.zeros(0, dtype=np.float64)
    return np.stack(curves, axis=0), np.asarray(areas, dtype=np.float64)


def _integrate_pro(fpr: np.ndarray, pro: np.ndarray, max_fpr: float = 0.3) -> float:
    """Area under PRO(FPR) on [0, max_fpr], divided by max_fpr (range [0, 1])."""
    x = np.concatenate([[0.0], fpr, [1.0]])
    y = np.concatenate([[0.0], pro, [1.0]])
    order = np.lexsort((y, x))
    x, y = x[order], y[order]
    xu, start = np.unique(x, return_index=True)
    yu = np.maximum.accumulate(np.maximum.reduceat(y, start))
    grid = np.concatenate([xu[xu < max_fpr], [max_fpr]])
    return _trapz(grid, np.interp(grid, xu, yu)) / max_fpr


def aupro_rank_invariant(masks: np.ndarray, amaps: np.ndarray, num_th: int = 300, max_fpr: float = 0.3) -> float:
    """Standard AUPRO: invariant to any strictly increasing transform of `amaps`."""
    if ndimage is None or masks.sum() == 0:
        return 0.0
    if float(amaps.max()) <= float(amaps.min()) + 1e-12:
        return 0.0
    neg_vals = amaps[masks <= 0.5]
    if neg_vals.size == 0:
        return 0.0
    bins, fpr = _neg_quantile_bins(neg_vals, max_fpr=max_fpr, num_th=num_th)
    curves, _ = _pro_curves(masks, amaps, bins)
    if curves is None:
        return 0.0
    return _integrate_pro(fpr, curves.mean(axis=0), max_fpr=max_fpr)


def _hist_ap(pos_vals: np.ndarray, neg_hist: np.ndarray, bins: np.ndarray) -> float:
    """Average precision (in %) of `pos_vals` against negatives already histogrammed on `bins`."""
    total_pos = float(len(pos_vals))
    total_neg = float(neg_hist.sum())
    if total_pos == 0 or total_neg == 0:
        return 0.0
    pos_hist, _ = np.histogram(pos_vals, bins=bins)
    tp = np.cumsum(pos_hist[::-1])[::-1][1:].astype(np.float64)
    fp = np.cumsum(neg_hist[::-1])[::-1][1:].astype(np.float64)
    rec = (tp / total_pos)[::-1]
    prec = (tp / np.maximum(tp + fp, 1e-12))[::-1]
    rec_inc = np.concatenate([[0.0], rec, [1.0]])
    prec_inc = np.concatenate([[1.0], prec, [total_pos / (total_pos + total_neg)]])
    return float(np.sum((rec_inc[1:] - rec_inc[:-1]) * prec_inc[1:])) * 100.0


def evaluate_anomaly_predictions(
    gt_sp: np.ndarray,
    pr_sp: np.ndarray,
    masks_np: np.ndarray,
    amaps_raw: np.ndarray,
    class_masks_dict: Optional[Dict[str, List[np.ndarray]]] = None,
    nuisance_masks_np: Optional[np.ndarray] = None,
    apply_gaussian_sigma: float = 4.0,
    compute_pro: bool = True,
) -> Dict:
    """
    Comprehensive evaluation function aligned with server benchmark protocol:
      - Image-AUROC, Image-F1, Image-AP
      - Pixel-AUROC, Pixel-F1, Pixel-AP, Pixel-AUPRO, Pixel-IoU
      - Mean-Score (Mean-7) & Mean-5
      - Per-Class Recall (Crack, Spalling, Corrosion, Efflorescence)
      - Nuisance-FPR (on Official S2DS Vegetation + ControlPoint pixels) & Normal-Image FPR
    """
    if apply_gaussian_sigma > 0:
        # Ultra-fast GPU separable 2D Gaussian filter (0.01s on A100 vs 15s on CPU scipy)
        if torch.cuda.is_available():
            with torch.no_grad():
                radius = int(4.0 * apply_gaussian_sigma + 0.5)
                coords = torch.arange(-radius, radius + 1, dtype=torch.float32, device="cuda")
                k1d = torch.exp(-(coords ** 2) / (2.0 * apply_gaussian_sigma ** 2))
                k1d = (k1d / k1d.sum()).view(1, 1, -1)
                kx = k1d.unsqueeze(2)  # [1, 1, 1, K]
                ky = k1d.unsqueeze(3)  # [1, 1, K, 1]
                amaps_smooth = np.empty_like(amaps_raw, dtype=np.float32)
                bs = 64
                for i in range(0, len(amaps_raw), bs):
                    t = torch.from_numpy(amaps_raw[i : i + bs]).cuda().unsqueeze(1)
                    t = F.pad(t, (radius, radius, radius, radius), mode="reflect")
                    t = F.conv2d(F.conv2d(t, kx), ky)
                    amaps_smooth[i : i + bs] = t.squeeze(1).cpu().numpy()
        else:
            amaps_smooth = np.empty_like(amaps_raw, dtype=np.float32)
            for i in range(len(amaps_raw)):
                amaps_smooth[i] = gaussian_filter(amaps_raw[i], sigma=apply_gaussian_sigma)
    else:
        amaps_smooth = amaps_raw.astype(np.float32, copy=False)

    amaps_norm = norm01_np(amaps_smooth)
    pr_sp_norm = norm01_np(pr_sp)

    if len(np.unique(gt_sp)) > 1:
        i_auroc = float(roc_auc_score(gt_sp, pr_sp_norm)) * 100.0
        i_ap = float(average_precision_score(gt_sp, pr_sp_norm)) * 100.0
        prec_s, rec_s, _ = precision_recall_curve(gt_sp, pr_sp_norm)
        i_f1 = float(np.max(2 * prec_s * rec_s / (prec_s + rec_s + 1e-8))) * 100.0
    else:
        i_auroc, i_ap, i_f1 = 50.0, 50.0, 50.0

    gt_px = masks_np.reshape(-1).astype(np.uint8)
    pr_px = amaps_norm.reshape(-1).astype(np.float32)

    pos_vals = pr_px[gt_px == 1]
    neg_vals = pr_px[gt_px == 0]
    total_pos = max(float(len(pos_vals)), 1.0)
    total_neg = max(float(len(neg_vals)), 1.0)

    fine_ths = np.linspace(0.0, 1.0, 4097, dtype=np.float32)
    bins_fine = np.concatenate([[-np.inf], fine_ths, [np.inf]])
    pos_hist_f, _ = np.histogram(pos_vals, bins=bins_fine)
    neg_hist_f, _ = np.histogram(neg_vals, bins=bins_fine)
    tp_f = np.cumsum(pos_hist_f[::-1])[::-1][1:].astype(np.float64)
    fp_f = np.cumsum(neg_hist_f[::-1])[::-1][1:].astype(np.float64)
    tpr_f = np.concatenate([[0.0], (tp_f / total_pos)[::-1], [1.0]])
    fpr_f = np.concatenate([[0.0], (fp_f / total_neg)[::-1], [1.0]])
    p_auroc = float(auc(fpr_f, tpr_f)) * 100.0

    rec_f = (tp_f / total_pos)[::-1]
    prec_f = (tp_f / np.maximum(tp_f + fp_f, 1e-12))[::-1]
    rec_inc = np.concatenate([[0.0], rec_f, [1.0]])
    prec_inc = np.concatenate([[1.0], prec_f, [total_pos / (total_pos + total_neg)]])
    p_ap = float(np.sum((rec_inc[1:] - rec_inc[:-1]) * prec_inc[1:])) * 100.0

    ths = np.linspace(0.01, 0.99, 99, dtype=np.float32)
    bins_99 = np.concatenate([[-np.inf], ths, [np.inf]])
    pos_hist_99, _ = np.histogram(pos_vals, bins=bins_99)
    neg_hist_99, _ = np.histogram(neg_vals, bins=bins_99)
    tp_99 = np.cumsum(pos_hist_99[::-1])[::-1][1:].astype(np.float64)
    fp_99 = np.cumsum(neg_hist_99[::-1])[::-1][1:].astype(np.float64)
    pred_pos_99 = tp_99 + fp_99
    f1_arr = (2.0 * tp_99) / (pred_pos_99 + total_pos + 1e-8)
    iou_arr = tp_99 / (pred_pos_99 + total_pos - tp_99 + 1e-8)
    best_idx = int(np.argmax(f1_arr))
    p_f1 = float(f1_arr[best_idx]) * 100.0
    p_iou = float(iou_arr[best_idx]) * 100.0
    best_th = float(ths[best_idx])
    # Pixel-AUPRO is the rank-invariant value; the old score-uniform value is kept as "-legacy".
    p_aupro, p_aupro_legacy = 0.0, 0.0
    pro_by_size = {}
    bins_q, fpr_q = None, None
    do_pro = (
        bool(compute_pro) and ndimage is not None and masks_np.sum() > 0 and neg_vals.size > 0
        and float(amaps_norm.max()) > float(amaps_norm.min()) + 1e-12
    )
    if do_pro:
        p_aupro_legacy = fast_aupro(masks_np, amaps_norm, num_th=40, max_fpr=0.3) * 100.0
        bins_q, fpr_q = _neg_quantile_bins(neg_vals, max_fpr=0.3, num_th=300)
        curves_all, areas_all = _pro_curves(masks_np, amaps_norm, bins_q)
        if curves_all is not None:
            p_aupro = _integrate_pro(fpr_q, curves_all.mean(axis=0), max_fpr=0.3) * 100.0
            # Diagnostic: AUPRO of small / medium / large ground-truth regions (area terciles).
            # The cut points depend only on the ground truth, so they are identical for every model.
            q1, q2 = np.quantile(areas_all, [1.0 / 3.0, 2.0 / 3.0])
            for tag, sel in (("small", areas_all <= q1),
                             ("medium", (areas_all > q1) & (areas_all <= q2)),
                             ("large", areas_all > q2)):
                if sel.any():
                    pro_by_size[f"AUPRO_{tag}"] = round(
                        _integrate_pro(fpr_q, curves_all[sel].mean(axis=0), max_fpr=0.3) * 100.0, 2)
            pro_by_size["area_cut_px"] = [round(float(q1), 1), round(float(q2), 1)]
            pro_by_size["num_regions"] = int(len(areas_all))

    pred_2d = (amaps_norm >= best_th)
    class_metrics = {}
    if class_masks_dict:
        for cname in DEFECT_CLASSES:
            if cname in class_masks_dict and len(class_masks_dict[cname]) > 0:
                cm_np = np.stack(class_masks_dict[cname], axis=0) > 0.5
                c_total = float(cm_np.sum())
                if c_total > 0:
                    c_tp = float((pred_2d & cm_np).sum())
                    class_metrics[f"{cname}_Recall"] = round((c_tp / c_total) * 100.0, 2)
                    # Threshold-free per-class scores. Positives: pixels of this class.
                    # Negatives: defect-free pixels (pixels of other defect classes are excluded).
                    # AP depends on the class pixel prevalence, so compare models within a class.
                    class_metrics[f"{cname}_AP"] = round(_hist_ap(amaps_norm[cm_np], neg_hist_f, bins_fine), 2)
                    if do_pro and bins_q is not None:
                        c_curves, _ = _pro_curves(cm_np, amaps_norm, bins_q)
                        if c_curves is not None:
                            class_metrics[f"{cname}_AUPRO"] = round(
                                _integrate_pro(fpr_q, c_curves.mean(axis=0), max_fpr=0.3) * 100.0, 2)

    # False positive analysis on normal images & explicit S2DS nuisance masks (Vegetation + ControlPoint)
    normal_img_mask = (gt_sp == 0)
    if normal_img_mask.any():
        norm_fpr = float(pred_2d[normal_img_mask].mean()) * 100.0
    else:
        norm_fpr = 0.0

    nuis_fpr = 0.0
    if nuisance_masks_np is not None:
        nuis_bool = (nuisance_masks_np > 0.5) & (masks_np <= 0.5)
        if nuis_bool.any():
            nuis_fpr = float(pred_2d[nuis_bool].mean()) * 100.0

    mean_score = (i_auroc + i_f1 + i_ap + p_auroc + p_f1 + p_ap + p_aupro) / 7.0
    mean_5 = (i_auroc + p_auroc + p_f1 + p_ap + p_aupro) / 5.0
    mean_score_legacy = (i_auroc + i_f1 + i_ap + p_auroc + p_f1 + p_ap + p_aupro_legacy) / 7.0
    mean_5_legacy = (i_auroc + p_auroc + p_f1 + p_ap + p_aupro_legacy) / 5.0

    return {
        "Image-AUROC": round(i_auroc, 2),
        "Image-F1": round(i_f1, 2),
        "Image-AP": round(i_ap, 2),
        "Pixel-AUROC": round(p_auroc, 2),
        "Pixel-F1": round(p_f1, 2),
        "Pixel-AP": round(p_ap, 2),
        "Pixel-AUPRO": round(p_aupro, 2),
        "Pixel-AUPRO-legacy": round(p_aupro_legacy, 2),
        "AUPRO-Impl": "quantile-v2",
        "Pixel-IoU": round(p_iou, 2),
        "Mean-Score": round(mean_score, 2),
        "Mean-5": round(mean_5, 2),
        "Mean-Score-legacy": round(mean_score_legacy, 2),
        "Mean-5-legacy": round(mean_5_legacy, 2),
        "AUPRO-By-Size": pro_by_size,
        "Best-Th": round(best_th, 4),
        "Normal-FPR": round(norm_fpr, 2),
        "Nuisance-FPR": round(nuis_fpr, 2),
        "Per-Class": class_metrics,
    }


if __name__ == "__main__":
    # Self-test: AUPRO must not change under a strictly increasing transform of the scores.
    rng = np.random.RandomState(0)
    n, hw = 24, 128
    masks = np.zeros((n, hw, hw), dtype=np.uint8)
    amaps = rng.beta(2, 6, size=(n, hw, hw)).astype(np.float32)
    for i in range(n):
        masks[i, 20:60, 20:60] = 1                       # large blob, strong response
        amaps[i, 20:60, 20:60] = rng.beta(6, 2, size=(40, 40))
        masks[i, 90, 10:110] = 1                         # thin line, weak response
        amaps[i, 90, 10:110] = rng.beta(3, 4, size=100)
    print(f"{'transform':<10}{'quantile-v2':>14}{'legacy':>10}")
    vals = []
    for name, g in (("v", lambda v: v), ("v**3", lambda v: v ** 3), ("v**8", lambda v: v ** 8), ("sqrt", np.sqrt)):
        a = norm01_np(g(amaps.astype(np.float64)).astype(np.float32))
        new, old = aupro_rank_invariant(masks, a) * 100.0, fast_aupro(masks, a) * 100.0
        vals.append(new)
        print(f"{name:<10}{new:>14.2f}{old:>10.2f}")
    spread = max(vals) - min(vals)
    print(f"quantile-v2 spread across transforms: {spread:.3f} (expected < 0.5)")
    assert spread < 0.5, "aupro_rank_invariant is not invariant to monotone transforms"
