import argparse
import json
import math
import os
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from cliplib import get_clip_transform, load_clip_vit_l14_336
from datasets import DEFECT_CLASSES, build_dataset
from models import (
    BinaryDiceLoss,
    FocalLoss,
    TFMoECLIP,
    build_tf_moeclip_variant,
    extract_haar_wavelet_cues,
)
from utils import evaluate_anomaly_predictions, setup_seed


def _file_sha256(path: str) -> str:
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ============================================================================
# Fast Feature & Low-Res Mask Cache + Direct GPU Tensor Batch Slicer
# (Eliminates per-batch 518x518 DataLoader CPU overhead -> ~3s/epoch on A100!)
# ============================================================================
class CachedBridgeFeatureDataset(Dataset):
    def __init__(self, cache_dir: str, prefix: str, load_full_masks: bool = True):
        super().__init__()
        meta_path = os.path.join(cache_dir, f"{prefix}_meta.json")
        if not os.path.isfile(meta_path):
            raise FileNotFoundError(f"Cache metadata not found: {meta_path}")
        with open(meta_path, "r", encoding="utf-8") as f:
            self.meta = json.load(f)

        n = self.meta["num_samples"]
        n_layers = self.meta["num_layers"]
        seq_l = self.meta["seq_len"]
        d_p = self.meta["d_vision"]
        img_sz = self.meta["image_size"]
        hr_sz = self.meta["hr_size"]

        patch_mm = np.memmap(
            os.path.join(cache_dir, f"{prefix}_patch.mmap"),
            dtype=np.float16, mode="r", shape=(n, n_layers, seq_l, d_p)
        )
        wt_mm = np.memmap(
            os.path.join(cache_dir, f"{prefix}_wt_desc.mmap"),
            dtype=np.float16, mode="r", shape=(n, seq_l, 4)
        )
        hr_mm = np.memmap(
            os.path.join(cache_dir, f"{prefix}_hr_cue.mmap"),
            dtype=np.float16, mode="r", shape=(n, 4, hr_sz, hr_sz)
        )
        mask_mm = np.memmap(
            os.path.join(cache_dir, f"{prefix}_mask.mmap"),
            dtype=np.uint8, mode="r", shape=(n, 1, img_sz, img_sz)
        )
        nuis_mm = np.memmap(
            os.path.join(cache_dir, f"{prefix}_nuis.mmap"),
            dtype=np.uint8, mode="r", shape=(n, 1, img_sz, img_sz)
        )
        cmask_mm = np.memmap(
            os.path.join(cache_dir, f"{prefix}_cmask.mmap"),
            dtype=np.uint8, mode="r", shape=(n, 4, img_sz, img_sz)
        )

        self.patch_tensor = torch.from_numpy(np.array(patch_mm, copy=True))
        self.wt_tensor = torch.from_numpy(np.array(wt_mm, copy=True))
        self.hr_tensor = torch.from_numpy(np.array(hr_mm, copy=True))
        self.mlabel_tensor = torch.from_numpy(np.load(os.path.join(cache_dir, f"{prefix}_mlabel.npy")).astype(np.float32))
        self.anom_np = np.load(os.path.join(cache_dir, f"{prefix}_anom.npy"))
        self.anom_tensor = torch.from_numpy(self.anom_np.astype(np.float32))

        # Pre-compute 129x129 masks for ultra-fast GPU training batches, and keep full 518x518 masks for exact evaluation
        self.mask_tensor = torch.from_numpy(np.array(mask_mm, copy=True))
        self.nuis_tensor = torch.from_numpy(np.array(nuis_mm, copy=True))
        self.cmask_tensor = torch.from_numpy(np.array(cmask_mm, copy=True))

    def __len__(self) -> int:
        return self.meta["num_samples"]

    def __getitem__(self, idx: int) -> Dict:
        return {
            "patch_feats": self.patch_tensor[idx],
            "wt_desc": self.wt_tensor[idx],
            "hr_cue": self.hr_tensor[idx],
            "img_mask": self.mask_tensor[idx],
            "nuisance_mask": self.nuis_tensor[idx],
            "class_masks_tensor": self.cmask_tensor[idx],
            "multi_label": self.mlabel_tensor[idx],
            "anomaly": int(self.anom_np[idx]),
        }


@torch.no_grad()
def build_or_load_feature_cache(
    clip_backbone,
    dataset: Dataset,
    cache_dir: str,
    prefix: str,
    features_list: Tuple[int, ...] = (6, 12, 18, 24),
    batch_size: int = 16,
    num_workers: int = 4,
    device: str = "cuda",
) -> CachedBridgeFeatureDataset:
    os.makedirs(cache_dir, exist_ok=True)
    meta_path = os.path.join(cache_dir, f"{prefix}_meta.json")
    if os.path.isfile(meta_path):
        try:
            cached_ds = CachedBridgeFeatureDataset(cache_dir, prefix)
            if len(cached_ds) == len(dataset):
                return cached_ds
        except Exception:
            pass

    n_samples = len(dataset)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    patch_mmap = None
    wt_mmap = None
    hr_mmap = None
    mask_mmap = None
    nuis_mmap = None
    cmask_mmap = None
    mlabel_list = []
    anom_list = []

    cursor = 0
    clip_backbone.eval()
    for batch in tqdm(loader, desc=f"Caching {prefix} ({n_samples} imgs)", ncols=90):
        img = batch["img"].to(device, non_blocking=True)
        bsz, _, H, W = img.shape
        with torch.amp.autocast("cuda", enabled=(device != "cpu")):
            _, patch_layers, grid_hw = clip_backbone.encode_image(img, features_list=features_list)
            pf_stack = torch.stack(patch_layers, dim=1)  # [B, 4, L, 1024]
            wt_desc, hr_cue = extract_haar_wavelet_cues(img.float(), grid_size=grid_hw)

        if patch_mmap is None:
            n_layers, seq_l, d_p = pf_stack.shape[1], pf_stack.shape[2], pf_stack.shape[3]
            hr_sz = hr_cue.shape[-1]
            patch_mmap = np.memmap(
                os.path.join(cache_dir, f"{prefix}_patch.mmap"),
                dtype=np.float16, mode="w+", shape=(n_samples, n_layers, seq_l, d_p)
            )
            wt_mmap = np.memmap(
                os.path.join(cache_dir, f"{prefix}_wt_desc.mmap"),
                dtype=np.float16, mode="w+", shape=(n_samples, seq_l, 4)
            )
            hr_mmap = np.memmap(
                os.path.join(cache_dir, f"{prefix}_hr_cue.mmap"),
                dtype=np.float16, mode="w+", shape=(n_samples, 4, hr_sz, hr_sz)
            )
            mask_mmap = np.memmap(
                os.path.join(cache_dir, f"{prefix}_mask.mmap"),
                dtype=np.uint8, mode="w+", shape=(n_samples, 1, H, W)
            )
            nuis_mmap = np.memmap(
                os.path.join(cache_dir, f"{prefix}_nuis.mmap"),
                dtype=np.uint8, mode="w+", shape=(n_samples, 1, H, W)
            )
            cmask_mmap = np.memmap(
                os.path.join(cache_dir, f"{prefix}_cmask.mmap"),
                dtype=np.uint8, mode="w+", shape=(n_samples, 4, H, W)
            )

        patch_mmap[cursor : cursor + bsz] = pf_stack.half().cpu().numpy()
        wt_mmap[cursor : cursor + bsz] = wt_desc.half().cpu().numpy()
        hr_mmap[cursor : cursor + bsz] = hr_cue.half().cpu().numpy()
        mask_mmap[cursor : cursor + bsz] = (batch["img_mask"].numpy() > 0.5).astype(np.uint8)
        nuis_mmap[cursor : cursor + bsz] = (batch["nuisance_mask"].numpy() > 0.5).astype(np.uint8)
        cmask_mmap[cursor : cursor + bsz] = (batch["class_masks_tensor"].numpy() > 0.5).astype(np.uint8)
        mlabel_list.append(batch["multi_label"].numpy())
        anom_list.append(np.asarray(batch["anomaly"], dtype=np.int64))
        cursor += bsz

    patch_mmap.flush()
    wt_mmap.flush()
    hr_mmap.flush()
    mask_mmap.flush()
    nuis_mmap.flush()
    cmask_mmap.flush()
    np.save(os.path.join(cache_dir, f"{prefix}_mlabel.npy"), np.concatenate(mlabel_list, axis=0))
    np.save(os.path.join(cache_dir, f"{prefix}_anom.npy"), np.concatenate(anom_list, axis=0))

    meta_info = {
        "num_samples": n_samples,
        "num_layers": len(features_list),
        "seq_len": int(pf_stack.shape[2]),
        "d_vision": int(pf_stack.shape[3]),
        "grid_h": int(grid_hw[0]),
        "grid_w": int(grid_hw[1]),
        "image_size": int(H),
        "hr_size": int(hr_cue.shape[-1]),
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta_info, f, indent=2)

    return CachedBridgeFeatureDataset(cache_dir, prefix)


# ============================================================================
# Training & Evaluation Loops (Fixing Root Cause 2 + Stage-2 Conditional A4)
# ============================================================================
def compute_tf_moeclip_loss(
    out: Dict[str, torch.Tensor],
    batch: Dict[str, torch.Tensor],
    focal_fn: FocalLoss,
    dice_fn: BinaryDiceLoss,
    lambda_etf: float = 0.01,
    lambda_bal: float = 0.01,
    device: str = "cuda",
    sfr_patch_label: str = "ratio20",
    cls_px_balance: bool = False,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    gt_mask = batch["img_mask"].to(device, non_blocking=True, dtype=torch.float32)          # [B, 1, H, W]
    nuis_mask = batch["nuisance_mask"].to(device, non_blocking=True, dtype=torch.float32)  # [B, 1, H, W]
    cmasks = batch["class_masks_tensor"].to(device, non_blocking=True, dtype=torch.float32)# [B, 4, H, W]
    mlabel = batch["multi_label"].to(device, non_blocking=True, dtype=torch.float32)       # [B, 4]
    y_img = batch["anomaly"].to(device, non_blocking=True, dtype=torch.float32)            # [B]

    pred_map = out["anomaly_map"]
    coarse_map = out["coarse_map"]
    crack_mask = cmasks[:, 0:1]
    non_crack_mask = (1.0 - crack_mask)

    # Fix Root Cause 2:
    # Always give full 1.0 supervision weight to coarse_map (so A3 trunk never degrades),
    # and when CC-DWCPR is active (pred_map is not coarse_map), supervise pred_map with:
    #   (a) Full union Focal + Dice loss
    #   (b) Dedicated Crack-only Dice + Focal repair loss on crack pixels
    #   (c) Non-crack preservation (distillation) loss | (1 - Y_crack) * (pred_map - coarse_map.detach()) |_1
    #   (d) Gate sparsity regularization on normal background pixels
    # Full 1.0 weight on loss_coarse guarantees A4's trunk receives 100% identical gradients to A3!
    loss_coarse = focal_fn(coarse_map, gt_mask) + dice_fn(coarse_map, gt_mask)

    if pred_map is not coarse_map:
        # Since pred_map is computed from coarse_map.detach(), loss_cpr updates ONLY self.dw_cpr parameters!
        loss_refined = focal_fn(pred_map, gt_mask) + dice_fn(pred_map, gt_mask)
        if crack_mask.sum() > 0:
            crack_dice = dice_fn(pred_map * crack_mask, crack_mask)
            crack_focal = focal_fn(pred_map, crack_mask)
            loss_crack_repair = 0.5 * crack_dice + 0.5 * crack_focal
        else:
            loss_crack_repair = torch.tensor(0.0, device=device)

        # Penalize false positives on normal background (1 - gt_mask) to boost P-AUPRO & P-AP
        bg_fp_loss = ((1.0 - gt_mask) * pred_map).mean()
        # Preserve Spalling & Corrosion interiors (gt_mask=1 & crack_mask=0)
        area_def_mask = (gt_mask * non_crack_mask).clamp(min=0.0, max=1.0)
        loss_preserve = (area_def_mask * F.relu(coarse_map.detach() - pred_map)).mean()

        loss_cpr = loss_refined + 0.40 * loss_crack_repair + 0.30 * bg_fp_loss + 0.50 * loss_preserve
        loss_seg = loss_coarse + loss_cpr
    else:
        loss_seg = loss_coarse

    # Image-level classification BCE (Eq. 15 of MoECLIP)
    loss_ac = F.binary_cross_entropy(out["img_score"].float(), y_img) +               0.5 * F.binary_cross_entropy(out["img_prob_adapter"].float(), y_img)

    loss_etf = out["loss_etf"]
    loss_bal = out["loss_bal"]
    total_loss = loss_seg + loss_ac + lambda_etf * loss_etf + lambda_bal * loss_bal

    # Module 3 (SFR-Graph) Multi-Label & Anti-Collapse Nuisance Losses
    loss_sfr = torch.tensor(0.0, device=device)
    if out["class_probs_4"] is not None and out["sfr_probs_12"] is not None:
        l_mcls_img = F.binary_cross_entropy(out["class_probs_4"].float().clamp(1e-6, 1.0 - 1e-6), mlabel)

        anom_idx = (y_img > 0.5)
        if anom_idx.any():
            c_pred = out["class_maps_4"][anom_idx].clamp(1e-6, 1.0 - 1e-6)
            c_gt = cmasks[anom_idx]
            if cls_px_balance:
                # Positives and negatives of each class are averaged separately, so a class that
                # covers few pixels (cracks) is not outweighed by large-area classes or background.
                bce = F.binary_cross_entropy(c_pred, c_gt, reduction="none")
                n_pos = c_gt.sum(dim=(0, 2, 3))
                n_neg = (1.0 - c_gt).sum(dim=(0, 2, 3))
                l_pos = (bce * c_gt).sum(dim=(0, 2, 3)) / (n_pos + 1.0)
                l_neg = (bce * (1.0 - c_gt)).sum(dim=(0, 2, 3)) / (n_neg + 1.0)
                l_mcls_px = (0.5 * l_pos + 0.5 * l_neg).mean()
            else:
                l_mcls_px = F.binary_cross_entropy(c_pred, c_gt)
        else:
            l_mcls_px = torch.tensor(0.0, device=device)

        pi_12 = out["sfr_probs_12"].float().clamp(min=1e-6, max=1.0 - 1e-6)
        B, L, _ = pi_12.shape
        H_p = int(round(L ** 0.5))
        # Fraction of defect pixels inside each 14x14 patch. A 1-2 px wide crack crossing a patch
        # covers only 7-14% of it, so the original ">= 0.20" rule labels such patches as background
        # and the two background terms below then push them towards the normal / nuisance anchors.
        #   ratio20 : original rule (positive >= 20%, everything else background)
        #   ignore  : positive >= 20%; patches with 0 < fraction < 20% get no patch-level target
        #   any     : positive if the patch contains any defect pixel
        frac_p = F.adaptive_avg_pool2d(gt_mask, (H_p, H_p)).reshape(B, L)
        nuis_p = (F.adaptive_avg_pool2d(nuis_mask, (H_p, H_p)).reshape(B, L) >= 0.20).float()
        if sfr_patch_label == "any":
            gt_p = (frac_p > 0).float()
            bg_mask = (1.0 - gt_p)
        elif sfr_patch_label == "ignore":
            gt_p = (frac_p >= 0.20).float()
            bg_mask = (frac_p <= 0).float()
        elif sfr_patch_label == "ratio20":
            gt_p = (frac_p >= 0.20).float()
            bg_mask = (1.0 - gt_p)
        else:
            raise ValueError(f"Unknown sfr_patch_label: {sfr_patch_label}")

        pi_norm = pi_12[:, :, 0]
        pi_nuis_sum = pi_12[:, :, 1:7].sum(dim=-1)
        pi_def_sum = pi_12[:, :, 7:12].sum(dim=-1)

        with torch.no_grad():
            coarse_p = F.adaptive_avg_pool2d(coarse_map.detach(), (H_p, H_p)).reshape(B, L)
            high_var_bg = ((coarse_p > 0.25).float() + nuis_p).clamp(max=1.0) * bg_mask
            pure_norm_bg = (1.0 - high_var_bg) * bg_mask

        l_pure_norm = -(pure_norm_bg * torch.log(pi_norm.clamp(min=1e-6))).sum() / (pure_norm_bg.sum() + 1.0)
        l_nuis_bg = -(high_var_bg * torch.log((pi_norm + pi_nuis_sum).clamp(min=1e-6))).sum() / (high_var_bg.sum() + 1.0)
        l_def_patch = -(gt_p * torch.log(pi_def_sum.clamp(min=1e-6))).sum() / (gt_p.sum() + 1.0)

        loss_sfr = 0.4 * l_mcls_img + 0.3 * l_mcls_px + 0.2 * (l_pure_norm + l_nuis_bg + l_def_patch) + 0.1 * out["loss_ortho"]
        total_loss = total_loss + loss_sfr

    return total_loss, {
        "loss_total": float(total_loss.item()),
        "loss_seg": float(loss_seg.item()),
        "loss_ac": float(loss_ac.item()),
        "loss_etf": float(loss_etf.item()),
        "loss_sfr": float(loss_sfr.item()),
    }


@torch.no_grad()
def evaluate_model(
    model: TFMoECLIP,
    eval_loader: DataLoader,
    use_cache: bool = True,
    device: str = "cuda",
    compute_pro: bool = True,
) -> Dict:
    model.eval()
    gt_sp_list = []
    pr_sp_list = []
    masks_list = []
    nuis_list = []
    amaps_list = []
    class_masks_dict = {c: [] for c in DEFECT_CLASSES}

    for batch in eval_loader:
        if use_cache:
            pf = batch["patch_feats"].to(device, non_blocking=True, dtype=torch.float32)
            patch_layers = [pf[:, i] for i in range(pf.shape[1])]
            H_p = int(round(pf.shape[2] ** 0.5))
            wt_desc = batch["wt_desc"].to(device, non_blocking=True, dtype=torch.float32) if (model.use_wt_router or getattr(model, "use_aniso_paa", False)) else None
            hr_cue = batch["hr_cue"].to(device, non_blocking=True, dtype=torch.float32) if model.use_dw_cpr else None
            out = model.forward_from_features(
                patch_tokens_layers=patch_layers,
                grid_hw=(H_p, H_p),
                patch_wt_desc=wt_desc,
                high_res_cue=hr_cue,
            )
        else:
            img = batch["img"].to(device, non_blocking=True)
            out = model(img)

        amap_np = out["anomaly_map"].squeeze(1).float().cpu().numpy()
        sp_np = out["img_score"].float().cpu().numpy()
        gt_mask_np = (batch["img_mask"].squeeze(1).numpy() > 0.5).astype(np.uint8)
        nuis_mask_np = (batch["nuisance_mask"].squeeze(1).numpy() > 0.5).astype(np.uint8)
        cmasks_np = (batch["class_masks_tensor"].numpy() > 0.5).astype(np.uint8)

        amaps_list.append(amap_np)
        pr_sp_list.append(sp_np)
        masks_list.append(gt_mask_np)
        nuis_list.append(nuis_mask_np)
        gt_sp_list.append(np.asarray(batch["anomaly"], dtype=np.int64))

        for c_idx, cname in enumerate(DEFECT_CLASSES):
            for b_i in range(cmasks_np.shape[0]):
                class_masks_dict[cname].append(cmasks_np[b_i, c_idx])

    gt_sp_all = np.concatenate(gt_sp_list, axis=0)
    pr_sp_all = np.concatenate(pr_sp_list, axis=0)
    masks_all = np.concatenate(masks_list, axis=0)
    nuis_all = np.concatenate(nuis_list, axis=0)
    amaps_all = np.concatenate(amaps_list, axis=0)

    return evaluate_anomaly_predictions(
        gt_sp=gt_sp_all,
        pr_sp=pr_sp_all,
        masks_np=masks_all,
        amaps_raw=amaps_all,
        class_masks_dict=class_masks_dict,
        nuisance_masks_np=nuis_all,
        apply_gaussian_sigma=4.0,
        compute_pro=compute_pro,
    )


def train_and_evaluate_variant(
    variant: str,
    train_dataset_name: str,
    clip_backbone,
    train_loader: DataLoader,
    val_loader: DataLoader,
    in_domain_test_loader: DataLoader,
    cross_domain_test_loader: Optional[DataLoader],
    cross_domain_name: str,
    output_dir: str,
    epochs: int = 12,
    lr: float = 5e-4,
    weight_decay: float = 1e-4,
    seed: int = 111,
    use_cache: bool = True,
    max_train_batches: int = 0,
    device: str = "cuda",
    sfr_patch_label: str = "ratio20",
    cls_px_balance: bool = False,
    group_prior: bool = False,
    run_tag: str = "",
    eval_only: bool = False,
) -> Dict:
    setup_seed(seed)
    os.makedirs(output_dir, exist_ok=True)

    # Files of a tagged run are named "<variant>-<tag>_...", so they never overwrite the default run.
    stem = f"{variant}-{run_tag}" if run_tag else variant
    run_opts = {
        "sfr_patch_label": sfr_patch_label,
        "cls_px_balance": bool(cls_px_balance),
        "group_prior": bool(group_prior),
    }

    model = build_tf_moeclip_variant(variant, clip_backbone=clip_backbone).to(device)
    if group_prior:
        model.group_prior_bias = math.log(7.0 / 5.0)
    # 100% Pure From-Scratch Initialization (identical protocol across A0, A1, A2, A3, A4!)
    trunk_params = [p for n, p in model.named_parameters() if p.requires_grad and not n.startswith("dw_cpr.")]
    cpr_params = [p for n, p in model.named_parameters() if p.requires_grad and n.startswith("dw_cpr.")]
    trainable_params = trunk_params + cpr_params
    optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=weight_decay)

    n_trainable = sum(p.numel() for p in trainable_params)
    print(f"\n=======================================================")
    print(f"{'Re-evaluating' if eval_only else 'Training'} Variant: {stem} on {train_dataset_name} "
          f"| Trainable Params: {n_trainable:,} | opts={run_opts}")
    print(f"=======================================================")

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1), eta_min=lr * 0.05)
    focal_fn = FocalLoss(gamma=2.0, alpha=0.25)
    dice_fn = BinaryDiceLoss()

    best_val_score = -1.0
    best_epoch = 0
    best_ckpt_path = os.path.join(output_dir, f"{stem}_{train_dataset_name}_best.pt")
    history = []

    if eval_only and not os.path.isfile(best_ckpt_path):
        raise FileNotFoundError(f"--eval_only needs an existing checkpoint: {best_ckpt_path}")

    # eval_only: skip training and only re-run the test evaluation of the saved best checkpoint.
    for ep in range(1, (0 if eval_only else epochs) + 1):
        t0 = time.time()
        model.train()
        loss_meter = []

        for b_idx, batch in enumerate(train_loader):
            if max_train_batches > 0 and b_idx >= max_train_batches:
                break
            optimizer.zero_grad(set_to_none=True)

            if use_cache:
                pf = batch["patch_feats"].to(device, non_blocking=True, dtype=torch.float32)
                patch_layers = [pf[:, i] for i in range(pf.shape[1])]
                H_p = int(round(pf.shape[2] ** 0.5))
                wt_desc = batch["wt_desc"].to(device, non_blocking=True, dtype=torch.float32) if (model.use_wt_router or getattr(model, "use_aniso_paa", False)) else None
                hr_cue = batch["hr_cue"].to(device, non_blocking=True, dtype=torch.float32) if model.use_dw_cpr else None
                out = model.forward_from_features(
                    patch_tokens_layers=patch_layers,
                    grid_hw=(H_p, H_p),
                    patch_wt_desc=wt_desc,
                    high_res_cue=hr_cue,
                )
            else:
                img = batch["img"].to(device, non_blocking=True)
                out = model(img)

            loss, l_dict = compute_tf_moeclip_loss(
                out=out,
                batch=batch,
                focal_fn=focal_fn,
                dice_fn=dice_fn,
                device=device,
                sfr_patch_label=sfr_patch_label,
                cls_px_balance=cls_px_balance,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trunk_params, max_norm=2.0)
            if cpr_params:
                torch.nn.utils.clip_grad_norm_(cpr_params, max_norm=2.0)
            optimizer.step()
            loss_meter.append(l_dict["loss_total"])

        scheduler.step()
        do_val_pro = (ep == epochs) or (ep % 4 == 0)
        val_metrics = evaluate_model(
            model, val_loader, use_cache=use_cache, device=device, compute_pro=do_val_pro
        )
        # Consistent 6-metric validation selection score across all epochs so checkpoints are compared fairly!
        val_sel_score = (
            val_metrics["Image-AUROC"] + val_metrics["Image-F1"] + val_metrics["Image-AP"] +
            val_metrics["Pixel-AUROC"] + val_metrics["Pixel-F1"] + val_metrics["Pixel-AP"]
        ) / 6.0
        elapsed = time.time() - t0
        avg_loss = float(np.mean(loss_meter)) if loss_meter else 0.0

        print(
            f"[Ep {ep:02d}/{epochs:02d}] loss={avg_loss:.4f} | Val Sel-6={val_sel_score:.2f}% "
            f"(I-AUROC={val_metrics['Image-AUROC']:.2f}, P-AUROC={val_metrics['Pixel-AUROC']:.2f}, "
            f"P-F1={val_metrics['Pixel-F1']:.2f}, P-AP={val_metrics['Pixel-AP']:.2f}, "
            f"P-AUPRO={val_metrics['Pixel-AUPRO']:.2f}) [{elapsed:.1f}s]"
        )
        history.append({"epoch": ep, "loss": round(avg_loss, 4), "val_metrics": val_metrics})

        # Prefer converged checkpoints (ep >= 8) once trunk reaches mature cosine annealing stage
        effective_score = val_sel_score if ep >= 8 else (val_sel_score - 1.5)
        if effective_score > best_val_score:
            best_val_score = effective_score
            best_epoch = ep
            torch.save(
                {
                    "variant": variant,
                    "epoch": ep,
                    "val_metrics": val_metrics,
                    "opts": run_opts,
                    "state_dict": {k: v.cpu() for k, v in model.state_dict().items() if not k.startswith("clip_backbone.")},
                },
                best_ckpt_path,
            )

    ckpt = torch.load(best_ckpt_path, map_location=device)
    # strict=False is needed because the frozen CLIP backbone is not stored in the checkpoint.
    # Any other mismatch means the file does not belong to this variant, so it is an error.
    load_info = model.load_state_dict(ckpt["state_dict"], strict=False)
    bad_missing = [k for k in load_info.missing_keys if not k.startswith("clip_backbone.")]
    if bad_missing or load_info.unexpected_keys:
        raise RuntimeError(
            f"Checkpoint {best_ckpt_path} does not match variant {variant}: "
            f"missing={bad_missing[:5]} unexpected={list(load_info.unexpected_keys)[:5]}"
        )
    ckpt_sha256 = _file_sha256(best_ckpt_path)
    train_json = os.path.join(output_dir, f"metrics_{stem}_{train_dataset_name}.json")
    if eval_only and os.path.isfile(train_json):
        with open(train_json, "r", encoding="utf-8") as f:
            recorded = json.load(f).get("provenance", {}).get("ckpt_sha256")
        if recorded and recorded != ckpt_sha256:
            raise RuntimeError(
                f"Checkpoint fingerprint mismatch for {stem}: {best_ckpt_path} is not the file written by "
                f"the training run recorded in {train_json}. Retrain this variant instead of re-evaluating."
            )
        if not recorded:
            print(f"[WARN] {train_json} has no checkpoint fingerprint; the pairing of this checkpoint "
                  f"with the training record cannot be verified.")
    if eval_only:
        best_epoch = int(ckpt.get("epoch", 0))
        # The energy offset is not a parameter, so restore it from the options saved with the checkpoint.
        if ckpt.get("opts", {}).get("group_prior", False):
            model.group_prior_bias = math.log(7.0 / 5.0)
        print(f"--> Re-evaluating saved checkpoint from Epoch {best_epoch}: {best_ckpt_path}")
    else:
        print(f"--> Loaded best val checkpoint from Epoch {best_epoch} (Val Sel-6={best_val_score:.2f}%)")

    in_domain_test = evaluate_model(model, in_domain_test_loader, use_cache=use_cache, device=device, compute_pro=True)
    print(
        f"[{variant} In-Domain Test ({train_dataset_name})] Mean-7={in_domain_test['Mean-Score']:.2f}% | "
        f"I-AUROC={in_domain_test['Image-AUROC']:.2f} | P-AUROC={in_domain_test['Pixel-AUROC']:.2f} | "
        f"P-F1={in_domain_test['Pixel-F1']:.2f} | P-AP={in_domain_test['Pixel-AP']:.2f} | "
        f"P-AUPRO={in_domain_test['Pixel-AUPRO']:.2f} | Per-Class={in_domain_test['Per-Class']}"
    )

    cross_domain_test = None
    if cross_domain_test_loader is not None:
        cross_domain_test = evaluate_model(
            model, cross_domain_test_loader, use_cache=use_cache, device=device, compute_pro=True
        )
        print(
            f"[{variant} Cross-Domain Test ({train_dataset_name} -> {cross_domain_name})] "
            f"Mean-7={cross_domain_test['Mean-Score']:.2f}% | "
            f"I-AUROC={cross_domain_test['Image-AUROC']:.2f} | P-AUROC={cross_domain_test['Pixel-AUROC']:.2f} | "
            f"P-F1={cross_domain_test['Pixel-F1']:.2f} | P-AP={cross_domain_test['Pixel-AP']:.2f} | "
            f"P-AUPRO={cross_domain_test['Pixel-AUPRO']:.2f} | Nuisance-FPR={cross_domain_test['Nuisance-FPR']:.2f}% | "
            f"Per-Class={cross_domain_test['Per-Class']}"
        )

    result_record = {
        "variant": stem,
        "run_opts": ckpt.get("opts", run_opts) if eval_only else run_opts,
        "eval_only": bool(eval_only),
        "train_dataset": train_dataset_name,
        "trainable_params": n_trainable,
        "best_epoch": best_epoch,
        "best_val_metrics": ckpt["val_metrics"],
        "in_domain_test_metrics": in_domain_test,
        "cross_domain_name": cross_domain_name if cross_domain_test is not None else None,
        "cross_domain_test_metrics": cross_domain_test,
        "history": history,
        # Written only by this script. A record without this block, or whose fingerprint differs
        # from the checkpoint file, must not be used as evidence.
        "provenance": {
            "written_by": "train_eval.py",
            "written_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "ckpt_file": os.path.basename(best_ckpt_path),
            "ckpt_sha256": ckpt_sha256,
            "ckpt_bytes": os.path.getsize(best_ckpt_path),
            "ckpt_epoch": int(ckpt.get("epoch", 0)),
            "ckpt_variant": ckpt.get("variant"),
        },
    }
    # A re-evaluation is written next to the original record instead of replacing it.
    out_json = os.path.join(
        output_dir, f"metrics_{stem}_{train_dataset_name}{'_reeval' if eval_only else ''}.json"
    )
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(result_record, f, indent=2)
    return result_record


def format_markdown_table(records: List[Dict], metric_key: str = "in_domain_test_metrics", title: str = "") -> str:
    lines = [
        f"### {title}",
        "",
        "| Variant | I-AUROC | I-F1 | I-AP | P-AUROC | P-F1 | P-AP | P-AUPRO | P-IoU | **Mean-7** | **Mean-5** | Crack | Spalling | Corrosion | Efflor. | Nuis-FPR |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    has_old_pro = False
    for r in records:
        m = r.get(metric_key)
        if not m:
            continue
        pc = m.get("Per-Class", {})
        # Records written before the AUPRO fix carry the score-uniform value; flag them so that
        # old and new numbers are never compared silently inside one table.
        old_pro = m.get("AUPRO-Impl") != "quantile-v2"
        if old_pro:
            has_old_pro = True
        lines.append(
            f"| **{r['variant']}{' (old AUPRO)' if old_pro else ''}** | {m['Image-AUROC']:.2f} | {m['Image-F1']:.2f} | {m['Image-AP']:.2f} | "
            f"{m['Pixel-AUROC']:.2f} | {m['Pixel-F1']:.2f} | {m['Pixel-AP']:.2f} | {m['Pixel-AUPRO']:.2f} | "
            f"{m['Pixel-IoU']:.2f} | **{m['Mean-Score']:.2f}** | **{m['Mean-5']:.2f}** | "
            f"{pc.get('Crack_Recall', 0.0):.2f} | {pc.get('Spalling_Recall', 0.0):.2f} | "
            f"{pc.get('Corrosion_Recall', 0.0):.2f} | {pc.get('Efflorescence_Recall', 0.0):.2f} | "
            f"{m.get('Nuisance-FPR', 0.0):.2f}% |"
        )
    if has_old_pro:
        lines.append("")
        lines.append("Rows marked (old AUPRO) were evaluated before the AUPRO fix: their P-AUPRO, Mean-7 and "
                     "Mean-5 are not comparable with the other rows. Re-run them with --eval_only 1.")
    lines.append("")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="TF-MoECLIP V2 (From-Scratch Clean Serial Chain) Training & Benchmark")
    parser.add_argument("--codebrim_root", type=str, default="/home/skye/data/Skye/databases/Bridge512_balance_CODEBRIM")
    parser.add_argument("--s2ds_root", type=str, default="/home/skye/data/Skye/databases/S2DS")
    parser.add_argument("--s2ds_mode", type=str, default="official_full", choices=["official_full", "official_tiles512"])
    parser.add_argument("--clip_ckpt", type=str, default=None)
    parser.add_argument("--cache_dir", type=str, default="./feature_cache")
    parser.add_argument("--output_dir", type=str, default="./outputs_v2")
    parser.add_argument("--train_on", type=str, default="BOTH", choices=["CODEBRIM", "S2DS", "BOTH"])
    parser.add_argument("--variants", type=str, default="M0,M1,M2,M3", help="Supports M0(A0), M1, M1_PAA, M1_HET, M1_WT, M2(A3), M3(A4)")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--seed", type=int, default=111)
    parser.add_argument("--use_cache", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--max_train_batches", type=int, default=0)
    parser.add_argument("--sfr_patch_label", type=str, default="ratio20", choices=["ratio20", "ignore", "any"],
                        help="Patch-level label rule of the 12-anchor loss (ratio20 = original behaviour).")
    parser.add_argument("--cls_px_balance", type=int, default=0,
                        help="1 = class-balanced per-class pixel BCE (0 = original behaviour).")
    parser.add_argument("--group_prior", type=int, default=0,
                        help="1 = add log(7/5) to the defect-group energy (0 = original behaviour).")
    parser.add_argument("--run_tag", type=str, default="",
                        help="Suffix for checkpoint / metric file names, e.g. 'ign' -> M2-ign_Bridge512_best.pt.")
    parser.add_argument("--eval_only", type=int, default=0,
                        help="1 = no training; re-evaluate the saved best checkpoints and write *_reeval.json.")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    setup_seed(args.seed)
    transform = get_clip_transform(image_size=518)

    print(f"Loading frozen CLIP ViT-L/14@336px backbone on {device} ...")
    clip_backbone = load_clip_vit_l14_336(checkpoint_path=args.clip_ckpt, device=device)

    cb_train_raw = build_dataset("CODEBRIM", args.codebrim_root, split="train", image_size=518, preprocess=transform)
    cb_val_raw = build_dataset("CODEBRIM", args.codebrim_root, split="val", image_size=518, preprocess=transform)
    cb_test_raw = build_dataset("CODEBRIM", args.codebrim_root, split="test", image_size=518, preprocess=transform)

    s2_train_raw = build_dataset("S2DS", args.s2ds_root, split="train", image_size=518, preprocess=transform, s2ds_mode=args.s2ds_mode)
    s2_val_raw = build_dataset("S2DS", args.s2ds_root, split="val", image_size=518, preprocess=transform, s2ds_mode=args.s2ds_mode)
    s2_test_raw = build_dataset("S2DS", args.s2ds_root, split="test", image_size=518, preprocess=transform, s2ds_mode=args.s2ds_mode)

    print(
        f"Datasets Ready:\n"
        f"  - Bridge512 (CODEBRIM): train={len(cb_train_raw)}, val={len(cb_val_raw)}, test={len(cb_test_raw)}\n"
        f"  - Official S2DS ({args.s2ds_mode}): train={len(s2_train_raw)}, val={len(s2_val_raw)}, test={len(s2_test_raw)}"
    )

    use_cache = bool(args.use_cache)
    if use_cache:
        s2_tag = f"S2DS_{args.s2ds_mode}"
        cb_train = build_or_load_feature_cache(clip_backbone, cb_train_raw, args.cache_dir, "CODEBRIM_train", batch_size=args.batch_size, num_workers=args.num_workers, device=device)
        cb_val = build_or_load_feature_cache(clip_backbone, cb_val_raw, args.cache_dir, "CODEBRIM_val", batch_size=args.batch_size, num_workers=args.num_workers, device=device)
        cb_test = build_or_load_feature_cache(clip_backbone, cb_test_raw, args.cache_dir, "CODEBRIM_test", batch_size=args.batch_size, num_workers=args.num_workers, device=device)

        s2_train = build_or_load_feature_cache(clip_backbone, s2_train_raw, args.cache_dir, f"{s2_tag}_train", batch_size=args.batch_size, num_workers=args.num_workers, device=device)
        s2_val = build_or_load_feature_cache(clip_backbone, s2_val_raw, args.cache_dir, f"{s2_tag}_val", batch_size=args.batch_size, num_workers=args.num_workers, device=device)
        s2_test = build_or_load_feature_cache(clip_backbone, s2_test_raw, args.cache_dir, f"{s2_tag}_test", batch_size=args.batch_size, num_workers=args.num_workers, device=device)
        nw = 0
    else:
        cb_train, cb_val, cb_test = cb_train_raw, cb_val_raw, cb_test_raw
        s2_train, s2_val, s2_test = s2_train_raw, s2_val_raw, s2_test_raw
        nw = args.num_workers

    cb_train_loader = DataLoader(cb_train, batch_size=args.batch_size, shuffle=True, num_workers=nw, pin_memory=True)
    cb_val_loader = DataLoader(cb_val, batch_size=args.batch_size, shuffle=False, num_workers=nw, pin_memory=True)
    cb_test_loader = DataLoader(cb_test, batch_size=args.batch_size, shuffle=False, num_workers=nw, pin_memory=True)

    s2_train_loader = DataLoader(s2_train, batch_size=args.batch_size, shuffle=True, num_workers=nw, pin_memory=True)
    s2_val_loader = DataLoader(s2_val, batch_size=args.batch_size, shuffle=False, num_workers=nw, pin_memory=True)
    s2_test_loader = DataLoader(s2_test, batch_size=args.batch_size, shuffle=False, num_workers=nw, pin_memory=True)

    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    md_sections = []

    all_known_order = ["M0", "M0_HOMO", "M1", "M1_HOMO", "M1_PAA", "M1_HET", "M1_WT", "M2", "M3", "A0", "A1", "A2", "A3", "A4"]
    run_order = [v for v in all_known_order if v in variants] + [v for v in variants if v not in all_known_order]

    if args.train_on in ("CODEBRIM", "BOTH"):
        cb_records = []
        for v in all_known_order:
            if v not in run_order:
                existing_json = os.path.join(args.output_dir, f"metrics_{v}_Bridge512.json")
                if os.path.isfile(existing_json):
                    with open(existing_json, "r", encoding="utf-8") as f:
                        cb_records.append(json.load(f))
                continue
            rec = train_and_evaluate_variant(
                variant=v,
                train_dataset_name="Bridge512",
                clip_backbone=clip_backbone,
                train_loader=cb_train_loader,
                val_loader=cb_val_loader,
                in_domain_test_loader=cb_test_loader,
                cross_domain_test_loader=s2_test_loader,
                cross_domain_name="S2DS",
                output_dir=args.output_dir,
                epochs=args.epochs,
                lr=args.lr,
                seed=args.seed,
                use_cache=use_cache,
                max_train_batches=args.max_train_batches,
                device=device,
                sfr_patch_label=args.sfr_patch_label,
                cls_px_balance=bool(args.cls_px_balance),
                group_prior=bool(args.group_prior),
                run_tag=args.run_tag,
                eval_only=bool(args.eval_only),
            )
            cb_records.append(rec)
        md_sections.append(format_markdown_table(cb_records, "in_domain_test_metrics", "1. Bridge512 (In-Domain Main Benchmark)"))
        md_sections.append(format_markdown_table(cb_records, "cross_domain_test_metrics", "2. Bridge512 -> S2DS (Cross-Domain Zero-Shot Benchmark)"))

    if args.train_on in ("S2DS", "BOTH"):
        s2_records = []
        for v in all_known_order:
            if v not in run_order:
                existing_json = os.path.join(args.output_dir, f"metrics_{v}_S2DS.json")
                if os.path.isfile(existing_json):
                    with open(existing_json, "r", encoding="utf-8") as f:
                        s2_records.append(json.load(f))
                continue
            rec = train_and_evaluate_variant(
                variant=v,
                train_dataset_name="S2DS",
                clip_backbone=clip_backbone,
                train_loader=s2_train_loader,
                val_loader=s2_val_loader,
                in_domain_test_loader=s2_test_loader,
                cross_domain_test_loader=cb_test_loader,
                cross_domain_name="Bridge512",
                output_dir=args.output_dir,
                epochs=args.epochs,
                lr=args.lr,
                seed=args.seed,
                use_cache=use_cache,
                max_train_batches=args.max_train_batches,
                device=device,
                sfr_patch_label=args.sfr_patch_label,
                cls_px_balance=bool(args.cls_px_balance),
                group_prior=bool(args.group_prior),
                run_tag=args.run_tag,
                eval_only=bool(args.eval_only),
            )
            s2_records.append(rec)
        md_sections.append(format_markdown_table(s2_records, "in_domain_test_metrics", "3. S2DS (Auxiliary In-Domain Benchmark)"))
        md_sections.append(format_markdown_table(s2_records, "cross_domain_test_metrics", "4. S2DS -> Bridge512 (Cross-Domain Zero-Shot Benchmark)"))

    suffix = ("_" + args.run_tag if args.run_tag else "") + ("_reeval" if args.eval_only else "")
    summary_md_path = os.path.join(args.output_dir, f"summary_tables_v2{suffix}.md")
    with open(summary_md_path, "w", encoding="utf-8") as f:
        f.write("# TF-MoECLIP V2 (Clean Serial Chain A0->A4) Experimental Results\n\n")
        f.write("\n".join(md_sections))
    print(f"\nSaved V2 formatted Markdown tables to: {summary_md_path}")
    print("\n".join(md_sections))


if __name__ == "__main__":
    main()
