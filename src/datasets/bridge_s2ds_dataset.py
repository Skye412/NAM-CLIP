import csv
import json
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image
import torch
import torch.utils.data as data
import torchvision.transforms as transforms

DEFECT_CLASSES = ["Crack", "Spalling", "Corrosion", "Efflorescence"]

# Palette in Bridge512_balance_CODEBRIM
CODEBRIM_RGB_MAP = {
    "Crack": (255, 0, 0),
    "Spalling": (0, 255, 0),
    "Corrosion": (0, 0, 255),
    "Efflorescence": (255, 255, 0),
}

# Verified 7-class RGB palette in Official S2DS (Benz & Rodehorst, GCPR 2022)
OFFICIAL_S2DS_RGB_MAP = {
    "Background": (0, 0, 0),
    "Crack": (255, 255, 255),         # 0.80M px across 743 images
    "Spalling": (255, 0, 0),          # 46.61M px
    "Corrosion": (255, 255, 0),       # 10.23M px
    "Efflorescence": (0, 255, 255),   # 6.68M px
    "Vegetation": (0, 255, 0),        # 21.29M px (Benign Nuisance 1)
    "ControlPoint": (0, 0, 255),      # 1.99M px  (Benign Nuisance 2: Geodetic fiducial marker)
}


class BridgeCodebrimDataset(data.Dataset):
    """
    Dataset loader for self-built Bridge512_balance_CODEBRIM.
    Supports:
      - Exact binary union anomaly mask
      - Multi-label 4-class masks (from multiclass_annotations/masks/ if available, else RGB palette)
      - Multi-label image vector y_cls in {0, 1}^4
    """
    def __init__(
        self,
        root: str,
        split: str = "train",
        image_size: int = 518,
        preprocess=None,
        return_class_masks: bool = True,
    ):
        super().__init__()
        self.root = root
        self.split = split
        self.image_size = image_size
        self.preprocess = preprocess
        self.return_class_masks = return_class_masks

        manifest_path = os.path.join(root, f"{split}_manifest.csv")
        if not os.path.isfile(manifest_path):
            raise FileNotFoundError(f"CODEBRIM manifest not found: {manifest_path}")

        self.samples = []
        with open(manifest_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                is_anom = 1 if row["category"] == "annotated" else 0
                img_rel = os.path.join(split, row["image_path"])
                mask_rel = os.path.join(split, row["mask_path"]) if (is_anom and row["mask_path"]) else ""
                sample_id = row["sample"]
                self.samples.append({
                    "sample_id": sample_id,
                    "img_path": os.path.join(root, img_rel),
                    "mask_path": os.path.join(root, mask_rel) if mask_rel else "",
                    "anomaly": is_anom,
                    "labels": row.get("labels", ""),
                    "multiclass_dir": os.path.join(root, split, "multiclass_annotations", "masks"),
                })

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict:
        item = self.samples[index]
        img = Image.open(item["img_path"]).convert("RGB")
        w, h = img.size
        anomaly = int(item["anomaly"])

        if anomaly == 0 or not item["mask_path"]:
            bin_mask_np = np.zeros((h, w), dtype=np.uint8)
            rgb_mask_np = None
        else:
            rgb_mask_np = np.asarray(Image.open(item["mask_path"]).convert("RGB"), dtype=np.uint8)
            bin_mask_np = ((rgb_mask_np > 0).any(axis=-1)).astype(np.uint8) * 255

        if self.preprocess is not None:
            img_tensor = self.preprocess(img)
        else:
            img_tensor = transforms.ToTensor()(img)

        mask_pil = Image.fromarray(bin_mask_np, mode="L").resize(
            (self.image_size, self.image_size), resample=Image.NEAREST
        )
        mask_tensor = (transforms.ToTensor()(mask_pil) > 0.5).float()  # [1, H, W]
        nuisance_tensor = torch.zeros_like(mask_tensor)

        class_masks_stack = []
        multi_label_vec = torch.zeros(len(DEFECT_CLASSES), dtype=torch.float32)

        for c_idx, cname in enumerate(DEFECT_CLASSES):
            if anomaly == 0:
                cm_pil = Image.fromarray(np.zeros((self.image_size, self.image_size), dtype=np.uint8), mode="L")
                cm_tensor = torch.zeros((1, self.image_size, self.image_size), dtype=torch.float32)
            else:
                mc_file = os.path.join(item["multiclass_dir"], f"{item['sample_id']}__{cname}.png")
                if os.path.isfile(mc_file):
                    cm_raw = (np.asarray(Image.open(mc_file).convert("L"), dtype=np.uint8) > 127).astype(np.uint8) * 255
                else:
                    rgb = CODEBRIM_RGB_MAP[cname]
                    cm_raw = (
                        (rgb_mask_np[:, :, 0] == rgb[0]) &
                        (rgb_mask_np[:, :, 1] == rgb[1]) &
                        (rgb_mask_np[:, :, 2] == rgb[2])
                    ).astype(np.uint8) * 255
                if cm_raw.any():
                    multi_label_vec[c_idx] = 1.0
                cm_pil = Image.fromarray(cm_raw, mode="L").resize(
                    (self.image_size, self.image_size), resample=Image.NEAREST
                )
                cm_tensor = (transforms.ToTensor()(cm_pil) > 0.5).float()
            class_masks_stack.append(cm_tensor)

        class_masks_tensor = torch.cat(class_masks_stack, dim=0)  # [4, H, W]

        out = {
            "img": img_tensor,
            "img_mask": mask_tensor,
            "nuisance_mask": nuisance_tensor,
            "class_masks_tensor": class_masks_tensor,
            "multi_label": multi_label_vec,
            "anomaly": anomaly,
            "img_path": item["img_path"],
            "sample_id": item["sample_id"],
        }
        if self.return_class_masks:
            out["class_masks"] = {
                cname: class_masks_tensor[i : i + 1] for i, cname in enumerate(DEFECT_CLASSES)
            }
        return out


class OfficialS2DSDataset(data.Dataset):
    """
    Dataset loader for the Official S2DS Dataset (Benz & Rodehorst, GCPR 2022)
    stored at /home/skye/data/Skye/databases/S2DS.
    - Official splits: train (563), val (87), test (93), each 1024x1024.
    - 7 classes in *_lab.png:
        * Structural Defects (anomaly=1): Crack, Spalling, Corrosion, Efflorescence
        * Benign Nuisances / Background (anomaly=0): Vegetation, ControlPoint, Background
    - Modes:
        * mode="official_tiles512": Non-overlapping 2x2 512x512 tiles of the official split
          (train: 2252 [867 normal / 1385 anomaly], val: 348 [130 normal / 218 anomaly],
           test: 372 [117 normal / 255 anomaly]). Matches 512x512 resolution of CODEBRIM
          while strictly preserving the official S2DS train/val/test split!
        * mode="official_full": Full 1024x1024 official images (train: 563 [97 normal / 466 anomaly],
          val: 87 [14 normal / 73 anomaly], test: 93 [13 normal / 80 anomaly]).
    """
    def __init__(
        self,
        root: str,
        split: str = "train",
        image_size: int = 518,
        preprocess=None,
        return_class_masks: bool = True,
        mode: str = "official_full",
    ):
        super().__init__()
        self.root = root
        self.split = split
        self.image_size = image_size
        self.preprocess = preprocess
        self.return_class_masks = return_class_masks
        self.mode = mode

        # Automatically resolve nested folder if official zip unpacked into a subdirectory (e.g. s2ds/ or S2DS/)
        eff_root = root
        for sub in ["s2ds", "S2DS", "structural_defects"]:
            if not os.path.isdir(os.path.join(eff_root, "train")) and os.path.isdir(os.path.join(eff_root, sub)):
                eff_root = os.path.join(eff_root, sub)
                break
        self.root = eff_root

        # Resolve split directory name (official zip may name validation 'val', 'validation', or 'valid')
        split_aliases = {
            "train": ["train", "training"],
            "val": ["val", "validation", "valid"],
            "test": ["test", "testing"],
        }
        resolved_split_dir = None
        for alias in split_aliases.get(split, [split]):
            cand = os.path.join(eff_root, alias)
            if os.path.isdir(cand):
                resolved_split_dir = cand
                break
        if resolved_split_dir is None:
            raise FileNotFoundError(f"Official S2DS split '{split}' not found under: {eff_root}")

        cand_img_dir = os.path.join(resolved_split_dir, "images")
        cand_msk_dir = os.path.join(resolved_split_dir, "masks")
        if os.path.isdir(cand_img_dir) and os.path.isdir(cand_msk_dir):
            img_dir = cand_img_dir
            msk_dir = cand_msk_dir
        else:
            img_dir = resolved_split_dir
            msk_dir = resolved_split_dir

        img_names = sorted([
            f for f in os.listdir(img_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg")) and not f.lower().endswith("_lab.png")
        ])
        self.entries = []

        # Cache split metadata json so scanning 1024x1024 masks only happens once
        cache_meta_file = os.path.join(root, f"_cached_{split}_{mode}_entries.json")
        if os.path.isfile(cache_meta_file):
            with open(cache_meta_file, "r", encoding="utf-8") as f:
                cached = json.load(f)
            # Re-bind current root path in case dataset was moved (e.g. Server -> Colab)
            for e in cached:
                e["img_path"] = os.path.join(img_dir, os.path.basename(e["img_path"]))
                e["mask_path"] = os.path.join(msk_dir, os.path.basename(e["mask_path"]))
            self.entries = cached
        else:
            for fn in img_names:
                stem = os.path.splitext(fn)[0]
                img_path = os.path.join(img_dir, fn)
                mask_path = os.path.join(msk_dir, f"{stem}_lab.png")
                if not os.path.isfile(mask_path):
                    raise FileNotFoundError(f"Missing official S2DS mask: {mask_path}")

                arr = np.asarray(Image.open(mask_path).convert("RGB"), dtype=np.uint8)
                H, W = arr.shape[:2]

                if mode == "official_full":
                    boxes = [(0, 0, W, H)]
                elif mode == "official_tiles512":
                    boxes = [
                        (0, 0, 512, 512),
                        (512, 0, 1024, 512),
                        (0, 512, 512, 1024),
                        (512, 512, 1024, 1024),
                    ]
                else:
                    raise ValueError(f"Unknown S2DS mode: {mode}")

                packed = (arr[:, :, 0].astype(np.int32) << 16) | (arr[:, :, 1].astype(np.int32) << 8) | arr[:, :, 2].astype(np.int32)
                for (x0, y0, x1, y1) in boxes:
                    sub_p = packed[y0:y1, x0:x1]
                    u_vals = set(np.unique(sub_p).tolist())
                    has_def = False
                    present_classes = []
                    for cname in DEFECT_CLASSES:
                        rgb = OFFICIAL_S2DS_RGB_MAP[cname]
                        code = (rgb[0] << 16) | (rgb[1] << 8) | rgb[2]
                        if code in u_vals:
                            has_def = True
                            present_classes.append(cname)
                    sample_id = f"s2ds_{stem}" if mode == "official_full" else f"s2ds_{stem}_x{x0:04d}_y{y0:04d}"
                    self.entries.append({
                        "sample_id": sample_id,
                        "img_path": img_path,
                        "mask_path": mask_path,
                        "crop_box": [x0, y0, x1, y1],
                        "anomaly": 1 if has_def else 0,
                        "labels": ",".join(present_classes),
                    })
            try:
                with open(cache_meta_file, "w", encoding="utf-8") as f:
                    json.dump(self.entries, f, indent=2)
            except OSError:
                pass

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, index: int) -> Dict:
        entry = self.entries[index]
        x0, y0, x1, y1 = entry["crop_box"]

        img_full = Image.open(entry["img_path"]).convert("RGB")
        if (x0, y0, x1, y1) != (0, 0, img_full.width, img_full.height):
            img = img_full.crop((x0, y0, x1, y1))
        else:
            img = img_full

        msk_full = Image.open(entry["mask_path"]).convert("RGB")
        if (x0, y0, x1, y1) != (0, 0, msk_full.width, msk_full.height):
            msk_crop = msk_full.crop((x0, y0, x1, y1))
        else:
            msk_crop = msk_full
        rgb_np = np.asarray(msk_crop, dtype=np.uint8)

        # Decode 4 structural defects
        class_masks_stack = []
        multi_label_vec = torch.zeros(len(DEFECT_CLASSES), dtype=torch.float32)
        union_def_np = np.zeros(rgb_np.shape[:2], dtype=bool)

        for c_idx, cname in enumerate(DEFECT_CLASSES):
            rgb = OFFICIAL_S2DS_RGB_MAP[cname]
            cm_bool = (
                (rgb_np[:, :, 0] == rgb[0]) &
                (rgb_np[:, :, 1] == rgb[1]) &
                (rgb_np[:, :, 2] == rgb[2])
            )
            union_def_np |= cm_bool
            if cm_bool.any():
                multi_label_vec[c_idx] = 1.0
            cm_pil = Image.fromarray(cm_bool.astype(np.uint8) * 255, mode="L").resize(
                (self.image_size, self.image_size), resample=Image.NEAREST
            )
            class_masks_stack.append((transforms.ToTensor()(cm_pil) > 0.5).float())

        class_masks_tensor = torch.cat(class_masks_stack, dim=0)  # [4, H, W]

        # Decode 2 explicit benign nuisances (Vegetation + ControlPoint)
        veg_rgb = OFFICIAL_S2DS_RGB_MAP["Vegetation"]
        cp_rgb = OFFICIAL_S2DS_RGB_MAP["ControlPoint"]
        nuis_bool = (
            ((rgb_np[:, :, 0] == veg_rgb[0]) & (rgb_np[:, :, 1] == veg_rgb[1]) & (rgb_np[:, :, 2] == veg_rgb[2])) |
            ((rgb_np[:, :, 0] == cp_rgb[0]) & (rgb_np[:, :, 1] == cp_rgb[1]) & (rgb_np[:, :, 2] == cp_rgb[2]))
        )
        nuis_pil = Image.fromarray(nuis_bool.astype(np.uint8) * 255, mode="L").resize(
            (self.image_size, self.image_size), resample=Image.NEAREST
        )
        nuisance_tensor = (transforms.ToTensor()(nuis_pil) > 0.5).float()

        union_pil = Image.fromarray(union_def_np.astype(np.uint8) * 255, mode="L").resize(
            (self.image_size, self.image_size), resample=Image.NEAREST
        )
        mask_tensor = (transforms.ToTensor()(union_pil) > 0.5).float()

        if self.preprocess is not None:
            img_tensor = self.preprocess(img)
        else:
            img_tensor = transforms.ToTensor()(img)

        anomaly = int(entry["anomaly"])
        out = {
            "img": img_tensor,
            "img_mask": mask_tensor,
            "nuisance_mask": nuisance_tensor,
            "class_masks_tensor": class_masks_tensor,
            "multi_label": multi_label_vec,
            "anomaly": anomaly,
            "img_path": entry["img_path"],
            "sample_id": entry["sample_id"],
        }
        if self.return_class_masks:
            out["class_masks"] = {
                cname: class_masks_tensor[i : i + 1] for i, cname in enumerate(DEFECT_CLASSES)
            }
        return out


def build_dataset(
    dataset_name: str,
    root: str,
    split: str = "train",
    image_size: int = 518,
    preprocess=None,
    return_class_masks: bool = True,
    s2ds_mode: str = "official_full",
):
    name_upper = dataset_name.upper()
    if "CODEBRIM" in name_upper:
        return BridgeCodebrimDataset(
            root=root,
            split=split,
            image_size=image_size,
            preprocess=preprocess,
            return_class_masks=return_class_masks,
        )
    elif "S2DS" in name_upper:
        if not os.path.isfile(os.path.join(root, "train_manifest.csv")):
            return OfficialS2DSDataset(
                root=root,
                split=split,
                image_size=image_size,
                preprocess=preprocess,
                return_class_masks=return_class_masks,
                mode=s2ds_mode,
            )
        else:
            return BridgeCodebrimDataset(
                root=root,
                split=split,
                image_size=image_size,
                preprocess=preprocess,
                return_class_masks=return_class_masks,
            )
    else:
        raise ValueError(f"Unsupported dataset_name: {dataset_name}")
