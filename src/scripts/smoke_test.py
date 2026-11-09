import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from cliplib import get_clip_transform, load_clip_vit_l14_336
from datasets import build_dataset
from models import build_tf_moeclip_variant, FocalLoss, BinaryDiceLoss
from train_eval import compute_tf_moeclip_loss

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    transform = get_clip_transform(518)

    print("1. Testing Bridge512_balance_CODEBRIM & Official S2DS loaders...")
    cb_train = build_dataset("CODEBRIM", "/home/skye/data/Skye/databases/Bridge512_balance_CODEBRIM", split="train", preprocess=transform)
    cb_val = build_dataset("CODEBRIM", "/home/skye/data/Skye/databases/Bridge512_balance_CODEBRIM", split="val", preprocess=transform)
    cb_test = build_dataset("CODEBRIM", "/home/skye/data/Skye/databases/Bridge512_balance_CODEBRIM", split="test", preprocess=transform)

    s2_tiles_train = build_dataset("S2DS", "/home/skye/data/Skye/databases/S2DS", split="train", preprocess=transform, s2ds_mode="official_tiles512")
    s2_tiles_val = build_dataset("S2DS", "/home/skye/data/Skye/databases/S2DS", split="val", preprocess=transform, s2ds_mode="official_tiles512")
    s2_tiles_test = build_dataset("S2DS", "/home/skye/data/Skye/databases/S2DS", split="test", preprocess=transform, s2ds_mode="official_tiles512")

    s2_full_train = build_dataset("S2DS", "/home/skye/data/Skye/databases/S2DS", split="train", preprocess=transform, s2ds_mode="official_full")
    s2_full_val = build_dataset("S2DS", "/home/skye/data/Skye/databases/S2DS", split="val", preprocess=transform, s2ds_mode="official_full")
    s2_full_test = build_dataset("S2DS", "/home/skye/data/Skye/databases/S2DS", split="test", preprocess=transform, s2ds_mode="official_full")

    print(f"  CODEBRIM: train={len(cb_train)}, val={len(cb_val)}, test={len(cb_test)}")
    print(f"  Official S2DS (official_tiles512): train={len(s2_tiles_train)}, val={len(s2_tiles_val)}, test={len(s2_tiles_test)}")
    print(f"  Official S2DS (official_full 1024x1024): train={len(s2_full_train)}, val={len(s2_full_val)}, test={len(s2_full_test)}")

    s0 = s2_tiles_train[0]
    print(f"  S2DS sample[0]: img={tuple(s0['img'].shape)}, mask={tuple(s0['img_mask'].shape)}, nuis={tuple(s0['nuisance_mask'].shape)}, cmasks={tuple(s0['class_masks_tensor'].shape)}, mlabel={s0['multi_label'].tolist()}, anom={s0['anomaly']}")

    print("2. Loading CLIP ViT-L/14@336px and testing A0, A1, A2, A3, A4 forward/backward...")
    clip_model = load_clip_vit_l14_336(device=device)
    loader = torch.utils.data.DataLoader(s2_tiles_train, batch_size=2, shuffle=True)
    batch = next(iter(loader))
    focal_fn = FocalLoss()
    dice_fn = BinaryDiceLoss()

    for var in ["A0", "A1", "A2", "A3", "A4"]:
        m = build_tf_moeclip_variant(var, clip_model).to(device)
        n_p = sum(p.numel() for p in m.parameters() if p.requires_grad)
        out = m(batch["img"].to(device))
        loss, l_dict = compute_tf_moeclip_loss(out, batch, focal_fn, dice_fn, device=device)
        loss.backward()
        print(f"  [{var}] trainable_params={n_p:,} | amap={tuple(out['anomaly_map'].shape)} | loss_total={l_dict['loss_total']:.4f} (seg={l_dict['loss_seg']:.4f}, ac={l_dict['loss_ac']:.4f}, etf={l_dict['loss_etf']:.4f}, sfr={l_dict['loss_sfr']:.4f})")
    print("ALL VARIANTS A0-A4 VERIFIED SUCCESSFULLY!")

if __name__ == "__main__":
    main()
