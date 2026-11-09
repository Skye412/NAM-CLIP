import os
import sys
import gc
import json
import time
import hashlib
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize

TF_SRC = "/home/skye/data/Skye/Tf_MoeClip/src"
if TF_SRC not in sys.path:
    sys.path.insert(0, TF_SRC)

from datasets.bridge_s2ds_dataset import build_dataset, DEFECT_CLASSES
from utils.metrics import evaluate_anomaly_predictions, setup_seed

CODEBRIM_ROOT = "/home/skye/data/Skye/databases/Bridge512_balance_CODEBRIM"
OFFICIAL_S2DS_ROOT = "/home/skye/data/Skye/databases/S2DS"
VISUALAD_OUT = "/home/skye/data/Skye/visualad/outputs"
OUT_DIR = "/home/skye/data/Skye/Tf_MoeClip/outputs/retrain_traceable_20261105_s111/external_sota"
os.makedirs(OUT_DIR, exist_ok=True)

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC


def isolate_external_repo(ext_dir):
    # Remove Tf_MoeClip/src (both symlink and realpath) from sys.path so its models/utils/datasets packages never collide
    real_tf = os.path.realpath(TF_SRC)
    sys.path[:] = [p for p in sys.path if os.path.realpath(p) != real_tf and p != TF_SRC]
    for k in list(sys.modules.keys()):
        if k in ("utils", "models", "model", "datasets", "loss") or k.startswith(("utils.", "models.", "model.", "datasets.", "loss.")):
            sys.modules.pop(k, None)
    os.chdir(ext_dir)
    sys.path.insert(0, ext_dir)


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def collect_and_eval(loader, infer_batch_fn, sigma=4.0, sp_mode="given", compute_pro=True):
    gt_sp_list, pr_sp_list, masks_list, nuis_list, amaps_list = [], [], [], [], []
    class_masks_dict = {c: [] for c in DEFECT_CLASSES}

    with torch.no_grad():
        for batch in tqdm(loader, leave=False):
            amap_batch, sp_batch = infer_batch_fn(batch)
            bsz = amap_batch.shape[0]
            gt_m = (batch["img_mask"] > 0.5).numpy().astype(np.uint8)
            nuis_m = (batch["nuisance_mask"] > 0.5).numpy().astype(np.uint8)
            cm_t = (batch["class_masks_tensor"] > 0.5).numpy().astype(np.uint8)
            anom = batch["anomaly"]
    
            for b in range(bsz):
                amaps_list.append(amap_batch[b].astype(np.float32))
                gt_sp_list.append(int(anom[b].item() if hasattr(anom[b], "item") else anom[b]))
                masks_list.append(gt_m[b, 0])
                nuis_list.append(nuis_m[b, 0])
                for c_idx, cname in enumerate(DEFECT_CLASSES):
                    class_masks_dict[cname].append(cm_t[b, c_idx])
                if sp_batch is not None:
                    pr_sp_list.append(float(sp_batch[b]))

    gt_sp = np.asarray(gt_sp_list, dtype=np.int64)
    masks_np = np.stack(masks_list, axis=0)
    nuis_np = np.stack(nuis_list, axis=0)
    amaps_raw = np.stack(amaps_list, axis=0)

    if sigma > 0 and torch.cuda.is_available():
        with torch.no_grad():
            radius = int(4.0 * sigma + 0.5)
            coords = torch.arange(-radius, radius + 1, dtype=torch.float32, device="cuda")
            k1d = torch.exp(-(coords ** 2) / (2.0 * sigma ** 2))
            k1d = (k1d / k1d.sum()).view(1, 1, -1)
            kx, ky = k1d.unsqueeze(2), k1d.unsqueeze(3)
            amaps_smooth = np.empty_like(amaps_raw, dtype=np.float32)
            for i in range(0, len(amaps_raw), 64):
                t = torch.from_numpy(amaps_raw[i:i+64]).cuda().unsqueeze(1)
                t = F.pad(t, (radius, radius, radius, radius), mode="reflect")
                t = F.conv2d(F.conv2d(t, kx), ky)
                amaps_smooth[i:i+64] = t.squeeze(1).cpu().numpy()
    else:
        amaps_smooth = amaps_raw

    if sp_mode == "topk1":
        flat = amaps_smooth.reshape(amaps_smooth.shape[0], -1)
        k = max(1, int(np.ceil(flat.shape[1] * 0.01)))
        pr_sp = np.sort(flat, axis=1)[:, -k:].mean(axis=1).astype(np.float32)
    elif sp_mode == "adaptclip":
        amap_max = amaps_smooth.reshape(amaps_smooth.shape[0], -1).max(axis=1)
        pr_sp = (np.asarray(pr_sp_list, dtype=np.float32) + amap_max) / 3.0
    elif sp_mode == "bayespfl":
        raw_sp = np.asarray(pr_sp_list, dtype=np.float32)
        sp_norm = (raw_sp - raw_sp.min()) / (raw_sp.max() - raw_sp.min() + 1e-8)
        mn, mx = float(amaps_smooth.min()), float(amaps_smooth.max())
        amaps_n = (amaps_smooth - mn) / (mx - mn + 1e-8)
        flat_amaps = amaps_n.reshape(amaps_n.shape[0], -1)
        kth = max(-2000, -flat_amaps.shape[1])
        pr_sp_tmp = np.partition(flat_amaps, kth=kth, axis=1)[:, kth:].mean(axis=1)
        pr_sp_tmp = (pr_sp_tmp - pr_sp_tmp.min()) / (pr_sp_tmp.max() - pr_sp_tmp.min() + 1e-8)
        pr_sp = 0.5 * sp_norm + 0.5 * pr_sp_tmp
    else:
        pr_sp = np.asarray(pr_sp_list, dtype=np.float32)

    return evaluate_anomaly_predictions(
        gt_sp=gt_sp,
        pr_sp=pr_sp,
        masks_np=masks_np,
        amaps_raw=amaps_smooth,
        class_masks_dict=class_masks_dict,
        nuisance_masks_np=nuis_np,
        apply_gaussian_sigma=0.0,
        compute_pro=compute_pro,
    )


def eval_method(method_name):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if method_name == "adaptclip":
        isolate_external_repo("/home/skye/data/Skye/AdaptCLIP_paper_a55cfbf")
        import adaptcliplib
        from adaptcliplib import TextualAdapter, VisualAdapter
        ckpt_path = f"{VISUALAD_OUT}/comp_adaptclip_CODEBRIM/best_model.pth"
        ckpt = torch.load(ckpt_path, map_location=device)
        preprocess = Compose([
            Resize((518, 518), interpolation=BICUBIC),
            CenterCrop((518, 518)),
            lambda img: img.convert("RGB"),
            ToTensor(),
            Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])
        model, _ = adaptcliplib.load("ViT-L/14@336px", device=device, download_root="/home/skye/data/Skye/Bayes-PFL-main/pretrained_weight")
        model.visual.DAPM_replace(DPAM_layer=20)
        textual_learner = TextualAdapter(model.to("cpu"), 518, 12)
        visual_learner = VisualAdapter(518, 14, input_dim=768, reduction=4)
        model.to(device).eval()
        textual_learner.to(device)
        visual_learner.to(device)
        textual_learner.prepare_static_text_feature(model)
        textual_learner.load_state_dict(ckpt["textual_learner"])
        visual_learner.load_state_dict(ckpt["visual_learner"])
        textual_learner.eval()
        visual_learner.eval()
        trainable_params = sum(p.numel() for p in textual_learner.parameters()) + sum(p.numel() for p in visual_learner.parameters())

        with torch.no_grad():
            static_tf = textual_learner.static_text_features
            l_prompts, tok_prompts = textual_learner()
            learned_tf = model.encode_text(l_prompts, tok_prompts).float()

        @torch.no_grad()
        def infer_fn(batch):
            img = batch["img"].to(device)
            with torch.cuda.amp.autocast():
                q_feats, q_pfeats = model.encode_image(img, [6, 12, 18, 24], DPAM_layer=20)
            q_feats = q_feats.float()
            q_pfeats = [pf.float() for pf in q_pfeats]
            g_vl_logit, l_vl_map = visual_learner(q_feats, q_pfeats, static_tf)
            g_tl_logit, l_tl_map = textual_learner.compute_global_local_score(q_feats, q_pfeats, learned_tf)
            amap = 0.5 * (l_vl_map[:, 1] + l_tl_map[:, 1])
            g_sum = g_vl_logit.softmax(-1)[:, 1] + g_tl_logit.softmax(-1)[:, 1]
            return amap.float().cpu().numpy(), g_sum.float().cpu().numpy()

        sigma, sp_mode, label, proto = 4.0, "adaptclip", "AdaptCLIP (AAAI 2026)", "15 Ep (val-best)"

    elif method_name == "anomalyclip":
        isolate_external_repo("/home/skye/data/Skye/third_party/AnomalyCLIP")
        try:
            import thop
        except ImportError:
            import types
            _thop = types.ModuleType("thop")
            _thop.profile = lambda *a, **kw: (0, 0)
            sys.modules["thop"] = _thop
        import AnomalyCLIP_lib
        from prompt_ensemble import AnomalyCLIP_PromptLearner
        from utils import get_transform
        ckpt_path = f"{VISUALAD_OUT}/comp_anomalyclip_CODEBRIM/best_model.pth"
        ckpt = torch.load(ckpt_path, map_location=device)
        preprocess, _ = get_transform(argparse.Namespace(image_size=518))
        params = {"Prompt_length": 12, "learnabel_text_embedding_depth": 9, "learnabel_text_embedding_length": 4}
        model, _ = AnomalyCLIP_lib.load("ViT-L/14@336px", device=device, design_details=params, download_root="/home/skye/data/Skye/Bayes-PFL-main/pretrained_weight")
        prompt_learner = AnomalyCLIP_PromptLearner(model.to("cpu"), params).to(device)
        model.to(device).eval()
        model.visual.DAPM_replace(DPAM_layer=20)
        prompt_learner.load_state_dict(ckpt["prompt_learner"])
        prompt_learner.eval()
        trainable_params = sum(p.numel() for p in prompt_learner.parameters() if p.requires_grad)

        with torch.no_grad():
            prompts, tok_prompts, comp_prompts = prompt_learner(cls_id=None)
            tf = model.encode_text_learn(prompts, tok_prompts, comp_prompts).float()
            tf = torch.stack(torch.chunk(tf, dim=0, chunks=2), dim=1)
            tf = tf / tf.norm(dim=-1, keepdim=True)

        @torch.no_grad()
        def infer_fn(batch):
            img = batch["img"].to(device)
            with torch.cuda.amp.autocast():
                img_f, patch_f = model.encode_image(img, [6, 12, 18, 24], DPAM_layer=20)
            img_f = img_f.float() / img_f.float().norm(dim=-1, keepdim=True)
            t_probs = (img_f.unsqueeze(1) @ tf.permute(0, 2, 1) / 0.07).softmax(-1)[:, 0, 1]
            amap_list = []
            for pf in patch_f:
                pf = pf.float() / pf.float().norm(dim=-1, keepdim=True)
                sim, _ = AnomalyCLIP_lib.compute_similarity(pf, tf[0])
                sim_map = AnomalyCLIP_lib.get_similarity_map(sim[:, 1:, :], 518)
                amap_list.append((sim_map[..., 1] + 1.0 - sim_map[..., 0]) / 2.0)
            amap = torch.stack(amap_list, dim=0).sum(dim=0)
            return amap.float().cpu().numpy(), t_probs.float().cpu().numpy()

        sigma, sp_mode, label, proto = 4.0, "given", "AnomalyCLIP (ICLR 2024)", "15 Ep (val-best)"

    elif method_name == "bayespfl":
        BAYESPFL_DIR = "/home/skye/data/Skye/Bayes-PFL-main"
        isolate_external_repo(BAYESPFL_DIR)
        from models.VPB import Context_Prompting, TextEncoder
        from models.model_CLIP import Load_CLIP, tokenize
        ckpt_path = f"{VISUALAD_OUT}/comp_bayespfl_CODEBRIM/best_model.pth"
        ckpt = torch.load(ckpt_path, map_location=device)
        best_stg = ckpt.get("stage", 1)

        preprocess = Compose([
            Resize((518, 518), interpolation=BICUBIC),
            CenterCrop((518, 518)),
            lambda img: img.convert("RGB"),
            ToTensor(),
            Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])
        args = argparse.Namespace(
            dataset="CODEBRIM", model="ViT-L-14-336", pretrained="openai", features_list=[6, 12, 18, 24],
            config_path=os.path.join(BAYESPFL_DIR, "open_clip_local/model_configs/ViT-L-14-336.json"),
            pretrained_path=os.path.join(BAYESPFL_DIR, "pretrained_weight/ViT-L-14-336px.pt"),
            num_flows=10, prompt_context_len=5, prompt_num=3, prompt_state_len=5,
            sample_num=5, batch_size=16, image_size=518, device_id=0, seed=333,
        )
        model_configs = json.load(open(args.config_path))
        args.vision_width = model_configs["vision_cfg"]["width"]
        args.text_width = model_configs["text_cfg"]["width"]
        args.embed_dim = model_configs["embed_dim"]
        model_clip, _, _ = Load_CLIP(518, args.pretrained_path, device=device)
        model_clip.to(device).eval()
        PFL_TextEncoder = TextEncoder(model_clip, args)
        MyModel = Context_Prompting(args=args).to(device)
        MyModel.load_state_dict(ckpt["MyModel"])
        MyModel.eval()
        trainable_params = sum(p.numel() for p in MyModel.parameters() if p.requires_grad)
        R_tot = args.prompt_num * args.sample_num

        @torch.no_grad()
        def infer_fn(batch):
            setup_seed(333)
            MyModel.zk_n = None
            MyModel.zk_a = None
            imgs = batch["img"].to(device)
            bsz = imgs.shape[0]
            with torch.cuda.amp.autocast():
                img_f_all, _, patch_t_all = model_clip.encode_image(imgs, args.features_list)
            img_f_all = img_f_all.float()
            patch_t_all = [pt.float() for pt in patch_t_all]
            amaps_b, sps_b = [], []
            for b in range(bsz):
                img_fb = img_f_all[b:b+1]
                pt_b = [pt[b:b+1] for pt in patch_t_all]
                te, _ = MyModel.forward_ensemble(PFL_TextEncoder, img_fb, pt_b, ["bridge"], device, tokenize, mode="test")
                pro_img, amap_list = MyModel(te, img_fb, pt_b, stage=best_stg, mode="test")
                pro_img = pro_img.squeeze(2)
                t_probs = torch.stack([pro_img[:, :R_tot], pro_img[:, R_tot:]], dim=-1).softmax(dim=-1)
                sps_b.append(float(t_probs[0, :, 1].mean().item()))
                layers_m = []
                for am in amap_list:
                    al = torch.stack([am[:, :R_tot], am[:, R_tot:]], dim=2).softmax(dim=2)[:, :, 1].mean(dim=1)
                    layers_m.append(al.float().cpu().numpy()[0])
                amaps_b.append(np.mean(layers_m, axis=0))
            return np.stack(amaps_b, axis=0), np.asarray(sps_b, dtype=np.float32)

        sigma, sp_mode, label, proto = 8.0, "bayespfl", "Bayes-PFL (CVPR 2025)", "Up to 30 Ep (2-Stg, early-stop)"

    elif method_name == "wpformer":
        WPFORMER_DIR = "/home/skye/data/Skye/WPFormer"
        isolate_external_repo(WPFORMER_DIR)
        from model.WPFormer import WPFormer
        ckpt_path = f"{VISUALAD_OUT}/comp_wpformer_CODEBRIM/best_model.pth"
        preprocess = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        model = WPFormer(method="pvt_v2_b2", channel=64).to(device)
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        model.eval()
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        @torch.no_grad()
        def infer_fn(batch):
            imgs = batch["img"].to(device)
            with torch.cuda.amp.autocast():
                preds = model(imgs)[-1]
            probs = torch.sigmoid(preds.float())
            probs_up = F.interpolate(probs, size=(518, 518), mode="bilinear", align_corners=False)[:, 0]
            return probs_up.float().cpu().numpy(), None

        sigma, sp_mode, label, proto = 1.5, "topk1", "WPFormer (CVPR 2025)", "60 Ep (val-best)"

    elif method_name == "mixercseg":
        MIXERCSEG_DIR = "/home/skye/data/Skye/MixerCSeg"
        isolate_external_repo(MIXERCSEG_DIR)
        from models import build_MixerCSeg
        ckpt_path = f"{VISUALAD_OUT}/comp_mixercseg_CODEBRIM/best_model.pth"
        preprocess = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        args_m = argparse.Namespace(BCELoss_ratio=0.87, DiceLoss_ratio=0.13, Norm_Type="GN", device="cuda", load_width=512, load_height=512)
        model, _ = build_MixerCSeg(args_m)
        model.to(device).load_state_dict(torch.load(ckpt_path, map_location=device))
        model.eval()
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        @torch.no_grad()
        def infer_fn(batch):
            imgs = batch["img"].to(device)
            out = model(imgs)
            if isinstance(out, (list, tuple)):
                out = out[-1]
            probs = torch.sigmoid(out.float()) if (out.min() < 0.0 or out.max() > 1.0) else out.float()
            probs_up = F.interpolate(probs, size=(518, 518), mode="bilinear", align_corners=False)[:, 0]
            return probs_up.float().cpu().numpy(), None

        sigma, sp_mode, label, proto = 1.5, "topk1", "MixerCSeg (CVPR 2026)", "50 Ep (val-best)"

    elif method_name == "dinomaly":
        DINOMALY_DIR = "/home/skye/data/Skye/Dinomaly"
        isolate_external_repo(DINOMALY_DIR)
        from dinomaly_bridge import build_model
        from utils import cal_anomaly_maps
        ckpt_path = "/home/skye/data/Skye/Dinomaly/runs/codebrim512-v0-b16-s1-review20261018/best_val.pth"
        preprocess = transforms.Compose([
            transforms.Resize((392, 392), interpolation=Image.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        model, trainable = build_model(device, variant="v0")
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        model.eval()
        trainable_params = sum(p.numel() for p in trainable.parameters())

        @torch.no_grad()
        def infer_fn(batch):
            imgs = batch["img"].to(device)
            en, de = model(imgs)
            amap, _ = cal_anomaly_maps(en, de, 518)
            return amap[:, 0].float().cpu().numpy(), None

        sigma, sp_mode, label, proto = 4.0, "topk1", "Dinomaly (CVPR 2025)", "10k iters (val-best)"

    else:
        raise ValueError(f"Unknown method: {method_name}")

    cb_test = build_dataset("CODEBRIM", CODEBRIM_ROOT, split="test", image_size=518, preprocess=preprocess)
    s2_test = build_dataset("S2DS", OFFICIAL_S2DS_ROOT, split="test", image_size=518, preprocess=preprocess, s2ds_mode="official_full")

    cb_loader = torch.utils.data.DataLoader(cb_test, batch_size=16, shuffle=False, num_workers=4, pin_memory=True)
    s2_loader = torch.utils.data.DataLoader(s2_test, batch_size=16, shuffle=False, num_workers=4, pin_memory=True)

    print(f"\n>>> Evaluating {label} (trained on CODEBRIM) on Bridge512 & Official S2DS ...", flush=True)
    in_m = collect_and_eval(cb_loader, infer_fn, sigma=sigma, sp_mode=sp_mode)
    print(f"[{label} In-Domain Bridge512] Mean-7={in_m['Mean-Score']:.2f} (legacy={in_m['Mean-Score-legacy']:.2f}) | P-AUPRO={in_m['Pixel-AUPRO']:.2f} (legacy={in_m['Pixel-AUPRO-legacy']:.2f})", flush=True)

    cross_m = collect_and_eval(s2_loader, infer_fn, sigma=sigma, sp_mode=sp_mode)
    print(f"[{label} Cross-Domain Bridge512 -> Official S2DS] Mean-7={cross_m['Mean-Score']:.2f} | P-AUPRO={cross_m['Pixel-AUPRO']:.2f} | Nuisance-FPR={cross_m['Nuisance-FPR']:.2f}%", flush=True)

    rec = {
        "method": label,
        "variant": f"comp_{method_name}",
        "protocol": proto,
        "train_dataset": "Bridge512",
        "trainable_params": trainable_params,
        "in_domain_test_metrics": in_m,
        "cross_domain_name": "S2DS",
        "cross_domain_test_metrics": cross_m,
        "provenance": {
            "written_by": "eval_ext_step1.py",
            "written_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "ckpt_file": ckpt_path,
            "ckpt_sha256": file_sha256(ckpt_path),
            "ckpt_bytes": os.path.getsize(ckpt_path),
        },
    }
    out_json = os.path.join(OUT_DIR, f"metrics_comp_{method_name}_Bridge512.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(rec, f, indent=2)
    print(f"Saved: {out_json}", flush=True)


if __name__ == "__main__":
    eval_method(sys.argv[1])
