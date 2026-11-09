import os
import sys
import gc
import json
import math
import time
import shutil
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
from eval_ext_step1 import collect_and_eval, file_sha256, isolate_external_repo

CODEBRIM_ROOT = "/home/skye/data/Skye/databases/Bridge512_balance_CODEBRIM"
OFFICIAL_S2DS_ROOT = "/home/skye/data/Skye/databases/S2DS"
OUT_DIR = "/home/skye/data/Skye/Tf_MoeClip/outputs/retrain_traceable_20261105_s111/external_sota"
os.makedirs(OUT_DIR, exist_ok=True)

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC


def train_and_eval_s2ds(method_name):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    ckpt_dir = os.path.join(OUT_DIR, f"ckpts_{method_name}_S2DS")
    os.makedirs(ckpt_dir, exist_ok=True)
    best_ckpt = os.path.join(ckpt_dir, "best_model.pth")

    if method_name == "adaptclip":
        isolate_external_repo("/home/skye/data/Skye/AdaptCLIP_paper_a55cfbf")
        import adaptcliplib
        from adaptcliplib import TextualAdapter, VisualAdapter, BinaryDiceLoss, FocalLoss
        setup_seed(10)
        epochs, batch_size, lr = 15, 8, 0.001
        preprocess = Compose([
            Resize((518, 518), interpolation=BICUBIC),
            CenterCrop((518, 518)),
            lambda img: img.convert("RGB"),
            ToTensor(),
            Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])
        s2_train = build_dataset("S2DS", OFFICIAL_S2DS_ROOT, split="train", image_size=518, preprocess=preprocess, s2ds_mode="official_full")
        s2_val = build_dataset("S2DS", OFFICIAL_S2DS_ROOT, split="val", image_size=518, preprocess=preprocess, s2ds_mode="official_full")
        s2_test = build_dataset("S2DS", OFFICIAL_S2DS_ROOT, split="test", image_size=518, preprocess=preprocess, s2ds_mode="official_full")
        cb_test = build_dataset("CODEBRIM", CODEBRIM_ROOT, split="test", image_size=518, preprocess=preprocess)

        model, _ = adaptcliplib.load("ViT-L/14@336px", device=device, download_root="/home/skye/data/Skye/Bayes-PFL-main/pretrained_weight")
        model.visual.DAPM_replace(DPAM_layer=20)
        textual_learner = TextualAdapter(model.to("cpu"), 518, 12)
        visual_learner = VisualAdapter(518, 14, input_dim=768, reduction=4)
        model.to(device).eval()
        textual_learner.to(device)
        visual_learner.to(device)
        textual_learner.prepare_static_text_feature(model)
        trainable_params = sum(p.numel() for p in textual_learner.parameters()) + sum(p.numel() for p in visual_learner.parameters())

        optimizer = torch.optim.Adam(list(textual_learner.parameters()) + list(visual_learner.parameters()), lr=lr, betas=(0.5, 0.999))
        loss_focal, loss_dice = FocalLoss(), BinaryDiceLoss()

        # Pre-extract features in RAM (only 563 train + 87 val images!)
        def extract_ram(ds):
            ldr = torch.utils.data.DataLoader(ds, batch_size=16, shuffle=False, num_workers=4)
            img_l, pf_l, gt_l, lab_l = [], [], [], []
            with torch.no_grad():
                for b in tqdm(ldr, desc="Extract RAM", leave=False):
                    im = b["img"].to(device)
                    with torch.cuda.amp.autocast():
                        qf, qpf = model.encode_image(im, [6, 12, 18, 24], DPAM_layer=20)
                        pfs = torch.stack(qpf, dim=1)
                    img_l.append(qf.half().cpu())
                    pf_l.append(pfs.half().cpu())
                    gt_l.append((b["img_mask"].squeeze(1) > 0.5).to(torch.uint8))
                    lab_l.append(torch.as_tensor(b["anomaly"], dtype=torch.long))
            return torch.cat(img_l, 0), torch.cat(pf_l, 0), torch.cat(gt_l, 0), torch.cat(lab_l, 0)

        tr_img, tr_pf, tr_gt, tr_lab = extract_ram(s2_train)
        va_loader = torch.utils.data.DataLoader(s2_val, batch_size=16, shuffle=False, num_workers=4)
        n_train = len(tr_lab)
        best_val = -1.0

        for ep in range(epochs):
            textual_learner.train()
            visual_learner.train()
            perm = torch.randperm(n_train)
            losses = []
            for start in range(0, n_train, batch_size):
                idx = perm[start:start+batch_size]
                if len(idx) < 2:
                    continue
                qf = tr_img[idx].float().to(device)
                pfb = tr_pf[idx].float().to(device)
                qpf = [pfb[:, i].contiguous() for i in range(pfb.shape[1])]
                lab = tr_lab[idx].to(device)
                gt = tr_gt[idx].float().to(device)

                stf = textual_learner.static_text_features
                g_vl, l_vl = visual_learner(qf, qpf, stf)
                loss_g = F.cross_entropy(g_vl, lab)
                loss_l = loss_focal(l_vl, gt) + loss_dice(l_vl[:, 1], gt) + loss_dice(l_vl[:, 0], 1.0 - gt)

                lp, tp = textual_learner()
                ltf = model.encode_text(lp, tp).float()
                g_tl, l_tl = textual_learner.compute_global_local_score(qf, qpf, ltf)
                loss_g = loss_g + F.cross_entropy(g_tl, lab)
                loss_l = loss_l + loss_focal(l_tl, gt) + loss_dice(l_tl[:, 1], gt) + loss_dice(l_tl[:, 0], 1.0 - gt)

                loss = loss_l + loss_g
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses.append(float(loss.item()))

            textual_learner.eval()
            visual_learner.eval()
            with torch.no_grad():
                stf = textual_learner.static_text_features
                lp, tp = textual_learner()
                ltf = model.encode_text(lp, tp).float()

                def val_infer(batch):
                    im = batch["img"].to(device)
                    with torch.cuda.amp.autocast():
                        qf, qpf = model.encode_image(im, [6, 12, 18, 24], DPAM_layer=20)
                    qf = qf.float()
                    qpf = [p.float() for p in qpf]
                    g_vl, l_vl = visual_learner(qf, qpf, stf)
                    g_tl, l_tl = textual_learner.compute_global_local_score(qf, qpf, ltf)
                    amap = 0.5 * (l_vl[:, 1] + l_tl[:, 1])
                    g_sum = g_vl.softmax(-1)[:, 1] + g_tl.softmax(-1)[:, 1]
                    return amap.float().cpu().numpy(), g_sum.float().cpu().numpy()

            vm = collect_and_eval(va_loader, val_infer, sigma=4.0, sp_mode="adaptclip", compute_pro=False)
            vkey = (vm["Image-AUROC"] + vm["Pixel-AUROC"] + vm["Pixel-F1"] + vm["Pixel-AP"]) / 4.0
            print(f"[AdaptCLIP S2DS Ep {ep+1}/{epochs}] loss={np.mean(losses):.4f} | Val Key={vkey:.2f} (P-AP={vm['Pixel-AP']:.2f})", flush=True)
            if vkey >= best_val:
                best_val = vkey
                torch.save({"textual_learner": textual_learner.state_dict(), "visual_learner": visual_learner.state_dict()}, best_ckpt)

        ckpt = torch.load(best_ckpt, map_location=device)
        textual_learner.load_state_dict(ckpt["textual_learner"])
        visual_learner.load_state_dict(ckpt["visual_learner"])
        textual_learner.eval()
        visual_learner.eval()
        with torch.no_grad():
            stf = textual_learner.static_text_features
            lp, tp = textual_learner()
            ltf = model.encode_text(lp, tp).float()

            def test_infer(batch):
                im = batch["img"].to(device)
                with torch.cuda.amp.autocast():
                    qf, qpf = model.encode_image(im, [6, 12, 18, 24], DPAM_layer=20)
                qf = qf.float()
                qpf = [p.float() for p in qpf]
                g_vl, l_vl = visual_learner(qf, qpf, stf)
                g_tl, l_tl = textual_learner.compute_global_local_score(qf, qpf, ltf)
                amap = 0.5 * (l_vl[:, 1] + l_tl[:, 1])
                g_sum = g_vl.softmax(-1)[:, 1] + g_tl.softmax(-1)[:, 1]
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
        from loss import FocalLoss, BinaryDiceLoss
        sys.modules.pop("utils", None); from utils import get_transform

        setup_seed(111)
        epochs, batch_size, lr = 15, 8, 0.001
        preprocess, _ = get_transform(argparse.Namespace(image_size=518))
        s2_train = build_dataset("S2DS", OFFICIAL_S2DS_ROOT, split="train", image_size=518, preprocess=preprocess, s2ds_mode="official_full")
        s2_val = build_dataset("S2DS", OFFICIAL_S2DS_ROOT, split="val", image_size=518, preprocess=preprocess, s2ds_mode="official_full")
        s2_test = build_dataset("S2DS", OFFICIAL_S2DS_ROOT, split="test", image_size=518, preprocess=preprocess, s2ds_mode="official_full")
        cb_test = build_dataset("CODEBRIM", CODEBRIM_ROOT, split="test", image_size=518, preprocess=preprocess)

        params = {"Prompt_length": 12, "learnabel_text_embedding_depth": 9, "learnabel_text_embedding_length": 4}
        model, _ = AnomalyCLIP_lib.load("ViT-L/14@336px", device=device, design_details=params, download_root="/home/skye/data/Skye/Bayes-PFL-main/pretrained_weight")
        prompt_learner = AnomalyCLIP_PromptLearner(model.to("cpu"), params).to(device)
        model.to(device).eval()
        model.visual.DAPM_replace(DPAM_layer=20)
        trainable_params = sum(p.numel() for p in prompt_learner.parameters() if p.requires_grad)

        optimizer = torch.optim.Adam(list(prompt_learner.parameters()), lr=lr, betas=(0.5, 0.999))
        loss_focal, loss_dice = FocalLoss(), BinaryDiceLoss()

        ldr = torch.utils.data.DataLoader(s2_train, batch_size=16, shuffle=False, num_workers=4)
        img_l, pf_l, gt_l, lab_l = [], [], [], []
        with torch.no_grad():
            for b in tqdm(ldr, desc="Extract AnomalyCLIP RAM", leave=False):
                im = b["img"].to(device)
                with torch.cuda.amp.autocast():
                    imf, pf = model.encode_image(im, [6, 12, 18, 24], DPAM_layer=20)
                imf = imf.float() / imf.float().norm(dim=-1, keepdim=True)
                pfs = torch.stack([p.float() / p.float().norm(dim=-1, keepdim=True) for p in pf], dim=1)
                img_l.append(imf.half().cpu())
                pf_l.append(pfs.half().cpu())
                gt_l.append((b["img_mask"].squeeze(1) > 0.5).to(torch.uint8))
                lab_l.append(torch.as_tensor(b["anomaly"], dtype=torch.long))
        tr_img, tr_pf, tr_gt, tr_lab = torch.cat(img_l, 0), torch.cat(pf_l, 0), torch.cat(gt_l, 0), torch.cat(lab_l, 0)
        va_loader = torch.utils.data.DataLoader(s2_val, batch_size=16, shuffle=False, num_workers=4)
        n_train = len(tr_lab)
        best_val = -1.0

        for ep in range(epochs):
            prompt_learner.train()
            perm = torch.randperm(n_train)
            losses = []
            for start in range(0, n_train, batch_size):
                idx = perm[start:start+batch_size]
                imf = tr_img[idx].float().to(device)
                pfb = tr_pf[idx].float().to(device)
                lab = tr_lab[idx].to(device)
                gt = tr_gt[idx].float().to(device)

                prompts, tok_p, comp_p = prompt_learner(cls_id=None)
                tf = model.encode_text_learn(prompts, tok_p, comp_p).float()
                tf = torch.stack(torch.chunk(tf, dim=0, chunks=2), dim=1)
                tf = tf / tf.norm(dim=-1, keepdim=True)

                t_probs = (imf.unsqueeze(1) @ tf.permute(0, 2, 1))[:, 0, ...] / 0.07
                if t_probs.dim() == 1:
                    t_probs = t_probs.unsqueeze(0)
                loss_img = F.cross_entropy(t_probs, lab)

                loss_seg = 0.0
                for i in range(pfb.shape[1]):
                    sim, _ = AnomalyCLIP_lib.compute_similarity(pfb[:, i], tf[0])
                    sim_m = AnomalyCLIP_lib.get_similarity_map(sim[:, 1:, :], 518).permute(0, 3, 1, 2)
                    loss_seg = loss_seg + loss_focal(sim_m, gt) + loss_dice(sim_m[:, 1], gt) + loss_dice(sim_m[:, 0], 1.0 - gt)

                loss = loss_img + 4.0 * loss_seg
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses.append(float(loss.item()))

            prompt_learner.eval()
            with torch.no_grad():
                prompts, tok_p, comp_p = prompt_learner(cls_id=None)
                tf = model.encode_text_learn(prompts, tok_p, comp_p).float()
                tf = torch.stack(torch.chunk(tf, dim=0, chunks=2), dim=1)
                tf = tf / tf.norm(dim=-1, keepdim=True)

                def val_infer(batch):
                    im = batch["img"].to(device)
                    with torch.cuda.amp.autocast():
                        imf, pf = model.encode_image(im, [6, 12, 18, 24], DPAM_layer=20)
                    imf = imf.float() / imf.float().norm(dim=-1, keepdim=True)
                    tp = (imf.unsqueeze(1) @ tf.permute(0, 2, 1) / 0.07).softmax(-1)[:, 0, 1]
                    alist = []
                    for p in pf:
                        p = p.float() / p.float().norm(dim=-1, keepdim=True)
                        sim, _ = AnomalyCLIP_lib.compute_similarity(p, tf[0])
                        sm = AnomalyCLIP_lib.get_similarity_map(sim[:, 1:, :], 518)
                        alist.append((sm[..., 1] + 1.0 - sm[..., 0]) / 2.0)
                    return torch.stack(alist, 0).sum(0).float().cpu().numpy(), tp.float().cpu().numpy()

            vm = collect_and_eval(va_loader, val_infer, sigma=4.0, sp_mode="given", compute_pro=False)
            vkey = (vm["Image-AUROC"] + vm["Pixel-AUROC"] + vm["Pixel-F1"] + vm["Pixel-AP"]) / 4.0
            print(f"[AnomalyCLIP S2DS Ep {ep+1}/{epochs}] loss={np.mean(losses):.4f} | Val Key={vkey:.2f} (P-AP={vm['Pixel-AP']:.2f})", flush=True)
            if vkey >= best_val:
                best_val = vkey
                torch.save({"prompt_learner": prompt_learner.state_dict()}, best_ckpt)

        ckpt = torch.load(best_ckpt, map_location=device)
        prompt_learner.load_state_dict(ckpt["prompt_learner"])
        prompt_learner.eval()
        with torch.no_grad():
            prompts, tok_p, comp_p = prompt_learner(cls_id=None)
            tf = model.encode_text_learn(prompts, tok_p, comp_p).float()
            tf = torch.stack(torch.chunk(tf, dim=0, chunks=2), dim=1)
            tf = tf / tf.norm(dim=-1, keepdim=True)

            def test_infer(batch):
                im = batch["img"].to(device)
                with torch.cuda.amp.autocast():
                    imf, pf = model.encode_image(im, [6, 12, 18, 24], DPAM_layer=20)
                imf = imf.float() / imf.float().norm(dim=-1, keepdim=True)
                tp = (imf.unsqueeze(1) @ tf.permute(0, 2, 1) / 0.07).softmax(-1)[:, 0, 1]
                alist = []
                for p in pf:
                    p = p.float() / p.float().norm(dim=-1, keepdim=True)
                    sim, _ = AnomalyCLIP_lib.compute_similarity(p, tf[0])
                    sm = AnomalyCLIP_lib.get_similarity_map(sim[:, 1:, :], 518)
                    alist.append((sm[..., 1] + 1.0 - sm[..., 0]) / 2.0)
                return torch.stack(alist, 0).sum(0).float().cpu().numpy(), tp.float().cpu().numpy()

        sigma, sp_mode, label, proto = 4.0, "given", "AnomalyCLIP (ICLR 2024)", "15 Ep (val-best)"

    elif method_name == "bayespfl":
        BAYESPFL_DIR = "/home/skye/data/Skye/Bayes-PFL-main"
        isolate_external_repo(BAYESPFL_DIR)
        from loss import FocalLoss, DiceLoss, Orthogonal_Loss
        from models.VPB import Context_Prompting, TextEncoder
        from models.model_CLIP import Load_CLIP, tokenize

        setup_seed(333)
        epochs, batch_size = 30, 16
        preprocess = Compose([
            Resize((518, 518), interpolation=BICUBIC),
            CenterCrop((518, 518)),
            lambda img: img.convert("RGB"),
            ToTensor(),
            Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])
        s2_train = build_dataset("S2DS", OFFICIAL_S2DS_ROOT, split="train", image_size=518, preprocess=preprocess, s2ds_mode="official_full")
        s2_val = build_dataset("S2DS", OFFICIAL_S2DS_ROOT, split="val", image_size=518, preprocess=preprocess, s2ds_mode="official_full")
        s2_test = build_dataset("S2DS", OFFICIAL_S2DS_ROOT, split="test", image_size=518, preprocess=preprocess, s2ds_mode="official_full")
        cb_test = build_dataset("CODEBRIM", CODEBRIM_ROOT, split="test", image_size=518, preprocess=preprocess)

        args = argparse.Namespace(
            dataset="S2DS", model="ViT-L-14-336", pretrained="openai", features_list=[6, 12, 18, 24],
            config_path=os.path.join(BAYESPFL_DIR, "open_clip_local/model_configs/ViT-L-14-336.json"),
            pretrained_path=os.path.join(BAYESPFL_DIR, "pretrained_weight/ViT-L-14-336px.pt"),
            num_flows=10, prompt_context_len=5, prompt_num=3, prompt_state_len=5,
            learning_rate=0.0001, learning_rate_PFL=0.001, learning_rate_prompt=0.001, learning_rate_linear=0.0001,
            sample_num=5, batch_size=batch_size, image_size=518, epochs=epochs, device_id=0, seed=333,
        )
        mc = json.load(open(args.config_path))
        args.vision_width, args.text_width, args.embed_dim = mc["vision_cfg"]["width"], mc["text_cfg"]["width"], mc["embed_dim"]
        model_clip, _, _ = Load_CLIP(518, args.pretrained_path, device=device)
        model_clip.to(device).eval()
        PFL_TextEncoder = TextEncoder(model_clip, args)
        MyModel = Context_Prompting(args=args).to(device)
        trainable_params = sum(p.numel() for p in MyModel.parameters() if p.requires_grad)

        p_prompt, p_prompt2, p_lin, p_lin_cls = [], [], [], []
        for n, m in MyModel.named_parameters():
            if "prompt" in n: p_prompt.append(m)
            elif "PFL" in n and "encoder" not in n and "decoder" not in n: p_prompt2.append(m)
            elif "temperature" not in n: p_lin.append(m)
            if "class_mapping" in n or "image_mapping" in n or "fuse" in n: p_lin_cls.append(m)

        opt1 = torch.optim.Adam([
            {"params": p_prompt, "lr": args.learning_rate_prompt},
            {"params": [MyModel.temperature_pixel, MyModel.temperature_image], "lr": 0.01},
            {"params": p_prompt2, "lr": args.learning_rate_PFL},
            {"params": p_lin, "lr": args.learning_rate_linear},
        ], lr=args.learning_rate, betas=(0.5, 0.999))
        opt2 = torch.optim.Adam([{"params": p_lin_cls, "lr": args.learning_rate_linear}], lr=args.learning_rate, betas=(0.5, 0.999))

        ldr = torch.utils.data.DataLoader(s2_train, batch_size=16, shuffle=False, num_workers=4)
        img_l, pf_l, gt_l, lab_l = [], [], [], []
        with torch.no_grad():
            for b in tqdm(ldr, desc="Extract BayesPFL RAM", leave=False):
                im = b["img"].to(device)
                with torch.cuda.amp.autocast():
                    imf, _, pt = model_clip.encode_image(im, args.features_list)
                img_l.append(imf.half().cpu())
                pf_l.append(torch.stack(pt, dim=1).half().cpu())
                gt_l.append((b["img_mask"].squeeze(1) > 0.5).to(torch.uint8))
                lab_l.append(torch.as_tensor(b["anomaly"], dtype=torch.long))
        tr_img, tr_pf, tr_gt, tr_lab = torch.cat(img_l, 0), torch.cat(pf_l, 0), torch.cat(gt_l, 0), torch.cat(lab_l, 0)
        va_loader = torch.utils.data.DataLoader(s2_val, batch_size=16, shuffle=False, num_workers=4)
        n_train = len(tr_lab)

        steps_per_ep = max(n_train // batch_size, 1)
        total_steps = max(steps_per_ep * epochs, 1)
        warmup_steps = int(0.1 * total_steps)
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            opt1,
            lambda s: float(s) / max(1, warmup_steps) if s < warmup_steps else max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(s - warmup_steps) / max(1, total_steps - warmup_steps))))
        )

        loss_dice_fn, loss_focal_fn, loss_ort_fn, loss_cross = DiceLoss(), FocalLoss(), Orthogonal_Loss(), nn.CrossEntropyLoss()
        stage, optimizer, patience, ap_max_px, ap_max_im, best_val = 1, opt1, 0, -1.0, -1.0, -1.0
        s1_ckpt = os.path.join(ckpt_dir, "stage1_best.pth")
        R_tot = args.prompt_num * args.sample_num

        for ep in range(epochs):
            MyModel.train()
            perm = torch.randperm(n_train)
            losses = []
            for start in range(0, n_train - batch_size + 1, batch_size):
                idx = perm[start:start+batch_size]
                imf = tr_img[idx].float().to(device)
                pfb = tr_pf[idx].float().to(device)
                pt = [pfb[:, i].contiguous() for i in range(pfb.shape[1])]
                lab = tr_lab[idx].to(device)
                gt = tr_gt[idx].float().to(device)

                te, l_reg = MyModel.forward_ensemble(PFL_TextEncoder, imf, pt, ["bridge"] * len(idx), device, tokenize, mode="train")
                pro_img, amap_list = MyModel(te, imf, pt, stage=stage, mode="train")
                pro_img = pro_img.squeeze(2)
                ip = torch.randint(0, args.prompt_num, (1, 1))[0]
                pro_sel = torch.cat([pro_img[:, ip], pro_img[:, ip + args.prompt_num]], dim=1)
                l_cls = loss_cross(pro_sel, lab)
                l_dreg = l_reg[1] + l_reg[2] + l_reg[3]
                l_txt = loss_ort_fn(te, args)
                l_foc, l_dic = 0.0, 0.0
                for am in amap_list:
                    ams = torch.cat([am[:, ip], am[:, ip + args.prompt_num]], dim=1).softmax(dim=1)
                    l_foc = l_foc + loss_focal_fn(ams.clone(), gt.clone())
                    l_dic = l_dic + loss_dice_fn(ams, gt)
                loss = (l_cls + l_foc + l_dic + l_dreg + l_txt) if stage == 1 else l_cls
                if torch.isnan(loss) or torch.isinf(loss):
                    optimizer.zero_grad()
                    continue
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(MyModel.parameters(), max_norm=2.0)
                optimizer.step()
                if stage == 1:
                    scheduler.step()
                losses.append(float(loss.item()))

            MyModel.eval()
            @torch.no_grad()
            def val_infer(batch):
                MyModel.zk_n, MyModel.zk_a = None, None
                imgs = batch["img"].to(device)
                bsz = imgs.shape[0]
                with torch.cuda.amp.autocast():
                    imf_a, _, pt_a = model_clip.encode_image(imgs, args.features_list)
                imf_a, pt_a = imf_a.float(), [p.float() for p in pt_a]
                am_b, sp_b = [], []
                for b in range(bsz):
                    te, _ = MyModel.forward_ensemble(PFL_TextEncoder, imf_a[b:b+1], [p[b:b+1] for p in pt_a], ["bridge"], device, tokenize, mode="test")
                    pi, al = MyModel(te, imf_a[b:b+1], [p[b:b+1] for p in pt_a], stage=stage, mode="test")
                    pi = pi.squeeze(2)
                    tp = torch.stack([pi[:, :R_tot], pi[:, R_tot:]], dim=-1).softmax(dim=-1)
                    sp_b.append(float(tp[0, :, 1].mean().item()))
                    lm = [torch.stack([a[:, :R_tot], a[:, R_tot:]], dim=2).softmax(dim=2)[:, :, 1].mean(dim=1).float().cpu().numpy()[0] for a in al]
                    am_b.append(np.mean(lm, axis=0))
                return np.stack(am_b, 0), np.asarray(sp_b, dtype=np.float32)

            vm = collect_and_eval(va_loader, val_infer, sigma=8.0, sp_mode="bayespfl", compute_pro=False)
            vkey = (vm["Image-AUROC"] + vm["Pixel-AUROC"] + vm["Pixel-F1"] + vm["Pixel-AP"]) / 4.0
            print(f"[Bayes-PFL S2DS Ep {ep+1}/{epochs} Stg{stage}] loss={np.mean(losses):.4f} | Val Key={vkey:.2f} (P-AP={vm['Pixel-AP']:.2f})", flush=True)
            if vkey >= best_val:
                best_val = vkey
                torch.save({"MyModel": MyModel.state_dict(), "stage": stage}, best_ckpt)

            if stage == 1:
                if vm["Pixel-AP"] >= ap_max_px:
                    patience, ap_max_px = 0, vm["Pixel-AP"]
                    torch.save({"MyModel": MyModel.state_dict(), "stage": 1}, s1_ckpt)
                else:
                    patience += 1
                    if patience >= 3 and (ep + 1) >= 8:
                        if os.path.exists(s1_ckpt):
                            MyModel.load_state_dict(torch.load(s1_ckpt, map_location=device)["MyModel"])
                        stage, optimizer, patience = 2, opt2, 0
            else:
                if vm["Image-AP"] >= ap_max_im:
                    patience, ap_max_im = 0, vm["Image-AP"]
                else:
                    patience += 1
                    if patience >= 3 and (ep + 1) >= 12:
                        break

        ckpt = torch.load(best_ckpt, map_location=device)
        MyModel.load_state_dict(ckpt["MyModel"])
        best_stg = ckpt.get("stage", 1)
        MyModel.eval()

        @torch.no_grad()
        def test_infer(batch):
            setup_seed(333)
            MyModel.zk_n, MyModel.zk_a = None, None
            imgs = batch["img"].to(device)
            bsz = imgs.shape[0]
            with torch.cuda.amp.autocast():
                imf_a, _, pt_a = model_clip.encode_image(imgs, args.features_list)
            imf_a, pt_a = imf_a.float(), [p.float() for p in pt_a]
            am_b, sp_b = [], []
            for b in range(bsz):
                te, _ = MyModel.forward_ensemble(PFL_TextEncoder, imf_a[b:b+1], [p[b:b+1] for p in pt_a], ["bridge"], device, tokenize, mode="test")
                pi, al = MyModel(te, imf_a[b:b+1], [p[b:b+1] for p in pt_a], stage=best_stg, mode="test")
                pi = pi.squeeze(2)
                tp = torch.stack([pi[:, :R_tot], pi[:, R_tot:]], dim=-1).softmax(dim=-1)
                sp_b.append(float(tp[0, :, 1].mean().item()))
                lm = [torch.stack([a[:, :R_tot], a[:, R_tot:]], dim=2).softmax(dim=2)[:, :, 1].mean(dim=1).float().cpu().numpy()[0] for a in al]
                am_b.append(np.mean(lm, axis=0))
            return np.stack(am_b, 0), np.asarray(sp_b, dtype=np.float32)

        sigma, sp_mode, label, proto = 8.0, "bayespfl", "Bayes-PFL (CVPR 2025)", "Up to 30 Ep (2-Stg, early-stop)"

    elif method_name == "wpformer":
        WPFORMER_DIR = "/home/skye/data/Skye/WPFormer"
        isolate_external_repo(WPFORMER_DIR)
        from model.WPFormer import WPFormer

        setup_seed(111)
        epochs, batch_size, lr = 60, 6, 8e-5
        preprocess = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        s2_train = build_dataset("S2DS", OFFICIAL_S2DS_ROOT, split="train", image_size=518, preprocess=preprocess, s2ds_mode="official_full")
        s2_val = build_dataset("S2DS", OFFICIAL_S2DS_ROOT, split="val", image_size=518, preprocess=preprocess, s2ds_mode="official_full")
        s2_test = build_dataset("S2DS", OFFICIAL_S2DS_ROOT, split="test", image_size=518, preprocess=preprocess, s2ds_mode="official_full")
        cb_test = build_dataset("CODEBRIM", CODEBRIM_ROOT, split="test", image_size=518, preprocess=preprocess)

        tr_loader = torch.utils.data.DataLoader(s2_train, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=4, pin_memory=True)
        va_loader = torch.utils.data.DataLoader(s2_val, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

        model = WPFormer(method="pvt_v2_b2", channel=64).to(device)
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-7)
        scaler = torch.cuda.amp.GradScaler()
        weights = [0.5, 0.7, 1.0, 1.0]
        best_val = -1.0

        def wp_loss(p_logit, gts):
            bce = F.binary_cross_entropy_with_logits(p_logit, gts, reduction="mean")
            pred = torch.sigmoid(p_logit)
            inter = (pred * gts).sum(dim=(1, 2, 3))
            union = pred.sum(dim=(1, 2, 3)) + gts.sum(dim=(1, 2, 3))
            return bce + (1.0 - (2.0 * inter + 1e-6) / (union + 1e-6)).mean()

        @torch.no_grad()
        def test_infer(batch):
            imgs = batch["img"].to(device)
            with torch.cuda.amp.autocast():
                preds = model(imgs)[-1]
            probs = torch.sigmoid(preds.float())
            probs_up = F.interpolate(probs, size=(518, 518), mode="bilinear", align_corners=False)[:, 0]
            return probs_up.float().cpu().numpy(), None

        for ep in range(epochs):
            model.train()
            losses = []
            for b in tr_loader:
                imgs = b["img"].to(device, non_blocking=True)
                gts512 = F.interpolate((b["img_mask"].to(device, non_blocking=True) > 0.5).float(), size=(512, 512), mode="nearest")
                optimizer.zero_grad()
                with torch.cuda.amp.autocast():
                    preds = model(imgs)
                    loss = sum(weights[i] * wp_loss(preds[i], gts512) for i in range(4))
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                losses.append(float(loss.item()))
            scheduler.step()
            if (ep + 1 in [4, 8, 12, 16, 20, 25, 30, 35, 40, 45, 50, 55, 60]) or (ep + 1 == epochs):
                model.eval()
                vm = collect_and_eval(va_loader, test_infer, sigma=1.5, sp_mode="topk1", compute_pro=False)
                vkey = (vm["Image-AUROC"] + vm["Pixel-AUROC"] + vm["Pixel-F1"] + vm["Pixel-AP"]) / 4.0
                print(f"[WPFormer S2DS Ep {ep+1}/{epochs}] loss={np.mean(losses):.4f} | Val Key={vkey:.2f} (P-AP={vm['Pixel-AP']:.2f})", flush=True)
                if vkey >= best_val:
                    best_val = vkey
                    torch.save(model.state_dict(), best_ckpt)

        model.load_state_dict(torch.load(best_ckpt, map_location=device))
        model.eval()
        sigma, sp_mode, label, proto = 1.5, "topk1", "WPFormer (CVPR 2025)", "60 Ep (val-best)"

    elif method_name == "mixercseg":
        MIXERCSEG_DIR = "/home/skye/data/Skye/MixerCSeg"
        isolate_external_repo(MIXERCSEG_DIR)
        from models import build_MixerCSeg

        setup_seed(42)
        epochs, batch_size, lr = 50, 4, 5e-4
        preprocess = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        s2_train = build_dataset("S2DS", OFFICIAL_S2DS_ROOT, split="train", image_size=518, preprocess=preprocess, s2ds_mode="official_full")
        s2_val = build_dataset("S2DS", OFFICIAL_S2DS_ROOT, split="val", image_size=518, preprocess=preprocess, s2ds_mode="official_full")
        s2_test = build_dataset("S2DS", OFFICIAL_S2DS_ROOT, split="test", image_size=518, preprocess=preprocess, s2ds_mode="official_full")
        cb_test = build_dataset("CODEBRIM", CODEBRIM_ROOT, split="test", image_size=518, preprocess=preprocess)

        tr_loader = torch.utils.data.DataLoader(s2_train, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=4, pin_memory=True)
        va_loader = torch.utils.data.DataLoader(s2_val, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

        args_m = argparse.Namespace(BCELoss_ratio=0.87, DiceLoss_ratio=0.13, Norm_Type="GN", device="cuda", load_width=512, load_height=512)
        model, criterion = build_MixerCSeg(args_m)
        model.to(device)
        criterion.to(device)
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
        best_val = -1.0

        @torch.no_grad()
        def test_infer(batch):
            imgs = batch["img"].to(device)
            out = model(imgs)
            if isinstance(out, (list, tuple)):
                out = out[-1]
            probs = torch.sigmoid(out.float()) if (out.min() < 0.0 or out.max() > 1.0) else out.float()
            probs_up = F.interpolate(probs, size=(518, 518), mode="bilinear", align_corners=False)[:, 0]
            return probs_up.float().cpu().numpy(), None

        for ep in range(epochs):
            poly_lr = max(1e-6, lr * ((1.0 - float(ep) / float(max(epochs, 1))) ** 0.9))
            for pg in optimizer.param_groups:
                pg["lr"] = poly_lr
            model.train()
            criterion.train()
            losses = []
            for b in tr_loader:
                imgs = b["img"].to(device, non_blocking=True)
                gts512 = F.interpolate((b["img_mask"].to(device, non_blocking=True) > 0.5).float(), size=(512, 512), mode="nearest")
                optimizer.zero_grad()
                out = model(imgs)
                loss = criterion(out, gts512)
                loss.backward()
                optimizer.step()
                losses.append(float(loss.item()))
            if (ep + 1 in [4, 8, 12, 16, 20, 25, 30, 35, 40, 45, 50]) or (ep + 1 == epochs):
                model.eval()
                vm = collect_and_eval(va_loader, test_infer, sigma=1.5, sp_mode="topk1", compute_pro=False)
                vkey = (vm["Image-AUROC"] + vm["Pixel-AUROC"] + vm["Pixel-F1"] + vm["Pixel-AP"]) / 4.0
                print(f"[MixerCSeg S2DS Ep {ep+1}/{epochs}] loss={np.mean(losses):.4f} | Val Key={vkey:.2f} (P-AP={vm['Pixel-AP']:.2f})", flush=True)
                if vkey >= best_val:
                    best_val = vkey
                    torch.save(model.state_dict(), best_ckpt)

        model.load_state_dict(torch.load(best_ckpt, map_location=device))
        model.eval()
        sigma, sp_mode, label, proto = 1.5, "topk1", "MixerCSeg (CVPR 2026)", "50 Ep (val-best)"

    elif method_name == "dinomaly":
        DINOMALY_DIR = "/home/skye/data/Skye/Dinomaly"
        isolate_external_repo(DINOMALY_DIR)
        from dinomaly_bridge import build_model
        sys.modules.pop("utils", None); from utils import cal_anomaly_maps, global_cosine_hm_percent, WarmCosineScheduler
        from optimizers import StableAdamW

        setup_seed(1)
        preprocess = transforms.Compose([
            transforms.Resize((392, 392), interpolation=Image.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        s2_train_full = build_dataset("S2DS", OFFICIAL_S2DS_ROOT, split="train", image_size=518, preprocess=preprocess, s2ds_mode="official_full")
        # Dinomaly is normal-only unsupervised: filter normal images in Official S2DS train split
        normal_indices = [i for i, e in enumerate(s2_train_full.entries) if int(e["anomaly"]) == 0]
        s2_train_norm = torch.utils.data.Subset(s2_train_full, normal_indices)
        s2_val = build_dataset("S2DS", OFFICIAL_S2DS_ROOT, split="val", image_size=518, preprocess=preprocess, s2ds_mode="official_full")
        s2_test = build_dataset("S2DS", OFFICIAL_S2DS_ROOT, split="test", image_size=518, preprocess=preprocess, s2ds_mode="official_full")
        cb_test = build_dataset("CODEBRIM", CODEBRIM_ROOT, split="test", image_size=518, preprocess=preprocess)

        tr_loader = torch.utils.data.DataLoader(s2_train_norm, batch_size=16, shuffle=True, drop_last=True, num_workers=2)
        va_loader = torch.utils.data.DataLoader(s2_val, batch_size=8, shuffle=False, num_workers=2)

        model, trainable = build_model(device, variant="v0")
        trainable_params = sum(p.numel() for p in trainable.parameters())
        total_iters = 2000  # 97 normal images -> ~6 batches/ep, 2000 iters = 333 passes over normal images
        optimizer = StableAdamW([{"params": trainable.parameters()}], lr=2e-3, betas=(0.9, 0.999), weight_decay=1e-4, amsgrad=True, eps=1e-10)
        scheduler = WarmCosineScheduler(optimizer, base_value=2e-3, final_value=2e-4, total_iters=total_iters, warmup_iters=100)
        best_val = -1.0

        @torch.no_grad()
        def test_infer(batch):
            imgs = batch["img"].to(device)
            en, de = model(imgs)
            amap, _ = cal_anomaly_maps(en, de, 518)
            return amap[:, 0].float().cpu().numpy(), None

        it = 0
        while it < total_iters:
            model.train()
            for b in tr_loader:
                if it >= total_iters:
                    break
                imgs = b["img"].to(device)
                en, de = model(imgs)
                p_final = 0.9
                p = min(p_final * it / 1000, p_final)
                loss = global_cosine_hm_percent(en, de, p=p, factor=0.1)
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(trainable.parameters(), max_norm=0.1)
                optimizer.step()
                scheduler.step()
                it += 1
                if it % 400 == 0 or it == total_iters:
                    model.eval()
                    vm = collect_and_eval(va_loader, test_infer, sigma=4.0, sp_mode="topk1", compute_pro=False)
                    vkey = (vm["Image-AUROC"] + vm["Pixel-AUROC"] + vm["Pixel-F1"] + vm["Pixel-AP"]) / 4.0
                    print(f"[Dinomaly S2DS iter {it}/{total_iters}] loss={loss.item():.4f} | Val Key={vkey:.2f}", flush=True)
                    if vkey >= best_val:
                        best_val = vkey
                        torch.save({"model": model.state_dict()}, best_ckpt)

        model.load_state_dict(torch.load(best_ckpt, map_location=device)["model"])
        model.eval()
        sigma, sp_mode, label, proto = 4.0, "topk1", "Dinomaly (CVPR 2025)", "Normal-Only (val-best)"

    else:
        raise ValueError(f"Unknown method: {method_name}")

    s2_test_loader = torch.utils.data.DataLoader(s2_test, batch_size=16, shuffle=False, num_workers=4, pin_memory=True)
    cb_test_loader = torch.utils.data.DataLoader(cb_test, batch_size=16, shuffle=False, num_workers=4, pin_memory=True)

    print(f"\n>>> Evaluating {label} (trained on Official S2DS) ...")
    in_m = collect_and_eval(s2_test_loader, test_infer, sigma=sigma, sp_mode=sp_mode)
    print(f"[{label} In-Domain Official S2DS] Mean-7={in_m['Mean-Score']:.2f} | P-AUPRO={in_m['Pixel-AUPRO']:.2f} | Nuisance-FPR={in_m['Nuisance-FPR']:.2f}%", flush=True)

    cross_m = collect_and_eval(cb_test_loader, test_infer, sigma=sigma, sp_mode=sp_mode)
    print(f"[{label} Cross-Domain Official S2DS -> Bridge512] Mean-7={cross_m['Mean-Score']:.2f} | P-AUPRO={cross_m['Pixel-AUPRO']:.2f}", flush=True)

    rec = {
        "method": label,
        "variant": f"comp_{method_name}",
        "protocol": proto,
        "train_dataset": "S2DS",
        "trainable_params": trainable_params,
        "in_domain_test_metrics": in_m,
        "cross_domain_name": "Bridge512",
        "cross_domain_test_metrics": cross_m,
        "provenance": {
            "written_by": "train_ext_step2.py",
            "written_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "ckpt_file": best_ckpt,
            "ckpt_sha256": file_sha256(best_ckpt),
            "ckpt_bytes": os.path.getsize(best_ckpt),
        },
    }
    out_json = os.path.join(OUT_DIR, f"metrics_comp_{method_name}_S2DS.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(rec, f, indent=2)
    print(f"Saved: {out_json}", flush=True)


if __name__ == "__main__":
    train_and_eval_s2ds(sys.argv[1])
