import hashlib
import os
import urllib.request
import warnings
from collections import OrderedDict
from typing import List, Tuple, Union

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from tqdm import tqdm

from .simple_tokenizer import SimpleTokenizer as _Tokenizer

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

OPENAI_DATASET_MEAN = (0.48145466, 0.4578275, 0.40821073)
OPENAI_DATASET_STD = (0.26862954, 0.26130258, 0.27577711)

VIT_L_14_336_URL = (
    "https://openaipublic.azureedge.net/clip/models/"
    "3035c92b350959924f9f00213499208652fc7ea050643e8b385c2dac08641f02/ViT-L-14-336px.pt"
)

_tokenizer = _Tokenizer()


def tokenize(texts: Union[str, List[str]], context_length: int = 77, truncate: bool = True) -> torch.LongTensor:
    if isinstance(texts, str):
        texts = [texts]
    sot_token = _tokenizer.encoder["<|startoftext|>"]
    eot_token = _tokenizer.encoder["<|endoftext|>"]
    all_tokens = [[sot_token] + _tokenizer.encode(text) + [eot_token] for text in texts]
    result = torch.zeros(len(all_tokens), context_length, dtype=torch.long)
    for i, tokens in enumerate(all_tokens):
        if len(tokens) > context_length:
            if truncate:
                tokens = tokens[:context_length]
                tokens[-1] = eot_token
            else:
                raise RuntimeError(f"Input {texts[i]} is too long for context length {context_length}")
        result[i, :len(tokens)] = torch.tensor(tokens)
    return result


def get_clip_transform(image_size: int = 518):
    return Compose([
        Resize((image_size, image_size), interpolation=BICUBIC),
        CenterCrop((image_size, image_size)),
        lambda img: img.convert("RGB"),
        ToTensor(),
        Normalize(OPENAI_DATASET_MEAN, OPENAI_DATASET_STD),
    ])


class LayerNorm(nn.LayerNorm):
    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model)),
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=attn_mask)[0]

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

    def forward(self, x: torch.Tensor):
        return self.resblocks(x)


class VisionTransformer(nn.Module):
    def __init__(self, input_resolution: int, patch_size: int, width: int, layers: int, heads: int, output_dim: int):
        super().__init__()
        self.input_resolution = input_resolution
        self.patch_size = patch_size
        self.width = width
        self.layers = layers
        self.output_dim = output_dim

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)
        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width))
        self.ln_pre = LayerNorm(width)
        self.transformer = Transformer(width, layers, heads)
        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

    def _interpolate_pos_embed(self, new_side: int, dtype: torch.dtype) -> torch.Tensor:
        pos_emb = self.positional_embedding.to(dtype)
        side = int(round((pos_emb.shape[0] - 1) ** 0.5))
        if side == new_side:
            return pos_emb
        cls_pos = pos_emb[:1, :]
        patch_pos = pos_emb[1:, :].reshape(1, side, side, -1).permute(0, 3, 1, 2).float()
        patch_pos = F.interpolate(patch_pos, size=(new_side, new_side), mode="bilinear", align_corners=False)
        patch_pos = patch_pos.permute(0, 2, 3, 1).reshape(new_side * new_side, -1).to(dtype)
        return torch.cat([cls_pos, patch_pos], dim=0)

    def forward(self, x: torch.Tensor, features_list: List[int] = (6, 12, 18, 24)):
        x = self.conv1(x)  # [B, width, H_p, W_p]
        B, C, H_p, W_p = x.shape
        x = x.reshape(B, C, H_p * W_p).permute(0, 2, 1)  # [B, L, width]
        cls_tok = self.class_embedding.to(x.dtype).unsqueeze(0).unsqueeze(0).expand(B, 1, -1)
        x = torch.cat([cls_tok, x], dim=1)  # [B, 1 + L, width]

        pos_emb = self._interpolate_pos_embed(H_p, x.dtype)
        x = x + pos_emb.unsqueeze(0)
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # [1 + L, B, width]
        patch_Tokens_layers = []
        feat_set = set(features_list)
        for idx, block in enumerate(self.transformer.resblocks):
            x = block(x)
            layer_num = idx + 1
            if layer_num in feat_set:
                x_nld = x.permute(1, 0, 2)  # [B, 1 + L, width]
                # Post-LN on patch tokens preserves layer scale stability for CLIP alignment
                patch_tokens = self.ln_post(x_nld[:, 1:, :])  # [B, L, width=1024]
                patch_Tokens_layers.append(patch_tokens)

        x_final = self.ln_post(x.permute(1, 0, 2))
        cls_out = x_final[:, 0, :]
        if self.proj is not None:
            cls_out = cls_out @ self.proj  # [B, 768]
        return cls_out, patch_Tokens_layers, (H_p, W_p)


class CLIPBackbone(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        image_resolution: int,
        vision_layers: int,
        vision_width: int,
        vision_patch_size: int,
        context_length: int,
        vocab_size: int,
        transformer_width: int,
        transformer_heads: int,
        transformer_layers: int,
    ):
        super().__init__()
        self.context_length = context_length
        self.embed_dim = embed_dim
        self.vision_width = vision_width

        vision_heads = vision_width // 64
        self.visual = VisionTransformer(
            input_resolution=image_resolution,
            patch_size=vision_patch_size,
            width=vision_width,
            layers=vision_layers,
            heads=vision_heads,
            output_dim=embed_dim,
        )

        self.transformer = Transformer(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            attn_mask=self.build_attention_mask(),
        )
        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, transformer_width))
        self.ln_final = LayerNorm(transformer_width)
        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def build_attention_mask(self):
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)
        return mask

    @property
    def dtype(self):
        return self.visual.conv1.weight.dtype

    def encode_image(self, image: torch.Tensor, features_list: List[int] = (6, 12, 18, 24)):
        return self.visual(image.type(self.dtype), features_list=features_list)

    def encode_text(self, text: torch.Tensor) -> torch.Tensor:
        x = self.token_embedding(text).type(self.dtype)
        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)
        x = self.ln_final(x).type(self.dtype)
        x = x[torch.arange(x.shape[0], device=x.device), text.argmax(dim=-1)] @ self.text_projection
        return x


def _download_clip_checkpoint(url: str, cache_dir: str) -> str:
    os.makedirs(cache_dir, exist_ok=True)
    filename = os.path.basename(url)
    expected_sha256 = url.split("/")[-2]
    target = os.path.join(cache_dir, filename)
    if os.path.isfile(target):
        if hashlib.sha256(open(target, "rb").read()).hexdigest().startswith(expected_sha256):
            return target
    print(f"Downloading CLIP ViT-L/14@336px checkpoint to {target} ...")
    with urllib.request.urlopen(url) as source, open(target, "wb") as output:
        total = int(source.headers.get("Content-Length", 0))
        with tqdm(total=total, ncols=80, unit="iB", unit_scale=True) as loop:
            while True:
                buf = source.read(65536)
                if not buf:
                    break
                output.write(buf)
                loop.update(len(buf))
    return target


def load_clip_vit_l14_336(checkpoint_path: str = None, device: Union[str, torch.device] = "cuda") -> CLIPBackbone:
    candidate_paths = []
    if checkpoint_path:
        candidate_paths.append(checkpoint_path)
    candidate_paths.extend([
        "/home/skye/data/Skye/Bayes-PFL-main/pretrained_weight/ViT-L-14-336px.pt",
        os.path.expanduser("~/.cache/clip/ViT-L-14-336px.pt"),
        "/content/cache/ViT-L-14-336px.pt",
    ])
    resolved_path = None
    for p in candidate_paths:
        if p and os.path.isfile(p):
            resolved_path = p
            break
    if resolved_path is None:
        resolved_path = _download_clip_checkpoint(VIT_L_14_336_URL, os.path.expanduser("~/.cache/clip"))

    try:
        jit_model = torch.jit.load(resolved_path, map_location="cpu").eval()
        state_dict = jit_model.state_dict()
    except RuntimeError:
        state_dict = torch.load(resolved_path, map_location="cpu")

    vision_width = state_dict["visual.conv1.weight"].shape[0]
    vision_layers = len([k for k in state_dict.keys() if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])
    vision_patch_size = state_dict["visual.conv1.weight"].shape[-1]
    grid_size = round((state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
    image_resolution = vision_patch_size * grid_size

    embed_dim = state_dict["text_projection"].shape[1]
    context_length = state_dict["positional_embedding"].shape[0]
    vocab_size = state_dict["token_embedding.weight"].shape[0]
    transformer_width = state_dict["ln_final.weight"].shape[0]
    transformer_heads = transformer_width // 64
    transformer_layers = len(set(k.split(".")[2] for k in state_dict if k.startswith("transformer.resblocks")))

    model = CLIPBackbone(
        embed_dim=embed_dim,
        image_resolution=image_resolution,
        vision_layers=vision_layers,
        vision_width=vision_width,
        vision_patch_size=vision_patch_size,
        context_length=context_length,
        vocab_size=vocab_size,
        transformer_width=transformer_width,
        transformer_heads=transformer_heads,
        transformer_layers=transformer_layers,
    )
    for key in ["input_resolution", "context_length", "vocab_size"]:
        if key in state_dict:
            del state_dict[key]
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model.to(device)
