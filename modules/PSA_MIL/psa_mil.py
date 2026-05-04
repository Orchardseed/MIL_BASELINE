import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def initialize_weights(module):
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


def compute_theta_from_local_k(decay_type, local_k, decay_clip):
    decay_clip = torch.tensor(decay_clip, dtype=torch.float32)
    local_k = torch.tensor(local_k, dtype=torch.float32)
    if decay_type == "Gaussian":
        return torch.sqrt(local_k**2 / (-2 * torch.log(decay_clip))).item()
    if decay_type == "Exponential":
        return (-torch.log(decay_clip) / local_k).item()
    if decay_type == "InverseQuadratic":
        return (local_k / torch.sqrt((1 - decay_clip) / decay_clip)).item()
    raise ValueError("decay_type must be Gaussian, Exponential, or InverseQuadratic")


def solve_for_local_k(decay_type, param, decay_clip):
    decay_clip = torch.tensor(decay_clip, dtype=param.dtype, device=param.device)
    if decay_type == "Gaussian":
        return torch.sqrt(-torch.log(decay_clip) * 2 * (param**2))
    if decay_type == "Exponential":
        return -torch.log(decay_clip) / param
    if decay_type == "InverseQuadratic":
        return torch.sqrt(((1 - decay_clip) / decay_clip) * (param**2))
    raise ValueError("decay_type must be Gaussian, Exponential, or InverseQuadratic")


class DecayNetwork(nn.Module):
    def __init__(self, decay_type="Gaussian", num_heads=8, decay_clip=0.001, min_local_k=1, max_local_k=25, init_k=7):
        super().__init__()
        self.decay_type = decay_type
        self.decay_clip = decay_clip
        self.min_local_k = min_local_k
        self.max_local_k = max_local_k

        theta1 = compute_theta_from_local_k(decay_type, min_local_k, decay_clip)
        theta2 = compute_theta_from_local_k(decay_type, max_local_k, decay_clip)
        self.theta_min = min(theta1, theta2)
        self.theta_max = max(theta1, theta2)
        self.lambda_p = nn.Parameter(self.init_around_k(init_k, num_heads))

    def init_around_k(self, target_k, num_heads):
        low = compute_theta_from_local_k(self.decay_type, target_k - 1, self.decay_clip)
        high = compute_theta_from_local_k(self.decay_type, target_k + 1, self.decay_clip)
        theta_min = min(low, high)
        theta_max = max(low, high)
        theta_range = theta_max - theta_min
        theta_min += theta_range * 0.25
        theta_max -= theta_range * 0.25
        lambda_min = (theta_min - self.theta_min) / (self.theta_max - self.theta_min)
        lambda_max = (theta_max - self.theta_min) / (self.theta_max - self.theta_min)
        return torch.linspace(lambda_min, lambda_max, num_heads)

    def reparam(self, lambda_p):
        lambda_p = torch.clamp(lambda_p, min=0, max=1)
        return self.theta_min + lambda_p * (self.theta_max - self.theta_min)

    def get_params(self, param_str):
        if param_str == "rates":
            return torch.clamp(self.lambda_p, min=0, max=1)
        if param_str == "thetas":
            return self.reparam(self.lambda_p)
        if param_str == "local_Ks":
            return torch.round(solve_for_local_k(self.decay_type, self.reparam(self.lambda_p), self.decay_clip))
        raise ValueError(f"Unknown param_str: {param_str}")

    def forward(self, distance, head_ind):
        theta = self.reparam(self.lambda_p)[head_ind]
        if self.decay_type == "Gaussian":
            return torch.exp(-(distance**2) / (2 * theta**2))
        if self.decay_type == "Exponential":
            return torch.exp(-theta * distance)
        if self.decay_type == "InverseQuadratic":
            return 1 / (1 + (distance / theta) ** 2)
        raise ValueError(f"Unknown decay_type: {self.decay_type}")


class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(dim, dim, bias=False),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return x + self.block(x)


class ResidualFullyConnected(nn.Module):
    def __init__(self, in_dim, out_dim, num_layers):
        super().__init__()
        layers = [nn.Linear(in_dim, out_dim, bias=False), nn.ReLU(inplace=True)]
        for _ in range(max(0, num_layers - 1)):
            layers.append(ResidualBlock(out_dim))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class SpatialAttention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=True,
        attn_drop=0.0,
        proj_drop=0.0,
        decay_type="Gaussian",
        decay_clip=0.001,
        min_local_k=1,
        max_local_k=25,
        init_k=7,
    ):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim={dim} must be divisible by num_heads={num_heads}")
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.sigma_sq = math.sqrt(self.head_dim)
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = nn.Identity()
        self.k_norm = nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.decay_nn = DecayNetwork(decay_type, num_heads, decay_clip, min_local_k, max_local_k, init_k)
        self.last_attn = None

    def forward(self, x, distance, indices):
        b, n, c = x.shape
        qkv = self.qkv(x).reshape(b, n, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q = self.q_norm(q)
        k = self.k_norm(k)

        out = torch.zeros_like(q)
        q_norm_sq = torch.sum(q**2, dim=-1, keepdim=True)
        k_norm_sq = torch.sum(k**2, dim=-1)
        local_k = solve_for_local_k(self.decay_nn.decay_type, self.decay_nn.reparam(self.decay_nn.lambda_p), self.decay_nn.decay_clip)
        num_elements = torch.round(math.pi * local_k**2).long().clamp(min=1, max=n)
        batch_idx = torch.arange(b, device=x.device).view(b, 1, 1)
        cls_patch_attn = []

        for head_idx in range(self.num_heads):
            k_count = int(num_elements[head_idx].item())
            selected_idx = indices[:, :, :k_count].to(x.device).long()
            k_h = k[:, head_idx]
            q_h = q[:, head_idx]
            v_h = v[:, head_idx]
            k_selected = k_h[batch_idx, selected_idx]
            v_selected = v_h[batch_idx, selected_idx]

            attn = torch.einsum("bnd,bnkd->bnk", q_h, k_selected) / self.sigma_sq
            selected_k_norm_sq = k_norm_sq[:, head_idx][batch_idx, selected_idx]
            attn = attn - 0.5 * (q_norm_sq[:, head_idx] + selected_k_norm_sq) / self.sigma_sq
            distance_k = distance.gather(dim=-1, index=selected_idx)
            decay = self.decay_nn(distance_k, head_idx) + 1e-6
            attn = attn + decay.log()
            attn = F.softmax(attn, dim=-1)
            attn = self.attn_drop(attn)
            out[:, head_idx] = torch.einsum("bnk,bnkd->bnd", attn.float(), v_selected)

            if n > 1:
                full_cls = torch.zeros(b, n, device=x.device, dtype=attn.dtype)
                full_cls.scatter_add_(1, selected_idx[:, 0, :], attn[:, 0, :])
                cls_patch_attn.append(full_cls[:, 1:])

        if cls_patch_attn:
            self.last_attn = torch.stack(cls_patch_attn, dim=1).mean(dim=1).detach()
        else:
            self.last_attn = None
        out = out.transpose(1, 2).reshape(b, n, c)
        return self.proj_drop(self.proj(out))


class SpatialBlock(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        mlp_ratio=4.0,
        qkv_bias=True,
        proj_drop=0.0,
        attn_drop=0.0,
        decay_type="Gaussian",
        decay_clip=0.001,
        min_local_k=1,
        max_local_k=25,
        init_k=7,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = SpatialAttention(
            dim=dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            decay_type=decay_type,
            decay_clip=decay_clip,
            min_local_k=min_local_k,
            max_local_k=max_local_k,
            init_k=init_k,
        )
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(proj_drop),
            nn.Linear(int(dim * mlp_ratio), dim),
            nn.Dropout(proj_drop),
        )

    def forward(self, x, distance, indices):
        x = x + self.attn(self.norm1(x), distance, indices)
        x = x + self.mlp(self.norm2(x))
        return x


class PSA_MIL(nn.Module):
    """PSA-MIL: Probabilistic Spatial Attention MIL.

    This module ports the PSA-MIL core idea into MIL_BASELINE: features are first
    adapted by residual fully-connected layers, then processed by spatial
    multi-head attention whose posterior combines feature similarity with a
    learnable distance-decay prior and dynamic local pruning. The input can be
    ``[B, N, in_dim]`` or ``[B, N, in_dim + 2]`` with appended ``(x, y)`` coords.
    If coords are absent, a pseudo-grid is generated for compatibility, but real
    h5/pt coords are recommended for faithfully using the spatial prior.

    ``WSI_feature`` is the pooled slide representation. ``WSI_attn`` is the
    attention pooling weight when ``pool_type=attention``; otherwise it is the
    final CLS-to-patch local posterior score.
    """

    def __init__(
        self,
        num_classes=2,
        in_dim=512,
        attn_dim=192,
        num_heads=3,
        depth=1,
        num_layers_adapter=2,
        patch_drop_rate=0.0,
        qkv_bias=True,
        pool_type="attention",
        mlp_ratio=4.0,
        dropout=0.1,
        decay_type="Gaussian",
        decay_clip=0.001,
        min_local_k=1,
        max_local_k=25,
        init_k=7,
        act=None,
    ):
        super().__init__()
        del act
        self.in_dim = in_dim
        self.pool_type = pool_type
        self.patch_drop_rate = patch_drop_rate
        hidden_dim = attn_dim * num_heads
        self.adapter = ResidualFullyConnected(in_dim, hidden_dim, num_layers_adapter)
        if pool_type == "cls_token":
            self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
            nn.init.normal_(self.cls_token, std=1e-6)
        elif pool_type == "attention":
            self.attention_pool = nn.Linear(hidden_dim, 1)
        elif pool_type != "avg":
            raise ValueError("pool_type must be one of: attention, cls_token, avg")

        self.blocks = nn.ModuleList(
            [
                SpatialBlock(
                    dim=hidden_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    proj_drop=dropout,
                    attn_drop=dropout,
                    decay_type=decay_type,
                    decay_clip=decay_clip,
                    min_local_k=min_local_k,
                    max_local_k=max_local_k,
                    init_k=init_k,
                )
                for _ in range(depth)
            ]
        )
        self.layer_norm = nn.LayerNorm(hidden_dim, eps=1e-6)
        self.head_drop = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_dim, num_classes)
        self.apply(initialize_weights)

    def _pseudo_coords(self, n, device, dtype):
        side = int(math.ceil(math.sqrt(n)))
        y = torch.arange(side, device=device, dtype=dtype).repeat_interleave(side)
        x = torch.arange(side, device=device, dtype=dtype).repeat(side)
        return torch.stack([x, y], dim=-1)[:n]

    def _split_features_coords(self, x, coords=None):
        if x.dim() == 2:
            x = x.unsqueeze(0)
        if coords is not None:
            if coords.dim() == 2:
                coords = coords.unsqueeze(0)
            return x[..., : self.in_dim], coords.float()
        if x.shape[-1] == self.in_dim + 2:
            return x[..., : self.in_dim], x[..., self.in_dim : self.in_dim + 2].float()
        coords = self._pseudo_coords(x.shape[1], x.device, x.dtype).unsqueeze(0).expand(x.shape[0], -1, -1)
        return x[..., : self.in_dim], coords

    def _build_distance_inputs(self, coords):
        distance = torch.cdist(coords.float(), coords.float(), p=2)
        positive = distance[distance > 0]
        if positive.numel() > 0:
            distance = distance / (positive.mean().detach() + 1e-6)
        indices = torch.argsort(distance, dim=-1)
        return distance, indices

    def forward(self, x, coords=None, return_WSI_attn=False, return_WSI_feature=False):
        forward_return = {}
        x, coords = self._split_features_coords(x, coords)
        h = self.adapter(x.float())

        if self.training and self.patch_drop_rate > 0 and h.shape[1] > 1:
            keep = max(1, int((1 - self.patch_drop_rate) * h.shape[1]))
            patch_idx = torch.randperm(h.shape[1], device=h.device)[:keep]
            h = h[:, patch_idx]
            coords = coords[:, patch_idx]

        if self.pool_type == "cls_token":
            cls_tokens = self.cls_token.expand(h.shape[0], -1, -1).to(h.device)
            h = torch.cat([cls_tokens, h], dim=1)
            cls_coords = coords.mean(dim=1, keepdim=True)
            coords = torch.cat([cls_coords, coords], dim=1)

        distance, indices = self._build_distance_inputs(coords)
        for block in self.blocks:
            h = block(h, distance, indices)
        h = self.layer_norm(h)

        if self.pool_type == "cls_token":
            slide_feature = h[:, 0]
        elif self.pool_type == "attention":
            attn_weights = torch.softmax(self.attention_pool(h).squeeze(-1), dim=-1)
            slide_feature = torch.sum(h * attn_weights.unsqueeze(-1), dim=1)
        else:
            slide_feature = h.mean(dim=1)

        logits = self.head(self.head_drop(slide_feature))
        forward_return["logits"] = logits
        if return_WSI_feature:
            forward_return["WSI_feature"] = slide_feature
        if return_WSI_attn:
            if self.pool_type == "attention":
                attn = attn_weights
            elif self.blocks and self.blocks[-1].attn.last_attn is not None:
                attn = self.blocks[-1].attn.last_attn
            else:
                patch_tokens = h[:, 1:] if self.pool_type == "cls_token" else h
                attn = torch.einsum("bnc,bc->bn", patch_tokens, slide_feature)
            forward_return["WSI_attn"] = attn.squeeze(0).unsqueeze(-1)
        return forward_return

