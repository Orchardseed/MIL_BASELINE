"""Official-default R2T-MIL adapted to the local MIL dictionary contract.

Paper: Feature Re-Embedding: Towards Foundation Model-Level Performance
       in Computational Pathology (CVPR 2024)
Reference: https://github.com/DearCaat/RRT-MIL
Reference commit: 3320b6616b2c3d96ddc5bcd1aecea865a064987f
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .region_attention import CrossRegionAttention, RegionAttention


def initialize_weights(module: nn.Module) -> None:
    for child in module.modules():
        if isinstance(child, nn.Conv2d):
            nn.init.xavier_normal_(child.weight)
            if child.bias is not None:
                child.bias.data.zero_()
        elif isinstance(child, nn.Linear):
            nn.init.xavier_normal_(child.weight)
            if child.bias is not None:
                child.bias.data.zero_()
        elif isinstance(child, nn.LayerNorm):
            nn.init.constant_(child.bias, 0)
            nn.init.constant_(child.weight, 1.0)


class TransLayer(nn.Module):
    def __init__(
        self,
        *,
        dim: int,
        num_heads: int,
        trans_dropout: float,
        region_num: int,
        attention_kind: str,
        epeg_k: int,
        crmsa_k: int,
    ) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        if attention_kind == "rmsa":
            self.attn = RegionAttention(
                dim,
                num_heads=num_heads,
                region_num=region_num,
                proj_drop=trans_dropout,
                epeg_k=epeg_k,
            )
        elif attention_kind == "crmsa":
            self.attn = CrossRegionAttention(
                dim,
                num_heads=num_heads,
                region_num=region_num,
                proj_drop=trans_dropout,
                crmsa_k=crmsa_k,
            )
        else:
            raise ValueError(f"unsupported attention kind: {attention_kind}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.attn(self.norm(x))


class RRTEncoder(nn.Module):
    def __init__(
        self,
        *,
        mlp_dim: int,
        region_num: int,
        n_layers: int,
        n_heads: int,
        trans_dropout: float,
        epeg_k: int,
        crmsa_k: int,
        crmsa_heads: int,
    ) -> None:
        super().__init__()
        self.final_dim = mlp_dim
        self.norm = nn.LayerNorm(self.final_dim)
        self.all_shortcut = False
        self.layers = nn.Sequential(
            *[
                TransLayer(
                    dim=mlp_dim,
                    num_heads=n_heads,
                    trans_dropout=trans_dropout,
                    region_num=region_num,
                    attention_kind="rmsa",
                    epeg_k=epeg_k,
                    crmsa_k=crmsa_k,
                )
                for _ in range(n_layers - 1)
            ]
        )
        self.cr_msa = TransLayer(
            dim=mlp_dim,
            num_heads=crmsa_heads,
            trans_dropout=trans_dropout,
            region_num=region_num,
            attention_kind="crmsa",
            epeg_k=epeg_k,
            crmsa_k=crmsa_k,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers.children():
            x = layer(x)
        x = self.cr_msa(x)
        return self.norm(x)


class Attention(nn.Module):
    def __init__(self, input_dim: int = 512) -> None:
        super().__init__()
        self.L = input_dim
        self.D = 128
        self.K = 1
        self.attention = nn.Sequential(
            nn.Linear(self.L, self.D, bias=False),
            nn.ReLU(),
            nn.Linear(self.D, self.K, bias=False),
        )

    def forward(
        self,
        x: torch.Tensor,
        no_norm: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        attention = self.attention(x)
        attention = torch.transpose(attention, -1, -2)
        raw_attention = attention.clone()
        attention = F.softmax(attention, dim=-1)
        pooled = torch.matmul(attention, x)
        return pooled, raw_attention if no_norm else attention


class DAttention(nn.Module):
    def __init__(self, input_dim: int = 512) -> None:
        super().__init__()
        self.gated = False
        self.attention = Attention(input_dim)

    def forward(
        self,
        x: torch.Tensor,
        return_attn: bool = False,
        no_norm: bool = False,
        **_kwargs,
    ):
        pooled, attention = self.attention(x, no_norm)
        pooled = pooled.squeeze(1)
        attention = attention.squeeze(1)
        if return_attn:
            return pooled, attention
        return pooled


class RRT_raw_MIL(nn.Module):
    def __init__(
        self,
        in_dim: int,
        num_classes: int,
        mlp_dim: int = 512,
        dropout: float = 0.25,
        region_num: int = 8,
        n_layers: int = 2,
        n_heads: int = 8,
        trans_dropout: float = 0.1,
        epeg_k: int = 15,
        crmsa_k: int = 3,
        crmsa_heads: int = 8,
    ) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.patch_to_emb = nn.Sequential(
            nn.Linear(in_dim, mlp_dim),
            nn.ReLU(),
        )
        self.dp = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.online_encoder = RRTEncoder(
            mlp_dim=mlp_dim,
            region_num=region_num,
            n_layers=n_layers,
            n_heads=n_heads,
            trans_dropout=trans_dropout,
            epeg_k=epeg_k,
            crmsa_k=crmsa_k,
            crmsa_heads=crmsa_heads,
        )
        self.pool_fn = DAttention(mlp_dim)
        self.predictor = nn.Linear(mlp_dim, num_classes)
        self.apply(initialize_weights)

    def _normalize_input(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim not in (2, 3):
            raise ValueError(
                f"RRT_raw_MIL input must have rank 2 or 3, got {x.ndim}"
            )
        if x.ndim == 2:
            x = x.unsqueeze(0)
        if x.shape[0] != 1:
            raise ValueError(
                "RRT_raw_MIL requires batch size 1, "
                f"got {x.shape[0]}"
            )
        if x.shape[1] == 0:
            raise ValueError("RRT_raw_MIL requires at least one instance")
        if x.shape[2] != self.in_dim:
            raise ValueError(
                "RRT_raw_MIL feature dimension mismatch: "
                f"expected {self.in_dim}, got {x.shape[2]}"
            )
        return x

    def forward(
        self,
        x: torch.Tensor,
        return_WSI_attn: bool = False,
        return_WSI_feature: bool = False,
    ) -> dict[str, torch.Tensor]:
        x = self._normalize_input(x)
        feature = self.dp(self.patch_to_emb(x))
        feature = self.online_encoder(feature)
        pooled, raw_attention = self.pool_fn(
            feature,
            return_attn=True,
            no_norm=True,
        )
        logits = self.predictor(pooled)
        result = {"logits": logits}
        if return_WSI_feature:
            result["WSI_feature"] = pooled.squeeze(0)
        if return_WSI_attn:
            result["WSI_attn"] = raw_attention.squeeze(0)
        return result
