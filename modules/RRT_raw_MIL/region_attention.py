import math

import torch
import torch.nn as nn


def region_partition(x: torch.Tensor, region_size: int) -> torch.Tensor:
    batch, height, width, channels = x.shape
    return (
        x.view(
            batch,
            height // region_size,
            region_size,
            width // region_size,
            region_size,
            channels,
        )
        .permute(0, 1, 3, 2, 4, 5)
        .contiguous()
        .view(-1, region_size, region_size, channels)
    )


def region_reverse(
    regions: torch.Tensor,
    region_size: int,
    height: int,
    width: int,
) -> torch.Tensor:
    batch = int(
        regions.shape[0]
        / (height * width / region_size / region_size)
    )
    return (
        regions.view(
            batch,
            height // region_size,
            width // region_size,
            region_size,
            region_size,
            -1,
        )
        .permute(0, 1, 3, 2, 4, 5)
        .contiguous()
        .view(batch, height, width, -1)
    )


def pad_to_regions(
    x: torch.Tensor,
    region_num: int,
) -> tuple[torch.Tensor, int, int, int, int]:
    batch, length, channels = x.shape
    side = math.ceil(math.sqrt(length))
    side += (-side) % region_num
    region_size = side // region_num
    add_length = side * side - length
    if add_length:
        padding = torch.zeros(
            batch,
            add_length,
            channels,
            device=x.device,
        )
        x = torch.cat([x, padding], dim=1)
    return x, side, side, add_length, region_size


class InnerAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        *,
        head_dim: int,
        num_heads: int,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        epeg: bool = True,
        epeg_k: int = 15,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = head_dim**-0.5

        self.qkv = nn.Linear(
            dim,
            head_dim * num_heads * 3,
            bias=qkv_bias,
        )
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(head_dim * num_heads, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.epeg_2d = False
        self.epeg_type = "attn"
        self.pe = (
            nn.Conv2d(
                num_heads,
                num_heads,
                (epeg_k, 1),
                padding=(epeg_k // 2, 0),
                groups=num_heads,
                bias=True,
            )
            if epeg
            else None
        )
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_regions, length, _channels = x.shape
        qkv = (
            self.qkv(x)
            .reshape(
                batch_regions,
                length,
                3,
                self.num_heads,
                self.head_dim,
            )
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = q * self.scale
        attention = q @ k.transpose(-2, -1)
        if self.pe is not None:
            attention = attention + self.pe(attention)
        attention = self.softmax(attention)
        attention = self.attn_drop(attention)
        x = (
            (attention @ v)
            .transpose(1, 2)
            .reshape(
                batch_regions,
                length,
                self.num_heads * self.head_dim,
            )
        )
        x = self.proj(x)
        return self.proj_drop(x)


class RegionAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        *,
        num_heads: int,
        region_num: int,
        proj_drop: float,
        epeg_k: int,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.region_size = None
        self.region_num = region_num
        self.min_region_num = 0
        self.min_region_ratio = 0.0
        self.attn = InnerAttention(
            dim,
            head_dim=dim // num_heads,
            num_heads=num_heads,
            qkv_bias=True,
            attn_drop=0.0,
            proj_drop=proj_drop,
            epeg=True,
            epeg_k=epeg_k,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, length, channels = x.shape
        x, height, width, add_length, region_size = pad_to_regions(
            x,
            self.region_num,
        )
        x = x.view(batch, height, width, channels)
        regions = region_partition(x, region_size)
        regions = regions.view(-1, region_size * region_size, channels)
        regions = self.attn(regions)
        regions = regions.view(-1, region_size, region_size, channels)
        x = region_reverse(regions, region_size, height, width)
        x = x.view(batch, height * width, channels)
        if add_length:
            x = x[:, :-add_length]
        return x


class CrossRegionAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        *,
        num_heads: int,
        region_num: int,
        proj_drop: float,
        crmsa_k: int,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.region_size = None
        self.region_num = region_num
        self.min_region_num = 0
        self.min_region_ratio = 0.0
        self.attn = InnerAttention(
            dim,
            head_dim=dim // num_heads,
            num_heads=num_heads,
            qkv_bias=True,
            attn_drop=0.0,
            proj_drop=proj_drop,
            epeg=False,
        )
        self.crmsa_mlp = False
        self.phi = nn.Parameter(torch.empty((dim, crmsa_k)))
        nn.init.kaiming_uniform_(self.phi, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, length, channels = x.shape
        x, height, width, add_length, region_size = pad_to_regions(
            x,
            self.region_num,
        )
        x = x.view(batch, height, width, channels)
        regions = region_partition(x, region_size)
        regions = regions.view(-1, region_size * region_size, channels)

        logits = torch.einsum(
            "w p c, c n -> w p n",
            regions,
            self.phi,
        ).transpose(1, 2)
        combine_weights = logits.softmax(dim=-1)
        dispatch_weights = logits.softmax(dim=1)
        logits_min, _ = logits.min(dim=-1)
        logits_max, _ = logits.max(dim=-1)
        dispatch_minmax = (logits - logits_min.unsqueeze(-1)) / (
            logits_max.unsqueeze(-1)
            - logits_min.unsqueeze(-1)
            + 1e-8
        )

        attended = torch.einsum(
            "w p c, w n p -> w n p c",
            regions,
            combine_weights,
        ).sum(dim=-2).transpose(0, 1)
        attended = self.attn(attended).transpose(0, 1)
        attended = torch.einsum(
            "w n c, w n p -> w n p c",
            attended,
            dispatch_minmax,
        )
        attended = torch.einsum(
            "w n p c, w n p -> w n p c",
            attended,
            dispatch_weights,
        ).sum(dim=1)

        attended = attended.view(
            -1,
            region_size,
            region_size,
            channels,
        )
        x = region_reverse(attended, region_size, height, width)
        x = x.view(batch, height * width, channels)
        if add_length:
            x = x[:, :-add_length]
        return x
