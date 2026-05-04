import numpy as np
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


class APFF(nn.Module):
    def __init__(
        self,
        dim,
        num_blocks=8,
        mode="AP",
        sparsity_threshold=0.01,
        hard_thresholding_fraction=1.0,
        hidden_size_factor=1,
        drop=0.0,
    ):
        super().__init__()
        if dim % num_blocks != 0:
            raise ValueError(f"hidden_dim={dim} must be divisible by num_blocks={num_blocks}")

        self.mode = mode
        self.hidden_size = dim
        self.sparsity_threshold = sparsity_threshold
        self.num_blocks = num_blocks
        self.block_size = self.hidden_size // self.num_blocks
        self.hard_thresholding_fraction = hard_thresholding_fraction
        self.hidden_size_factor = hidden_size_factor
        self.scale = 0.02

        hidden = self.block_size * self.hidden_size_factor
        self.w1 = nn.Parameter(self.scale * torch.randn(2, self.num_blocks, self.block_size, hidden))
        self.b1 = nn.Parameter(self.scale * torch.randn(2, self.num_blocks, hidden))
        self.w2 = nn.Parameter(self.scale * torch.randn(2, self.num_blocks, hidden, self.block_size))
        self.b2 = nn.Parameter(self.scale * torch.randn(2, self.num_blocks, self.block_size))
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        dtype = x.dtype
        x = x.float()
        b, n, c = x.shape

        x = torch.fft.rfft(x, dim=1, norm="ortho")
        x = x.reshape(b, n // 2 + 1, self.num_blocks, self.block_size)

        hidden = self.block_size * self.hidden_size_factor
        o1_real = torch.zeros(b, n // 2 + 1, self.num_blocks, hidden, device=x.device)
        o1_imag = torch.zeros_like(o1_real)
        o2_real = torch.zeros_like(x.real)
        o2_imag = torch.zeros_like(x.real)

        total_modes = n // 2 + 1
        kept_modes = max(1, int(total_modes * self.hard_thresholding_fraction))

        o1_real[:, :kept_modes] = F.relu(
            torch.einsum("...bi,bio->...bo", x[:, :kept_modes].real, self.w1[0])
            - torch.einsum("...bi,bio->...bo", x[:, :kept_modes].imag, self.w1[1])
            + self.b1[0]
        )
        o1_imag[:, :kept_modes] = F.relu(
            torch.einsum("...bi,bio->...bo", x[:, :kept_modes].imag, self.w1[0])
            + torch.einsum("...bi,bio->...bo", x[:, :kept_modes].real, self.w1[1])
            + self.b1[1]
        )

        o1_real = self.drop(o1_real)
        o1_imag = self.drop(o1_imag)

        o2_real[:, :kept_modes] = (
            torch.einsum("...bi,bio->...bo", o1_real[:, :kept_modes], self.w2[0])
            - torch.einsum("...bi,bio->...bo", o1_imag[:, :kept_modes], self.w2[1])
            + self.b2[0]
        )
        o2_imag[:, :kept_modes] = (
            torch.einsum("...bi,bio->...bo", o1_imag[:, :kept_modes], self.w2[0])
            + torch.einsum("...bi,bio->...bo", o1_real[:, :kept_modes], self.w2[1])
            + self.b2[1]
        )

        x = torch.stack([self.drop(o2_real), self.drop(o2_imag)], dim=-1)
        if self.mode == "LP":
            x = F.softshrink(x, lambd=self.sparsity_threshold)
        elif self.mode == "HP":
            x = x - F.softshrink(x, lambd=self.sparsity_threshold)

        x = torch.view_as_complex(x)
        x = x.reshape(b, n // 2 + 1, c)
        x = torch.fft.irfft(x, n=n, dim=1, norm="ortho")
        return x.type(dtype)


class Mlp(nn.Module):
    def __init__(self, dim, mlp_ratio=1.0, drop=0.0):
        super().__init__()
        hidden_dim = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.drop(self.act(self.fc1(x)))
        x = self.drop(self.fc2(x))
        return x


class FourierBlock(nn.Module):
    def __init__(
        self,
        dim=512,
        num_blocks=8,
        mode="AP",
        sparsity_threshold=0.01,
        hard_thresholding_fraction=1.0,
        hidden_size_factor=1,
        mlp_ratio=1.0,
        drop=0.0,
    ):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.attn = APFF(
            dim=dim,
            num_blocks=num_blocks,
            mode=mode,
            sparsity_threshold=sparsity_threshold,
            hard_thresholding_fraction=hard_thresholding_fraction,
            hidden_size_factor=hidden_size_factor,
            drop=drop,
        )
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(dim, mlp_ratio=mlp_ratio, drop=drop)

    def forward(self, x):
        x = x + self.attn(self.norm(x))
        x = x + self.mlp(self.norm2(x))
        return x


class FOURIERMIL(nn.Module):
    """FourierMIL: Fourier filtering-based MIL for WSI classification.

    This is a MIL_BASELINE-compatible adaptation of the ``ijcv2025-main``
    FourierMIL implementation. It keeps the original method principle:
    projected patch tokens are padded to a square length, a CLS token is prepended,
    and adaptive Fourier filtering (APFF) blocks model token interactions in the
    frequency domain before slide-level classification.

    Input: ``[B, N, in_dim]`` patch features.
    Output: ``{"logits": logits}`` with logits shaped ``[B, num_classes]``.
    ``WSI_feature`` is the FourierMIL CLS token. FourierMIL does not define an
    explicit patch attention distribution, so ``WSI_attn`` is the patch-to-CLS
    dot-product similarity used as an instance importance score for heatmaps.
    """

    def __init__(
        self,
        num_classes=2,
        in_dim=512,
        hidden_dim=512,
        num_layers=2,
        num_blocks=8,
        dropout=0.1,
        act=nn.ReLU(),
        mode="AP",
        sparsity_threshold=0.01,
        hard_thresholding_fraction=1.0,
        hidden_size_factor=1,
        mlp_ratio=1.0,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes

        self._fc1 = [nn.Linear(in_dim, hidden_dim), act]
        if dropout:
            self._fc1.append(nn.Dropout(dropout))
        self._fc1 = nn.Sequential(*self._fc1)

        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim))
        nn.init.normal_(self.cls_token, std=1e-6)
        self.layers = nn.ModuleList(
            [
                FourierBlock(
                    dim=hidden_dim,
                    num_blocks=num_blocks,
                    mode=mode,
                    sparsity_threshold=sparsity_threshold,
                    hard_thresholding_fraction=hard_thresholding_fraction,
                    hidden_size_factor=hidden_size_factor,
                    mlp_ratio=mlp_ratio,
                    drop=dropout,
                )
                for _ in range(num_layers)
            ]
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self._fc2 = nn.Linear(hidden_dim, num_classes)
        self.apply(initialize_weights)

    def forward(self, x, return_WSI_attn=False, return_WSI_feature=False):
        forward_return = {}
        if x.dim() == 2:
            x = x.unsqueeze(0)
        n = x.shape[1]

        h = self._fc1(x.float())
        side = int(np.ceil(np.sqrt(n)))
        add_length = side * side - n
        if add_length > 0:
            h = torch.cat([h, h[:, :add_length, :]], dim=1)

        cls_tokens = self.cls_token.expand(h.shape[0], -1, -1).to(h.device)
        h = torch.cat((cls_tokens, h), dim=1)
        for layer in self.layers:
            h = layer(h)

        h = self.norm(h)
        cls_token = h[:, 0]
        patch_tokens = h[:, 1 : n + 1]
        logits = self._fc2(cls_token)

        forward_return["logits"] = logits
        if return_WSI_feature:
            forward_return["WSI_feature"] = cls_token
        if return_WSI_attn:
            attn = torch.einsum("bnc,bc->bn", patch_tokens, cls_token)
            forward_return["WSI_attn"] = attn.squeeze(0).unsqueeze(-1)
        return forward_return

