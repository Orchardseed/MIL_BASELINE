import numpy as np
import torch
import torch.nn as nn

from .nystrom_attention import NystromAttention


def initialize_weights(module):
    for m in module.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


class TransLayer(nn.Module):
    def __init__(
        self,
        dim=512,
        num_heads=8,
        num_landmarks=None,
        pinv_iterations=6,
        dropout=0.1,
        residual=True,
    ):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.attn = NystromAttention(
            dim=dim,
            dim_head=dim // num_heads,
            heads=num_heads,
            num_landmarks=num_landmarks or dim // 2,
            pinv_iterations=pinv_iterations,
            residual=residual,
            dropout=dropout,
        )

    def forward(self, x):
        return x + self.attn(self.norm(x))


class PPEG(nn.Module):
    def __init__(self, dim=512):
        super().__init__()
        self.proj = nn.Conv2d(dim, dim, 7, 1, 7 // 2, groups=dim)
        self.proj1 = nn.Conv2d(dim, dim, 5, 1, 5 // 2, groups=dim)
        self.proj2 = nn.Conv2d(dim, dim, 3, 1, 3 // 2, groups=dim)

    def forward(self, x, h, w):
        b, _, c = x.shape
        cls_token, feat_token = x[:, 0], x[:, 1:]
        cnn_feat = feat_token.transpose(1, 2).view(b, c, h, w)
        x = self.proj(cnn_feat) + cnn_feat + self.proj1(cnn_feat) + self.proj2(cnn_feat)
        x = x.flatten(2).transpose(1, 2)
        return torch.cat((cls_token.unsqueeze(1), x), dim=1)


class TRANS_MIL(nn.Module):
    """TransMIL: Transformer based correlated MIL for WSI classification.

    This is a complete rewrite of the local baseline module using the original
    TransMIL architecture from the extracted ``TransMIL-main`` project: feature
    projection, adaptive square padding, a CLS token, two Nystrom self-attention
    layers, and PPEG positional encoding between the attention layers. The
    wrapper keeps MIL_BASELINE's standard forward contract by returning a dict
    with ``logits``. ``WSI_feature`` is the final CLS token. ``WSI_attn`` is the
    final patch-to-CLS similarity score used by this baseline's heatmap tools for
    self-attention MIL models.
    """

    def __init__(
        self,
        num_classes=2,
        dropout=0.1,
        act=nn.ReLU(),
        in_dim=512,
        hidden_dim=512,
        num_heads=8,
        num_landmarks=None,
        pinv_iterations=6,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes

        self.pos_layer = PPEG(dim=hidden_dim)
        self._fc1 = [nn.Linear(in_dim, hidden_dim), act]
        if dropout:
            self._fc1.append(nn.Dropout(dropout))
        self._fc1 = nn.Sequential(*self._fc1)

        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim))
        nn.init.normal_(self.cls_token, std=1e-6)
        self.layer1 = TransLayer(hidden_dim, num_heads, num_landmarks, pinv_iterations, dropout)
        self.layer2 = TransLayer(hidden_dim, num_heads, num_landmarks, pinv_iterations, dropout)
        self.norm = nn.LayerNorm(hidden_dim)
        self._fc2 = nn.Linear(hidden_dim, num_classes)
        self.apply(initialize_weights)

    def forward(self, x, return_WSI_attn=False, return_WSI_feature=False):
        forward_return = {}
        if x.dim() == 2:
            x = x.unsqueeze(0)
        n = x.shape[1]

        h = self._fc1(x.float())
        grid_h = int(np.ceil(np.sqrt(n)))
        grid_w = grid_h
        add_length = grid_h * grid_w - n
        if add_length > 0:
            h = torch.cat([h, h[:, :add_length, :]], dim=1)

        cls_tokens = self.cls_token.expand(h.shape[0], -1, -1).to(h.device)
        h = torch.cat((cls_tokens, h), dim=1)
        h = self.layer1(h)
        h = self.pos_layer(h, grid_h, grid_w)
        h = self.layer2(h)
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

