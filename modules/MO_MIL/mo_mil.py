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


def _get_optional_mamba2():
    try:
        from mamba_latest.mamba_ssm import Mamba2

        return Mamba2
    except Exception:
        try:
            from mamba_ssm import Mamba2

            return Mamba2
        except Exception:
            return None


class TorchSequenceBlock(nn.Module):
    """Pure PyTorch sequence block used when Mamba2 is unavailable.

    Reasonable implementation assumption: the public MoMIL code uses Mamba2 to
    model each token order. To keep this baseline dependency-free by default, we
    use a depthwise temporal convolution plus gated feed-forward projection as a
    lightweight sequence mixer. The multi-order construction around this block
    stays aligned with the released MoMIL source.
    """

    def __init__(self, hidden_dim=512, d_conv=4, expand=2, dropout=0.0):
        super().__init__()
        padding = d_conv // 2
        self.norm = nn.LayerNorm(hidden_dim)
        self.dwconv = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=d_conv, padding=padding, groups=hidden_dim)
        self.gate = nn.Linear(hidden_dim, hidden_dim)
        self.up = nn.Linear(hidden_dim, hidden_dim * expand)
        self.down = nn.Linear(hidden_dim * expand, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        x = self.norm(x)
        y = self.dwconv(x.transpose(1, 2)).transpose(1, 2)
        y = y[:, : x.shape[1], :]
        y = y * torch.sigmoid(self.gate(x))
        y = self.down(F.gelu(self.up(y)))
        return residual + self.dropout(y)


class Mamba2SequenceBlock(nn.Module):
    def __init__(self, hidden_dim=512, d_state=64, d_conv=4, expand=2):
        super().__init__()
        mamba2 = _get_optional_mamba2()
        if mamba2 is None:
            raise ImportError(
                "MO_MIL sequence_layer='mamba2' requires mamba_latest.mamba_ssm.Mamba2 "
                "or mamba_ssm.Mamba2. Use sequence_layer='torch' for the dependency-free fallback."
            )
        self.mamba = mamba2(d_model=hidden_dim, d_state=d_state, d_conv=d_conv, expand=expand)

    def forward(self, x):
        return x + self.mamba(x)


class MO_MIL(nn.Module):
    """MO_MIL / MoMIL baseline adapter.

    Paper: "MoMIL: Multi-order Enhanced Multiple Instance Learning for
    Computational Pathology"
    ScienceDirect: https://www.sciencedirect.com/science/article/abs/pii/S0262885626000247
    DOI: 10.1016/j.imavis.2026.105918
    Repo: https://github.com/YuqiZhang-Buaa/MoMIL

    The public repo's ``models/MoMIL.py`` implements multi-order enhancement by
    projecting patch features, padding tokens to a square length, processing
    three order views with Mamba2 blocks, concatenating those order tokens, then
    applying attention pooling and a linear classifier:
    1. original token order
    2. feature-channel reversed token order
    3. square-grid transposed token order

    Baseline adaptation:
    - forward returns ``{"logits": logits, "aux": ...}`` and supports
      ``return_WSI_feature`` / ``return_WSI_attn``.
    - ``sequence_layer='mamba2'`` uses the original Mamba2 block when installed.
      ``sequence_layer='torch'`` is the default dependency-free fallback and is
      a reasonable implementation assumption for environments without Mamba2.
    - ``max_instances`` and ``sampling`` avoid quadratic/long-sequence memory
      pressure on very large WSI bags.
    """

    def __init__(
        self,
        in_dim=512,
        num_classes=2,
        hidden_dim=512,
        dropout=0.25,
        act=nn.GELU(),
        first_order_attn_hidden=128,
        layer=2,
        second_order_type="multi_order_sequence",
        second_order_layers=None,
        num_heads=4,
        fusion_type="concat",
        use_coords=False,
        sequence_layer="torch",
        d_state=64,
        d_conv=4,
        expand=2,
        max_instances=4096,
        sampling="uniform",
    ):
        super().__init__()
        del use_coords
        if second_order_layers is not None:
            layer = second_order_layers
        if fusion_type not in {"concat", "gated"}:
            raise ValueError("fusion_type must be 'concat' or 'gated'")
        if second_order_type not in {"multi_order_sequence", "self_attn"}:
            raise ValueError("second_order_type must be 'multi_order_sequence' or 'self_attn'")
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads for the self_attn ablation branch")

        self.in_dim = in_dim
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.layer = layer
        self.second_order_type = second_order_type
        self.fusion_type = fusion_type
        self.sequence_layer = sequence_layer
        self.max_instances = max_instances
        self.sampling = sampling

        self.project = nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.LayerNorm(hidden_dim), act, nn.Dropout(dropout))

        block_cls = Mamba2SequenceBlock if sequence_layer == "mamba2" else TorchSequenceBlock
        if sequence_layer not in {"mamba2", "torch"}:
            raise ValueError("sequence_layer must be 'torch' or 'mamba2'")
        self.layers = nn.ModuleList([block_cls(hidden_dim, d_state, d_conv, expand) if sequence_layer == "mamba2" else block_cls(hidden_dim, d_conv, expand, dropout) for _ in range(layer)])
        self.layers_reverse = nn.ModuleList([block_cls(hidden_dim, d_state, d_conv, expand) if sequence_layer == "mamba2" else block_cls(hidden_dim, d_conv, expand, dropout) for _ in range(layer)])
        self.layers_transpose = nn.ModuleList([block_cls(hidden_dim, d_state, d_conv, expand) if sequence_layer == "mamba2" else block_cls(hidden_dim, d_conv, expand, dropout) for _ in range(layer)])

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 2,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.self_attn_encoder = nn.TransformerEncoder(encoder_layer, num_layers=max(1, layer))

        self.norm = nn.LayerNorm(hidden_dim)
        self.first_attention = nn.Sequential(
            nn.Linear(hidden_dim, first_order_attn_hidden),
            nn.Tanh(),
            nn.Linear(first_order_attn_hidden, 1),
        )
        self.order_attention = nn.Sequential(
            nn.Linear(hidden_dim, first_order_attn_hidden),
            nn.Sigmoid(),
            nn.Linear(first_order_attn_hidden, 1),
        )

        if fusion_type == "gated":
            self.fusion_gate = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim), nn.Sigmoid())
            self.fusion_proj = nn.Identity()
        else:
            self.fusion_gate = None
            self.fusion_proj = nn.Sequential(
                nn.Linear(hidden_dim * 4, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
            )
        self.classifier = nn.Linear(hidden_dim, num_classes)
        self.apply(initialize_weights)

    @staticmethod
    def _transpose_token_reembedding(x, width):
        b, n, c = x.shape
        return x.reshape(b, -1, width, c).transpose(1, 2).reshape(b, n, c)

    def _sample_tokens(self, x, mask):
        n = x.shape[1]
        if self.max_instances is None or self.max_instances <= 0 or n <= self.max_instances:
            return x, mask, None, n
        if self.training and self.sampling == "random":
            idx = torch.randperm(n, device=x.device)[: self.max_instances].sort().values
        elif self.sampling == "head":
            idx = torch.arange(self.max_instances, device=x.device)
        else:
            idx = torch.linspace(0, n - 1, self.max_instances, device=x.device).long()
        x = x.index_select(1, idx)
        if mask is not None:
            mask = mask.index_select(1, idx)
        return x, mask, idx, n

    def _pad_square(self, h, mask):
        n = h.shape[1]
        side = int(math.ceil(math.sqrt(n)))
        add_length = side * side - n
        if add_length > 0:
            h = torch.cat([h, h[:, :add_length, :]], dim=1)
            if mask is not None:
                mask = torch.cat([mask, torch.zeros(mask.shape[0], add_length, device=mask.device, dtype=mask.dtype)], dim=1)
        return h, mask, n, side

    def _run_layers(self, h, layers):
        for layer in layers:
            h = layer(h)
        return h

    def _masked_attention_pool(self, tokens, attention_module, mask=None):
        attn_raw = attention_module(tokens).squeeze(-1)
        if mask is not None:
            attn_raw = attn_raw.masked_fill(~mask.bool(), torch.finfo(attn_raw.dtype).min)
        attn = F.softmax(attn_raw, dim=-1)
        pooled = torch.bmm(attn.unsqueeze(1), tokens).squeeze(1)
        return pooled, attn_raw, attn

    def forward(self, x, mask=None, coords=None, return_WSI_attn=False, return_WSI_feature=False):
        del coords
        if x.dim() == 2:
            x = x.unsqueeze(0)
        if mask is not None and mask.dim() == 1:
            mask = mask.unsqueeze(0)

        forward_return = {}
        h = self.project(x.float())
        h, mask, sampled_idx, input_n = self._sample_tokens(h, mask)
        h, mask, original_n, width = self._pad_square(h, mask)
        padded_n = h.shape[1]

        if mask is None:
            valid_mask = torch.ones(h.shape[:2], dtype=torch.bool, device=h.device)
        else:
            valid_mask = mask.bool()

        h1_tokens = h[:, :original_n, :]
        h1_mask = valid_mask[:, :original_n]
        h1, attn_first_raw, attn_first = self._masked_attention_pool(h1_tokens, self.first_attention, h1_mask)

        if self.second_order_type == "self_attn":
            encoded = self.self_attn_encoder(h, src_key_padding_mask=~valid_mask)
            h2_tokens = encoded[:, :original_n, :]
            h2, attn_second_raw, attn_second = self._masked_attention_pool(h2_tokens, self.order_attention, h1_mask)
            order_tokens = h2_tokens
            order_mask = h1_mask
        else:
            h_0 = self._run_layers(h, self.layers)
            h_1 = self._run_layers(h.flip([-1]), self.layers_reverse)
            h_2 = self._run_layers(self._transpose_token_reembedding(h, width), self.layers_transpose)
            order_tokens = torch.cat([h_0, h_1, h_2], dim=1)
            order_mask = torch.cat([valid_mask, valid_mask, valid_mask], dim=1)
            order_tokens = self.norm(order_tokens)
            h2, attn_second_raw, attn_second = self._masked_attention_pool(order_tokens, self.order_attention, order_mask)

        if self.fusion_type == "gated":
            gate = self.fusion_gate(torch.cat([h1, h2], dim=-1))
            fused = gate * h1 + (1 - gate) * h2
        else:
            gate = None
            fused = self.fusion_proj(torch.cat([h1, h2, h1 * h2, torch.abs(h1 - h2)], dim=-1))

        logits = self.classifier(fused)
        forward_return["logits"] = logits
        forward_return["aux"] = {
            "attn_first_order": attn_first,
            "attn_second_order": attn_second,
            "attn_first_order_raw": attn_first_raw,
            "attn_second_order_raw": attn_second_raw,
            "fusion_gate": gate,
            "sequence_layer": self.sequence_layer,
            "second_order_type": self.second_order_type,
        }
        if return_WSI_feature:
            forward_return["WSI_feature"] = fused
        if return_WSI_attn:
            if self.second_order_type == "multi_order_sequence":
                attn_for_patches = (
                    attn_second[:, :original_n]
                    + attn_second[:, padded_n : padded_n + original_n]
                    + attn_second[:, 2 * padded_n : 2 * padded_n + original_n]
                )
            else:
                attn_for_patches = attn_second
            if sampled_idx is not None:
                full_attn = torch.zeros(attn_for_patches.shape[0], input_n, device=attn_for_patches.device, dtype=attn_for_patches.dtype)
                full_attn.index_copy_(1, sampled_idx, attn_for_patches)
                attn_for_patches = full_attn
            forward_return["WSI_attn"] = attn_for_patches.squeeze(0).unsqueeze(-1)
        return forward_return
