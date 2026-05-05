import math
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

warnings.filterwarnings("ignore")


def initialize_weights(module):
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


def compute_axial_cis(dim, t_x, t_y, theta=1000.0):
    device = t_x.device
    freqs_x = 1.0 / (theta ** (torch.arange(0, dim, 4, device=device).float()[: dim // 4] / dim))
    freqs_y = 1.0 / (theta ** (torch.arange(0, dim, 4, device=device).float()[: dim // 4] / dim))
    freqs_x = torch.einsum("...k,m->...m", t_x, freqs_x)
    freqs_y = torch.einsum("...k,m->...m", t_y, freqs_y)
    freqs_cis_x = torch.polar(torch.ones_like(freqs_x).float(), freqs_x.float())
    freqs_cis_y = torch.polar(torch.ones_like(freqs_y).float(), freqs_y.float())
    return torch.cat([freqs_cis_x, freqs_cis_y], dim=-1)


def rotary_90(coords):
    values = [0, torch.pi / 2, torch.pi, 3 * torch.pi / 2]
    theta = torch.tensor(values[torch.randint(0, len(values), (1,)).item()], dtype=coords.dtype, device=coords.device)
    center = torch.mean(coords / 100, dim=2, keepdim=True) * 100
    rotation_matrix = torch.tensor(
        [[torch.cos(theta), -torch.sin(theta)], [torch.sin(theta), torch.cos(theta)]],
        dtype=coords.dtype,
        device=coords.device,
    )
    return torch.einsum("...j,jk->...k", coords - center, rotation_matrix.t()) + center


def project_coords(coords):
    coords = coords - torch.min(coords, dim=-2, keepdim=True).values
    coords = coords.clamp(min=0).long()
    max_index = int(torch.max(coords).item()) + 1
    if max_index <= 1:
        return coords.float()
    projected = torch.randint(0, 10240, (*coords.shape[:-2], max_index, coords.shape[-1]), device=coords.device).float()
    projected = torch.sort(projected, dim=-2).values
    return torch.gather(projected, dim=-2, index=coords)


def random_project(coords, p=0.8):
    if torch.rand(1, device=coords.device) < p:
        return project_coords(rotary_90(coords))
    return coords


def apply_rotary_emb(xq, xk, coords, theta=1000.0, use_random_project=True):
    if use_random_project:
        coords = random_project(coords)
    t_x = coords[..., 0].reshape(*coords[..., 0].shape, 1)
    t_y = coords[..., 1].reshape(*coords[..., 1].shape, 1)
    dim = xq.shape[-1]
    freqs_cis = compute_axial_cis(dim, t_x, t_y, theta=theta)
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq).to(xq.device), xk_out.type_as(xk).to(xk.device)


def make_grid_coords(num_tokens, device, dtype):
    side = int(math.ceil(math.sqrt(num_tokens)))
    y = torch.arange(side, device=device, dtype=dtype).repeat_interleave(side)
    x = torch.arange(side, device=device, dtype=dtype).repeat(side)
    return torch.stack([x, y], dim=-1)[:num_tokens]


def minimum_bounding_rectangle(coords_np):
    try:
        from scipy.spatial import ConvexHull

        if len(coords_np) < 5:
            raise ValueError
        hull = ConvexHull(coords_np)
        hull_coords = coords_np[hull.vertices]
        min_area = float("inf")
        best_rect = None
        for i in range(len(hull_coords)):
            p1 = hull_coords[i]
            p2 = hull_coords[(i + 1) % len(hull_coords)]
            edge_dir = p2 - p1
            edge_dir = edge_dir / (np.linalg.norm(edge_dir) + 1e-6)
            angle = np.arctan2(edge_dir[1], edge_dir[0])
            rotation_matrix = np.array([[np.cos(angle), np.sin(angle)], [-np.sin(angle), np.cos(angle)]])
            rotated = np.dot(coords_np - p1, rotation_matrix)
            min_x, min_y = np.min(rotated, axis=0)
            max_x, max_y = np.max(rotated, axis=0)
            area = (max_x - min_x) * (max_y - min_y)
            if area < min_area:
                min_area = area
                rect = np.array([[min_x, min_y], [max_x, min_y], [max_x, max_y], [min_x, max_y]])
                best_rect = np.dot(rect, rotation_matrix.T) + p1
        return best_rect
    except Exception:
        min_xy = np.min(coords_np, axis=0)
        max_xy = np.max(coords_np, axis=0)
        return np.array([[min_xy[0], min_xy[1]], [max_xy[0], min_xy[1]], [max_xy[0], max_xy[1]], [min_xy[0], max_xy[1]]])


def distance_to_line(coords, line_start, line_end):
    line_vec = line_end - line_start
    point_vecs = coords - line_start
    cross = np.cross(line_vec, point_vecs)
    return np.abs(cross) / (np.linalg.norm(line_vec) + 1e-6)


def find_region_ag(coords, aggregate_num):
    coords_np = coords.detach().cpu().numpy()
    mbr = minimum_bounding_rectangle(coords_np)
    length = np.linalg.norm(mbr[1] - mbr[0])
    width = np.linalg.norm(mbr[3] - mbr[0])
    m = max(1, int(np.ceil(length / np.sqrt(max(1, aggregate_num)))))
    n = max(1, int(np.ceil(width / np.sqrt(max(1, aggregate_num)))))
    bottom_left = mbr[0]
    bottom_right = mbr[1]
    top_left = mbr[3]
    dist_bottom = distance_to_line(coords_np, bottom_left, bottom_right)
    dist_left = distance_to_line(coords_np, bottom_left, top_left)
    region_width = max(np.max(dist_bottom) / n, 1e-6)
    region_height = max(np.max(dist_left) / m, 1e-6)
    x_indices = np.clip((dist_bottom / region_width).astype(int), 0, m - 1)
    y_indices = np.clip((dist_left / region_height).astype(int), 0, n - 1)
    return torch.from_numpy(x_indices * n + y_indices).to(coords.device).long()


def fuse_labels(coords, ratio=2):
    min_xy = torch.min(coords, dim=0).values
    max_xy = torch.max(coords, dim=0).values
    width = (max_xy[0] - min_xy[0]).clamp(min=1)
    height = (max_xy[1] - min_xy[1]).clamp(min=1)
    m = max(1, int(torch.ceil(width / ratio).item()))
    n = max(1, int(torch.ceil(height / ratio).item()))
    x_idx = ((coords[:, 0] - min_xy[0]) // ratio).clamp(0, m - 1).long()
    y_idx = ((coords[:, 1] - min_xy[1]) // ratio).clamp(0, n - 1).long()
    return y_idx * m + x_idx


def create_tokens(coords, k_neighbors=8, max_dist=6 * np.sqrt(2)):
    n = coords.shape[0]
    k_neighbors = min(k_neighbors, max(0, n - 1))
    if k_neighbors == 0:
        return torch.zeros(n, 1, device=coords.device, dtype=torch.long), torch.zeros(n, 1, device=coords.device)
    distance = torch.cdist(coords.float(), coords.float(), p=2)
    vals, indices = torch.topk(distance, k=k_neighbors + 1, largest=False, dim=-1)
    self_indices = torch.arange(n, device=coords.device).unsqueeze(1).repeat(1, k_neighbors + 1)
    mask = vals < max_dist
    neighbor_index = torch.where(mask, indices, self_indices)
    neighbor_out = torch.where(mask, indices, torch.full_like(indices, -1)).float()
    return neighbor_index.long(), neighbor_out


class FlexDocumentAttention(nn.Module):
    def __init__(self, dim=512, num_heads=8, qkv_bias=False, qk_norm=False, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim={dim} must be divisible by num_heads={num_heads}")
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.kv = nn.Linear(dim, 2 * dim, bias=qkv_bias)
        self.query = nn.Parameter(torch.randn(1, dim))
        nn.init.normal_(self.query, std=1e-6)
        self.q_norm = nn.LayerNorm(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = nn.LayerNorm(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, coords, region_indices):
        b, n, c = x.shape
        _, inverse = torch.unique(region_indices, sorted=True, return_inverse=True)
        bin_count = torch.bincount(inverse)
        bin_count = bin_count[bin_count > 0]

        q = self.query.unsqueeze(0).expand(b, n, -1).reshape(b, n, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        kv = self.kv(x).reshape(b, -1, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]
        q, k = self.q_norm(q), self.k_norm(k)
        coords = coords.reshape(b, 1, -1, 2).expand(-1, self.num_heads, -1, -1)

        split_q = torch.split(q, bin_count.tolist(), dim=2)
        split_k = torch.split(k, bin_count.tolist(), dim=2)
        split_v = torch.split(v, bin_count.tolist(), dim=2)
        split_coords = torch.split(coords, bin_count.tolist(), dim=2)
        region_tokens = []
        region_coords = []
        for q_i, k_i, v_i, coords_i in zip(split_q, split_k, split_v, split_coords):
            token = F.scaled_dot_product_attention(q_i, k_i, v_i, scale=self.scale)
            token = token.permute(0, 2, 1, 3).reshape(b, -1, c)
            token = self.proj_drop(self.proj(token)).mean(dim=1, keepdim=True)
            coord = F.scaled_dot_product_attention(q_i, k_i, coords_i, scale=self.scale).mean(dim=2, keepdim=True)
            region_tokens.append(token)
            region_coords.append(coord)
        return torch.cat(region_tokens, dim=1), torch.cat(region_coords, dim=2)


class PileAttention(nn.Module):
    def __init__(self, dim=512, num_heads=8, qkv_bias=False, qk_norm=False, attn_drop=0.0, proj_drop=0.0, k_neighbors=8):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim={dim} must be divisible by num_heads={num_heads}")
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = nn.LayerNorm(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = nn.LayerNorm(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop_p = attn_drop
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.k_neighbors = k_neighbors

    def forward(self, x, seman_tokens, coords, seman_coords, attn_mask):
        b, n_1, c = x.shape
        k_neighbors = min(n_1 - 1, self.k_neighbors)
        x = torch.cat([x, seman_tokens], dim=1)
        coords = coords.unsqueeze(1).expand(-1, self.num_heads, -1, -1)
        coords = torch.cat([coords, seman_coords], dim=2)
        _, n, _ = x.shape

        qkv = self.qkv(x).view(b, n, 3, self.num_heads, c // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q, k = apply_rotary_emb(q, k, coords=coords, use_random_project=self.training)
        q = self.q_norm(q)
        k = self.k_norm(k)

        out_fuse = F.scaled_dot_product_attention(
            q[:, :, n_1:, :],
            k,
            v,
            dropout_p=self.attn_drop_p if self.training else 0.0,
            is_causal=False,
            scale=self.scale,
        )

        neighbor_index, neighbor_flag = attn_mask
        neighbor_index = neighbor_index[:, : k_neighbors + 1]
        neighbor_flag = neighbor_flag[:, : k_neighbors + 1]
        mask_index = neighbor_index.reshape(1, 1, n_1, k_neighbors + 1, 1).expand(
            b, self.num_heads, -1, -1, c // self.num_heads
        )
        pile_q = q[:, :, :n_1, :].unsqueeze(3)
        pile_k = torch.gather(k[:, :, :n_1, :].unsqueeze(3).expand(-1, -1, -1, k_neighbors + 1, -1), 2, mask_index)
        pile_v = torch.gather(v[:, :, :n_1, :].unsqueeze(3).expand(-1, -1, -1, k_neighbors + 1, -1), 2, mask_index)
        pile_k = torch.cat([pile_k, k[:, :, n_1:, :].unsqueeze(2).expand(-1, -1, n_1, -1, -1)], dim=3)
        pile_v = torch.cat([pile_v, v[:, :, n_1:, :].unsqueeze(2).expand(-1, -1, n_1, -1, -1)], dim=3)

        local_mask = neighbor_flag.unsqueeze(0).unsqueeze(1).expand(b, self.num_heads, -1, -1)
        global_mask = torch.zeros(b, self.num_heads, pile_k.shape[2], pile_k.shape[3] - k_neighbors - 1, device=x.device)
        con_mask = torch.cat([local_mask, global_mask], dim=-1)
        full_mask = torch.zeros(pile_k.shape[:-1], device=x.device).masked_fill(con_mask == -1, float("-inf")).unsqueeze(-2)
        pile_out = F.scaled_dot_product_attention(
            pile_q,
            pile_k,
            pile_v,
            attn_mask=full_mask,
            dropout_p=self.attn_drop_p if self.training else 0.0,
            is_causal=False,
            scale=self.scale,
        ).squeeze(3)

        out = torch.cat([pile_out, out_fuse], dim=2).permute(0, 2, 1, 3).reshape(b, -1, c)
        return self.proj_drop(self.proj(out))


class StableBlock(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_norm=False,
        proj_drop=0.0,
        attn_drop=0.0,
        k_neighbors=8,
    ):
        super().__init__()
        self.parti_attention = PileAttention(dim, num_heads, qkv_bias, qk_norm, attn_drop, proj_drop, k_neighbors)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(proj_drop),
            nn.Linear(int(dim * mlp_ratio), dim),
            nn.Dropout(proj_drop),
        )

    def forward(self, x, seman_x, coords, seman_coords, mask):
        n = x.shape[1]
        residual = torch.cat([x, seman_x], dim=1)
        x = residual + self.parti_attention(x, seman_x, coords, seman_coords, mask)
        x = x + self.mlp(self.norm2(x))
        return x[:, :n, :], x[:, n:, :], coords, seman_coords, mask


class StableSequential(nn.Sequential):
    def forward(self, x, seman_x, coords, seman_coords, mask):
        for module in self:
            x, seman_x, coords, seman_coords, mask = module(x, seman_x, coords, seman_coords, mask)
        return torch.cat([x, seman_x], dim=1)


class STABLE_MIL(nn.Module):
    """StableMIL for MIL_BASELINE.

    This implementation keeps the StableMIL pipeline from the extracted project:
    patch tokens are fused into stable regional tokens, regions are sorted, a
    document-level semantic attention token is built per region group, and local
    pile attention with rotary 2D coordinates performs spatially stable
    aggregation. The baseline wrapper computes the original preprocessing
    tensors on the fly from feature coordinates.

    Input can be ``[B, N, in_dim]`` or ``[B, N, in_dim + 2]`` with coords. Real
    coords are strongly recommended; feature-only inputs use a pseudo-grid only
    to keep standard MIL CSVs runnable. ``WSI_attn`` is the region/global
    similarity mapped back to every original patch, so it is an importance score
    for visualization.
    """

    def __init__(
        self,
        num_classes=2,
        in_dim=512,
        hidden_dim=512,
        depth=3,
        num_heads=8,
        aggregate_num=128,
        k_neighbors=8,
        max_dist=6 * np.sqrt(2),
        ratio=2,
        dropout=0.1,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_norm=False,
        coord_scale=512.0,
        learnable_mapping=False,
        act=None,
    ):
        super().__init__()
        del act
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.aggregate_num = aggregate_num
        self.k_neighbors = k_neighbors
        self.max_dist = max_dist + 1e-4
        self.ratio = ratio
        self.coord_scale = coord_scale
        self.learnable_mapping = learnable_mapping

        if learnable_mapping:
            self.mapping = nn.Linear(int(ratio**2) * in_dim, hidden_dim)
        else:
            self.mapping = nn.Linear(in_dim, hidden_dim)
        self.act1 = nn.GELU()
        self.ag_attn = FlexDocumentAttention(hidden_dim, num_heads, qkv_bias, qk_norm, dropout, dropout)
        self.blocks = StableSequential(
            *[
                StableBlock(hidden_dim, num_heads, mlp_ratio, qkv_bias, qk_norm, dropout, dropout, k_neighbors)
                for _ in range(depth)
            ]
        )
        self.norm = nn.Identity()
        self.fc_norm = nn.LayerNorm(hidden_dim)
        self.head_drop = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_dim, num_classes)
        self.apply(initialize_weights)

    def _split_features_coords(self, x, coords=None):
        if x.dim() == 2:
            x = x.unsqueeze(0)
        if coords is not None:
            if coords.dim() == 2:
                coords = coords.unsqueeze(0)
            return x[..., : self.in_dim], coords.float()
        if x.shape[-1] == self.in_dim + 2:
            return x[..., : self.in_dim], x[..., self.in_dim : self.in_dim + 2].float()
        coords = make_grid_coords(x.shape[1], x.device, x.dtype).unsqueeze(0).expand(x.shape[0], -1, -1)
        return x[..., : self.in_dim], coords

    def _fuse_token_single(self, x, coords):
        labels = fuse_labels(coords, self.ratio)
        unique_labels, inverse = torch.unique(labels, sorted=True, return_inverse=True)
        max_points = int(self.ratio**2)
        region_tokens = []
        region_coords = []
        for region_idx in range(unique_labels.numel()):
            mask = inverse == region_idx
            feat = x[mask]
            coord = coords[mask]
            if feat.size(0) >= max_points:
                padded_feat = feat[:max_points]
                padded_coord = coord[:max_points]
            else:
                pad_feat = feat.mean(dim=0, keepdim=True).expand(max_points - feat.size(0), -1)
                pad_coord = coord.mean(dim=0, keepdim=True).expand(max_points - coord.size(0), -1)
                padded_feat = torch.cat([feat, pad_feat], dim=0)
                padded_coord = torch.cat([coord, pad_coord], dim=0)
            if self.learnable_mapping:
                token = self.act1(self.mapping(padded_feat.reshape(1, -1))).squeeze(0)
            else:
                token = self.act1(self.mapping(padded_feat.mean(dim=0)))
            region_tokens.append(token)
            region_coords.append(padded_coord.mean(dim=0) / self.ratio)
        return torch.stack(region_tokens), torch.stack(region_coords), inverse

    def _prepare_single(self, x, coords):
        coords = coords / self.coord_scale
        region_tokens, region_coords, patch_to_region = self._fuse_token_single(x, coords)
        if region_coords.shape[0] < 2:
            region_indices = torch.zeros(region_coords.shape[0], device=x.device, dtype=torch.long)
            region_sorted_index = torch.arange(region_coords.shape[0], device=x.device)
        else:
            region_indices = find_region_ag(region_coords, self.aggregate_num)
            region_sorted_index = torch.argsort(region_indices)
        sorted_position = torch.empty_like(region_sorted_index)
        sorted_position[region_sorted_index] = torch.arange(region_sorted_index.numel(), device=x.device)
        patch_to_sorted_region = sorted_position[patch_to_region]
        sorted_coords = region_coords[region_sorted_index]
        attention_mask = create_tokens(sorted_coords, self.k_neighbors, self.max_dist)
        return region_tokens, region_coords, region_indices, region_sorted_index, attention_mask, patch_to_sorted_region

    def forward(self, x, coords=None, return_WSI_attn=False, return_WSI_feature=False):
        forward_return = {}
        x, coords = self._split_features_coords(x, coords)
        slide_features = []
        patch_scores = []

        for batch_idx in range(x.shape[0]):
            region_tokens, region_coords, region_indices, region_sorted_index, attention_mask, patch_to_region = self._prepare_single(
                x[batch_idx].float(), coords[batch_idx]
            )
            h = region_tokens.unsqueeze(0)
            h = h[:, region_sorted_index]
            h_coords = region_coords.unsqueeze(0)[:, region_sorted_index]
            region_labels = region_indices[region_sorted_index]
            seman_x, seman_coords = self.ag_attn(h, h_coords, region_labels)
            h = self.blocks(h, seman_x, h_coords, seman_coords, attention_mask)
            h = self.norm(h)
            slide_feature = self.fc_norm(h.mean(dim=1))
            slide_features.append(slide_feature)
            if return_WSI_attn:
                region_scores = torch.einsum("rc,bc->r", h.squeeze(0), slide_feature)
                patch_scores.append(region_scores[patch_to_region].unsqueeze(-1))

        slide_feature = torch.cat(slide_features, dim=0)
        logits = self.head(self.head_drop(slide_feature))
        forward_return["logits"] = logits
        if return_WSI_feature:
            forward_return["WSI_feature"] = slide_feature
        if return_WSI_attn:
            forward_return["WSI_attn"] = patch_scores[0] if len(patch_scores) == 1 else patch_scores
        return forward_return

