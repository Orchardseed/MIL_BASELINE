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


class DAG_MIL(nn.Module):
    """DAG-MIL with real coordinate support.

    The model accepts either ``[B, N, in_dim]`` plus an explicit ``coords``
    argument, or ``[B, N, in_dim + 2]`` where the last two values are ``(x, y)``
    coordinates. Feature-only input falls back to a pseudo grid so standard MIL
    csv files remain runnable, but real h5/pt coordinates are preferred.
    """

    def __init__(
        self,
        in_dim=1024,
        num_classes=2,
        dim_hidden=512,
        topk=6,
        stride=512,
        agg_type='bi-interaction',
        dropout=0.3,
        max_instances=None,
        sampling='uniform',
        eval_sampling='uniform',
    ):
        super().__init__()
        self.in_dim = in_dim
        self.topk = topk
        self.stride = stride
        self.agg_type = agg_type
        self.max_instances = max_instances
        self.sampling = sampling
        self.eval_sampling = eval_sampling

        self.fc1 = nn.Sequential(nn.Linear(in_dim, dim_hidden), nn.LeakyReLU())
        self.W_head = nn.Linear(dim_hidden, dim_hidden)
        self.W_tail = nn.Linear(dim_hidden, dim_hidden)
        self.offset_net = nn.Sequential(
            nn.Linear(dim_hidden, dim_hidden),
            nn.ReLU(),
            nn.Linear(dim_hidden, topk * 2),
            nn.Sigmoid(),
        )

        if agg_type == 'gcn':
            self.linear1 = nn.Linear(dim_hidden, dim_hidden)
            self.linear2 = None
        elif agg_type == 'bi-interaction':
            self.linear1 = nn.Linear(dim_hidden, dim_hidden)
            self.linear2 = nn.Linear(dim_hidden, dim_hidden)
        else:
            raise ValueError("agg_type must be one of: gcn, bi-interaction")

        self.out_linear = nn.Linear(dim_hidden, dim_hidden)
        self.activation = nn.LeakyReLU()
        self.message_dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(dim_hidden)
        self.classifier = nn.Linear(dim_hidden, num_classes)
        self.att_net = nn.Sequential(
            nn.Linear(dim_hidden, dim_hidden // 2),
            nn.LeakyReLU(),
            nn.Linear(dim_hidden // 2, 1),
        )
        self.expand_alpha = nn.Parameter(torch.randn(1))
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
            return x[..., : self.in_dim], x[..., self.in_dim:self.in_dim + 2].float()
        coords = self._pseudo_coords(x.shape[1], x.device, x.dtype).unsqueeze(0).expand(x.shape[0], -1, -1)
        return x[..., : self.in_dim], coords

    def _sample_instances(self, x, coords):
        n = x.shape[1]
        if self.max_instances in [None, 0] or n <= self.max_instances:
            return x, coords, None, n

        mode = self.sampling if self.training else self.eval_sampling
        if mode == 'random':
            idx = torch.randperm(n, device=x.device)[: self.max_instances].sort().values
        elif mode == 'head':
            idx = torch.arange(self.max_instances, device=x.device)
        elif mode == 'uniform':
            idx = torch.linspace(0, n - 1, self.max_instances, device=x.device).long()
        else:
            raise ValueError("sampling/eval_sampling must be one of: uniform, random, head")
        return x.index_select(1, idx), coords.index_select(1, idx), idx, n

    def forward(self, x, coords=None, return_WSI_attn=False, return_WSI_feature=False):
        forward_return = {}
        x, coords = self._split_features_coords(x, coords)
        x, coords, sampled_idx, original_n = self._sample_instances(x.float(), coords.float())

        b, n, _ = x.shape
        h = self.fc1(x)
        h = (h + h.mean(dim=1, keepdim=True)) * 0.5
        e_h = self.W_head(h)
        e_t = self.W_tail(h)

        offset = self.offset_net(e_h).reshape(b, n, self.topk, 2)
        offset = offset * self.stride * math.sqrt(max(n, 1)) * torch.sigmoid(self.expand_alpha)
        query_coords = coords.unsqueeze(2) + offset
        dist = torch.cdist(query_coords.reshape(b, n * self.topk, 2), coords, p=2)
        knn_index = dist.argmin(dim=-1).view(b, n, self.topk)

        batch_indices = torch.arange(b, device=knn_index.device).view(-1, 1, 1)
        nb_h = e_t[batch_indices, knn_index, :]
        e_h_norm = F.normalize(e_h, dim=-1)
        nb_h_norm = F.normalize(nb_h, dim=-1)
        h_expand = e_h_norm.unsqueeze(2).expand(-1, -1, self.topk, -1)
        sim_score = torch.sum(h_expand * nb_h_norm, dim=-1)
        edge_weight = F.softmax(sim_score, dim=-1)

        weighted_nb = edge_weight.unsqueeze(-1) * nb_h
        gate = torch.tanh(h_expand + weighted_nb)
        ka_weight = torch.einsum('bnkd,bnkd->bnk', nb_h, gate)
        ka_prob = F.softmax(ka_weight, dim=-1).unsqueeze(2)
        e_nh = torch.matmul(ka_prob, nb_h).squeeze(2)

        if self.agg_type == 'gcn':
            embedding = self.activation(self.linear1(e_h + e_nh))
        else:
            sum_embedding = self.activation(self.linear1(e_h + e_nh))
            bi_embedding = self.activation(self.linear2(e_h * e_nh))
            embedding = sum_embedding + bi_embedding

        embedding = self.activation(self.out_linear(embedding))
        h = self.message_dropout(embedding)
        attn = torch.softmax(self.att_net(h).squeeze(-1), dim=-1)
        wsi_feature = torch.sum(h * attn.unsqueeze(-1), dim=1)
        wsi_feature = self.norm(wsi_feature)
        logits = self.classifier(wsi_feature)

        forward_return['logits'] = logits
        if return_WSI_feature:
            forward_return['WSI_feature'] = wsi_feature
        if return_WSI_attn:
            if sampled_idx is not None:
                full_attn = torch.zeros(attn.shape[0], original_n, device=attn.device, dtype=attn.dtype)
                full_attn.index_copy_(1, sampled_idx, attn)
                attn = full_attn
            forward_return['WSI_attn'] = attn.squeeze(0).unsqueeze(-1) if attn.shape[0] == 1 else attn.unsqueeze(-1)
        return forward_return
