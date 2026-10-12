"""Tensor-only contracts for prepared MIL inputs, independent of execution tools."""

import torch


def check_feature_structure(features, coords=None, *, batched=False,
                            expected_dim=None, requires_coords=False):
    """Check declared axes/types without casting, copying, or scanning values."""
    if not isinstance(features, torch.Tensor) or not features.is_floating_point():
        raise TypeError('Features must be a floating Tensor.')
    rank = 3 if batched else 2
    form = '(B, N, D)' if batched else '(N, D)'
    if features.ndim != rank or any(size == 0 for size in features.shape):
        raise ValueError(f'Invalid feature shape: expected nonempty {form}, got {tuple(features.shape)}.')
    if expected_dim is not None and features.shape[-1] != expected_dim:
        raise ValueError(f'Feature width mismatch: expected {expected_dim}, got {features.shape[-1]}.')
    if coords is None:
        if requires_coords:
            raise ValueError('Coordinates are required.')
        return
    if not isinstance(coords, torch.Tensor) or coords.dtype == torch.bool or coords.is_complex() or coords.is_quantized:
        raise TypeError('Coordinates must be a real numeric Tensor.')
    if coords.ndim != rank or coords.shape[-1] != 2:
        raise ValueError(f'Invalid coordinate shape: expected {form.replace("D", "2")}, got {tuple(coords.shape)}.')
    if coords.shape[:-1] != features.shape[:-1]:
        raise ValueError(f'Coordinate count/batch mismatch: features {tuple(features.shape)}, coordinates {tuple(coords.shape)}.')


def prepare_feature_bag(features, coords=None, *, allow_singleton_batch=False):
    """Prepare one FP32 bag and optional coordinates, preserving source tensors."""
    if allow_singleton_batch:
        if isinstance(features, torch.Tensor) and features.ndim == 3 and features.shape[0] == 1:
            features = features.squeeze(0)
        if isinstance(coords, torch.Tensor) and coords.ndim == 3 and coords.shape[0] == 1:
            coords = coords.squeeze(0)
    check_feature_structure(features, coords)
    features = features.float()
    if not torch.isfinite(features).all():
        raise ValueError('Features must remain finite in FP32.')
    if coords is not None and not torch.isfinite(coords).all():
        raise ValueError('Coordinates must be finite.')
    return features, coords
