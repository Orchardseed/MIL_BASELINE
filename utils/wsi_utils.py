import pickle
from pathlib import Path
from collections import Counter
from collections.abc import Mapping
from torch.utils.data import WeightedRandomSampler
import pandas as pd
import h5py
import torch
from torch.utils.data import Dataset
from .input_utils import prepare_feature_bag


def read_samples_from_csv(csv_path, group):
    if group not in ("train", "val", "test"):
        raise ValueError("Expected train, val or test.")
    rows = pd.read_csv(csv_path)[[group + "_slide_path", group + "_label"]]
    missing = rows.isna()
    if (missing.iloc[:, 0] ^ missing.iloc[:, 1]).any():
        raise ValueError("Incomplete sample row: path and label are both required.")
    rows = rows.loc[~missing.all(axis=1)]
    return [dict(feature_path=path, label=int(label))
            for path, label in rows.itertuples(index=False, name=None)]

def resolve_cdp_samples(samples, centroid_dir):
    """Resolve saved centroid paths without reading original features."""
    if not centroid_dir:
        raise ValueError("BeyesGuassian_pt_dir is required for CDP_MIL.")
    resolved = []
    for sample in samples:
        filename = Path(sample["feature_path"]).name.replace(
            ".pt", "_bayesian_gaussian.pt"
        )
        centroid_path = Path(centroid_dir) / filename
        if not centroid_path.is_file():
            raise FileNotFoundError(
                f"Required centroid file not found: {centroid_path}"
            )
        resolved.append({**sample, "feature_path": str(centroid_path)})
    return resolved

def attach_h5_coordinate_paths(samples, h5_csv_path):
    paths = pd.read_csv(h5_csv_path)["h5_path"].dropna().tolist()
    by_name = {Path(path).stem: path for path in paths}
    paired = []
    for sample in samples:
        record = dict(sample)
        match = by_name.get(Path(sample["feature_path"]).stem)
        if match is not None:
            record["coords_path"] = match
        paired.append(record)
    return paired

def read_pickle(path):
    with open(path, "rb") as stream:
        return pickle.load(stream)

def read_torch(path):
    return torch.load(path, map_location="cpu")

def read_hdf5(path, keys):
    with h5py.File(path, "r") as stream:
        return {key: stream[key][()] for key in keys if key in stream}

def read_field(container, key):
    value = container[key]
    return value[()] if isinstance(value, h5py.Dataset) else value

def read_optional_position_field(container, key):
    if key is None or not isinstance(container, (Mapping, h5py.Group)):
        return None
    if key not in container:
        return None
    value = read_field(container, key)
    if value is None:
        raise ValueError("Invalid coordinate data.")
    return value

def read_coordinate_file(path, reader, key):
    if reader == "read_hdf5":
        if key is None:
            raise ValueError("HDF5 coordinate key is required.")
        loaded = read_hdf5(path, [key])
    elif reader == "read_pickle":
        loaded = read_pickle(path)
    elif reader == "read_torch":
        loaded = read_torch(path)
    else:
        raise ValueError("Unknown coordinate reader.")
    if key is None:
        if loaded is None:
            raise ValueError("Invalid coordinate data.")
        return loaded
    return read_optional_position_field(loaded, key)

class BaseWSIDataset(Dataset):
    def __init__(self, samples, input_config):
        self.samples = samples
        self.features_config = dict(input_config.get("features", {}))
        self.coordinates_config = dict(input_config.get("coordinates", {}))

    def __len__(self):
        return len(self.samples)

    @property
    def labels_list(self):
        return [sample["label"] for sample in self.samples]

    def __getitem__(self, idx):
        sample = self.samples[idx]
        raw_features, raw_positions = self._read_sample(sample)
        features = torch.as_tensor(raw_features)
        coords = None if raw_positions is None else self._positions_to_coords(raw_positions)
        features, coords = prepare_feature_bag(
            features, coords, allow_singleton_batch=True
        )
        label = torch.as_tensor(sample["label"])
        return (features, label) if coords is None else (features, label, coords)

    def _read_sample(self, sample):
        raise NotImplementedError

    def _extract_features(self, container):
        key = self.features_config.get("key")
        return container if key is None else read_field(container, key)

    def _read_external_positions(self, sample):
        path = sample.get("coords_path")
        if path is None:
            return None
        return read_coordinate_file(
            path, self.coordinates_config.get("reader"),
            self.coordinates_config.get("key")
        )

    def _read_positions(self, sample, container):
        source = self.coordinates_config.get("source", "same_file")
        if source not in ("same_file", "optional_file"):
            raise ValueError("Unknown coordinate source.")
        positions = read_optional_position_field(
            container, self.coordinates_config.get("key")
        )
        if positions is not None or source == "same_file":
            return positions
        return self._read_external_positions(sample)

    def _positions_to_coords(self, positions):
        coords = torch.as_tensor(positions)
        setting = self.coordinates_config.get("position_format")
        if setting in ("xy_top_left", "xy_center"):
            # Shared tensor preparation owns the declared point batch axis.
            return coords
        if setting != "xyxy":
            raise ValueError("Unknown position format.")
        # Only box conversion needs an (N, 4) view before preparation.
        if coords.ndim == 3 and coords.shape[0] == 1:
            coords = coords.squeeze(0)
        if coords.ndim != 2 or coords.shape[1] != 4:
            raise ValueError("Expected box shape (N, 4).")
        if coords.dtype == torch.bool or coords.is_complex() or coords.is_quantized:
            raise TypeError("Boxes must be real numeric data.")
        point = self.coordinates_config.get("box_point", "center")
        if point == "top_left":
            return coords[:, :2]
        if point == "center":
            boxes = coords.to(torch.float64)
            return (boxes[:, :2] + boxes[:, 2:]) / 2
        raise ValueError("Unknown box point.")

    def is_None_Dataset(self):
        return len(self.samples) == 0

    def is_with_labels(self):
        return bool(self.samples)  # current labeled record route only

    def get_balanced_sampler(self, replacement=True):
        labels = self.labels_list
        counts = Counter(labels)
        return WeightedRandomSampler(
            [1.0 / counts[label] for label in labels],
            num_samples=len(labels), replacement=replacement
        )


class PickleWSIDataset(BaseWSIDataset):
    def _read_sample(self, sample):
        loaded = read_pickle(sample["feature_path"])
        return self._extract_features(loaded), self._read_positions(sample, loaded)

class TorchWSIDataset(BaseWSIDataset):
    def _read_sample(self, sample):
        loaded = read_torch(sample["feature_path"])
        return self._extract_features(loaded), self._read_positions(sample, loaded)

class HDF5WSIDataset(BaseWSIDataset):
    def _read_sample(self, sample):
        key = self.features_config.get("key")
        if key is None:
            raise ValueError("HDF5 feature key is required.")
        position_key = self.coordinates_config.get("key")
        # Both modes inspect embedded positions before any optional-file read.
        keys = [key] + ([] if position_key is None else [position_key])
        loaded = read_hdf5(sample["feature_path"], keys)
        return self._extract_features(loaded), self._read_positions(sample, loaded)

def select_wsi_dataset(samples, input_config):
    reader = input_config["features"]["reader"]
    if reader == "read_pickle":
        cls = PickleWSIDataset
    elif reader == "read_torch":
        cls = TorchWSIDataset
    elif reader == "read_hdf5":
        cls = HDF5WSIDataset
    else:
        raise ValueError("Unknown feature reader.")
    return cls(samples, input_config)

def build_wsi_dataset(csv_path, group, dataset_config):
    samples = read_samples_from_csv(csv_path, group)
    source = dataset_config.get("coordinates", {}).get("source", "same_file")
    # Resolve pairs once; generic item reading uses them only if needed.
    if source == "optional_file" and dataset_config.get("h5_csv_path"):
        samples = attach_h5_coordinate_paths(samples, dataset_config["h5_csv_path"])
    return select_wsi_dataset(samples, dataset_config)
