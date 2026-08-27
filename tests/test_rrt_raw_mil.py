"""Run from the independent repository: python -m pytest tests/test_rrt_raw_mil.py."""

from copy import deepcopy

import hashlib
import importlib
import math
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest
import torch
import yaml

from utils.general_utils import init_epoch_info_log, add_epoch_info_log
from utils.loop_utils import cal_scores
from utils.model_utils import get_model_from_yaml
from utils.yaml_utils import read_yaml


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def deterministic_cpu():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    torch.manual_seed(42)
    yield
    torch.set_num_threads(previous)


@pytest.fixture
def config():
    config = read_yaml(ROOT / 'configs' / 'RRT_raw_MIL.yaml')
    config.Model.in_dim = 32
    config.Model.mlp_dim = 32
    config.General.num_classes = 2
    return config


def test_factory_builds_independently_from_yaml():
    from modules.RRT_raw_MIL import RRT_raw_MIL

    config = read_yaml(ROOT / 'configs' / 'RRT_raw_MIL.yaml')
    model = get_model_from_yaml(config).eval()
    assert type(model) is RRT_raw_MIL
    assert Path(importlib.import_module(RRT_raw_MIL.__module__).__file__).is_relative_to(ROOT)
    with torch.no_grad():
        output = model(torch.randn(1, 73, config.Model.in_dim))
    assert output['logits'].shape == (1, config.General.num_classes)
    assert torch.isfinite(output['logits']).all()


@pytest.mark.parametrize('shape', [(1, 32), (17, 32), (64, 32), (1, 65, 32)])
def test_bag_shapes_and_optional_outputs(config, shape):
    model = get_model_from_yaml(config).eval()
    bag = torch.randn(*shape)
    for feature, attention in [(False, False), (True, False), (False, True), (True, True)]:
        with torch.no_grad():
            output = model(bag, return_WSI_feature=feature, return_WSI_attn=attention)
        expected = {'logits'}
        if feature:
            expected.add('WSI_feature')
            assert output['WSI_feature'].shape == (32,)
        if attention:
            expected.add('WSI_attn')
            assert output['WSI_attn'].shape == (shape[-2],)
        assert set(output) == expected
        assert output['logits'].shape == (1, 2)
        assert all(torch.isfinite(value).all() for value in output.values())


@pytest.mark.parametrize('shape,message', [
    ((2, 8, 32), 'batch size 1'), ((0, 32), 'at least one instance'),
    ((8, 31), 'feature dimension'), ((1, 1, 8, 32), 'rank 2 or 3'),
])
def test_invalid_inputs(config, shape, message):
    with pytest.raises(ValueError, match=message):
        get_model_from_yaml(config)(torch.randn(*shape))


def test_factory_rejects_unsupported_branch(config):
    config.Model.ffn = True
    with pytest.raises(ValueError, match='unsupported RRT_raw_MIL ablation fields'):
        get_model_from_yaml(config)


def test_region_and_width_overrides(config):
    config.Model.region_num = 4
    config.Model.mlp_dim = 64
    model = get_model_from_yaml(config).eval()
    assert model.online_encoder.layers[0].attn.region_num == 4
    assert model.online_encoder.cr_msa.attn.region_num == 4
    with torch.no_grad():
        output = model(torch.randn(37, 32), return_WSI_feature=True)
    assert output['WSI_feature'].shape == (64,)


def test_gradients_optimizer_and_strict_checkpoint_roundtrip(config, tmp_path):
    model = get_model_from_yaml(config)
    bag = torch.randn(1, 19, 32)
    before = model.predictor.weight.detach().clone()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss = torch.nn.functional.cross_entropy(model(bag)['logits'], torch.tensor([1]))
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.requires_grad]
    assert all(g is not None and torch.isfinite(g).all() for g in grads)
    optimizer.step()
    assert not torch.equal(before, model.predictor.weight)
    checkpoint = tmp_path / 'weights.pth'
    torch.save(model.state_dict(), checkpoint)
    restored = get_model_from_yaml(config).eval()
    restored.load_state_dict(torch.load(checkpoint, weights_only=True), strict=True)
    with torch.no_grad():
        torch.testing.assert_close(model.eval()(bag)['logits'], restored(bag)['logits'], rtol=0, atol=0)


def test_existing_train_and_val_loops(config):
    from utils.loop_utils import train_loop, val_loop

    model = get_model_from_yaml(config)
    loader = [(torch.randn(1, 17 + i, 32), torch.tensor([i])) for i in range(2)]
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss, _ = train_loop(torch.device('cpu'), model, loader, criterion, optimizer, None)
    val_loss, metrics = val_loop(torch.device('cpu'), 2, model, loader, criterion)
    assert math.isfinite(loss) and math.isfinite(val_loss)
    assert math.isfinite(metrics['macro_auc'])
    assert len(metrics['_raw_probs']) == 2


@pytest.mark.parametrize('pipeline', ['Train_Val_Test', 'Train_Val', 'Train_Test'])
def test_training_dispatch_and_test_entry(config, tmp_path, monkeypatch, pipeline):
    from process.process_all import process

    runner = importlib.import_module('process.RRT_raw_MIL.process_rrt_raw_mil')
    # Only the CUDA device choice is redirected; model/data/loops/logging/checkpoints are real.
    cpu_torch = Mock(wraps=torch)
    cpu_torch.device.side_effect = lambda *_: torch.device('cpu')
    monkeypatch.setattr(runner, 'torch', cpu_torch)
    rows = {}
    for group in ('train', 'val', 'test'):
        enabled = group == 'train' or group.lower() in pipeline.lower().split('_')
        paths = []
        for label in range(2):
            path = tmp_path / f'{group}_{label}.pt'
            torch.save(torch.randn(17 + label, 32), path)
            paths.append(str(path) if enabled else None)
        rows[group + '_slide_path'] = paths
        rows[group + '_label'] = [0, 1] if enabled else [None, None]
    dataset_csv = tmp_path / 'dataset.csv'
    pd.DataFrame(rows).to_csv(dataset_csv, index=False)
    config.Dataset.dataset_csv_path = str(dataset_csv)
    config.Dataset.DATASET_NAME = 'synthetic'
    config.General.num_epochs = 2
    config.General.num_workers = 0
    config.General.device = 0
    config.Model.scheduler.warmup = 0
    config.Logs.now_log_dir = str(tmp_path / 'train_output')
    config_path = tmp_path / 'config.yaml'
    config_path.write_text(yaml.safe_dump(config.to_dict()), encoding='utf-8')
    process(config, str(config_path), None)
    output = Path(config.Logs.now_log_dir)
    checkpoint = output / 'Last_EPOCH_2.pth'
    assert checkpoint.is_file()
    log = pd.read_csv(output / 'Log_seed42_synthetic_RRT_raw_MIL.csv')
    assert len(log) == 2 and log['train_loss'].map(math.isfinite).all()
    assert tuple(log.columns) == tuple(init_epoch_info_log())
    assert log['epoch'].tolist() == [1, 2]
    if pipeline == 'Train_Test':
        assert pd.isna(log.loc[0, 'test_loss'])
        assert math.isfinite(log.loc[1, 'test_loss'])
    assert config.General.process_pipeline == pipeline
    restored = get_model_from_yaml(config)
    restored.load_state_dict(torch.load(checkpoint, weights_only=True), strict=True)
    if pipeline == 'Train_Val_Test':
        entry = importlib.import_module('test_mil')
        monkeypatch.setattr(entry, 'torch', cpu_torch)
        test_output = tmp_path / 'test_output'
        entry.test(SimpleNamespace(yaml_path=str(config_path), test_dataset_csv=str(dataset_csv),
                                   model_weight_path=str(checkpoint), test_log_dir=str(test_output)))
        assert (test_output / 'Test_Log_RRT_raw_MIL.txt').is_file()


def test_test_entry_loads_rrt_raw_checkpoint(config, tmp_path, monkeypatch):
    entry = importlib.import_module('test_mil')
    cpu_torch = Mock(wraps=torch)
    cpu_torch.device.side_effect = lambda *_: torch.device('cpu')
    monkeypatch.setattr(entry, 'torch', cpu_torch)
    paths = []
    for label in range(2):
        path = tmp_path / f'bag_{label}.pt'
        torch.save(torch.randn(17 + label, 32), path)
        paths.append(str(path))
    dataset_csv = tmp_path / 'test_data.csv'
    pd.DataFrame({'test_slide_path': paths, 'test_label': [0, 1]}).to_csv(dataset_csv, index=False)
    checkpoint = tmp_path / 'model.pth'
    torch.save(get_model_from_yaml(config).state_dict(), checkpoint)
    config_path = tmp_path / 'config.yaml'
    config_path.write_text(yaml.safe_dump(config.to_dict()), encoding='utf-8')
    test_output = tmp_path / 'evaluation'
    entry.test(SimpleNamespace(yaml_path=str(config_path), test_dataset_csv=str(dataset_csv),
                               model_weight_path=str(checkpoint), test_log_dir=str(test_output)))
    log = (test_output / 'Test_Log_RRT_raw_MIL.txt').read_text(encoding='utf-8')
    assert 'test_loss' in log and 'macro_auc' in log


@pytest.mark.parametrize("with_details", [False, True])
@pytest.mark.parametrize(
    "has_val,has_test",
    [(True, True), (True, False), (False, True), (False, False)],
)
def test_epoch_log_preserves_schema_and_metric_inputs(
    with_details, has_val, has_test
):
    metrics = cal_scores([[0.8, 0.2], [0.1, 0.9]], [0, 1], 2)
    if with_details:
        metrics.update(
            _raw_probs=[[0.8, 0.2], [0.1, 0.9]],
            _raw_labels=[0, 1],
        )
    else:
        metrics.pop("per_class")

    original = deepcopy(metrics)
    log = init_epoch_info_log()
    columns = tuple(log)

    for epoch in range(2):
        result = add_epoch_info_log(
            log, epoch, 0.3,
            0.2 if has_val else None,
            0.4 if has_test else None,
            metrics if has_val else None,
            metrics if has_test else None,
        )
        assert result == (0 if not has_val and not has_test else None)

    assert tuple(log) == columns
    assert log["epoch"] == [1, 2]
    assert all(len(values) == 2 for values in log.values())
    assert log["train_loss"] == [0.3, 0.3]
    assert log["val_loss"] == ([0.2, 0.2] if has_val else [None, None])
    assert log["test_loss"] == ([0.4, 0.4] if has_test else [None, None])

    for prefix, enabled in (("val_", has_val), ("test_", has_test)):
        for column in columns:
            if column.startswith(prefix) and column != prefix + "loss":
                expected = metrics[column[len(prefix):]] if enabled else None
                np.testing.assert_equal(log[column], [expected, expected])

    np.testing.assert_equal(metrics, original)


def test_epoch_log_rejects_missing_required_metric():
    metrics = cal_scores([[0.8, 0.2], [0.1, 0.9]], [0, 1], 2)
    metrics.pop("acc")

    with pytest.raises(KeyError, match="acc"):
        add_epoch_info_log(
            init_epoch_info_log(), 0, 0.3, 0.2, None, metrics, None
        )

def test_rrt_raw_source_snapshot_hashes():
    """Guard the reviewed brain_source_cls source snapshot from drift."""
    expected = {
        "modules/RRT_raw_MIL/__init__.py":
            "9022588477f125d3e998897d39d55dde7ef3c3afd4a185fc9f5b8b110d35058f",
        "modules/RRT_raw_MIL/region_attention.py":
            "3b3c9852e515fae034cc7c4082d5fbc9ce1143582d386b678bce87eb46e73974",
        "modules/RRT_raw_MIL/rrt_raw_mil.py":
            "6c8972823fa04dd6e5fc741487188becfcdb6c6063e9ec0b789ad3550e68dadd",
    }
    actual = {
        name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
        for name in expected
    }
    assert actual == expected
