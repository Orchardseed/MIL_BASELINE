import torch

from modules.AB_MIL.ab_mil import AB_MIL
from modules.MO_MIL.mo_mil import MO_MIL


def test_mo_mil_forward_backward():
    model = MO_MIL(
        in_dim=32,
        num_classes=2,
        hidden_dim=64,
        first_order_attn_hidden=16,
        second_order_layers=1,
        sequence_layer="torch",
        max_instances=32,
        dropout=0.0,
    )
    x = torch.randn(2, 23, 32)
    mask = torch.ones(2, 23, dtype=torch.bool)
    mask[1, 17:] = False
    y = torch.tensor([0, 1])

    out = model(x, mask=mask, return_WSI_feature=True, return_WSI_attn=True)
    assert out["logits"].shape == (2, 2)
    assert out["WSI_feature"].shape == (2, 64)
    assert "aux" in out
    assert out["aux"]["attn_first_order"].shape == (2, 23)

    loss = torch.nn.CrossEntropyLoss()(out["logits"], y)
    loss.backward()
    assert any(p.grad is not None for p in model.parameters() if p.requires_grad)


def test_ab_mil_regression_smoke():
    model = AB_MIL(L=64, D=16, num_classes=2, dropout=0.0, in_dim=32)
    x = torch.randn(1, 23, 32)
    y = torch.tensor([1])
    out = model(x)
    assert out["logits"].shape == (1, 2)
    loss = torch.nn.CrossEntropyLoss()(out["logits"], y)
    loss.backward()

