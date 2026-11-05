import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from medsteer.finetune import load_trainable, q_score, spec_loss, znorm  # noqa: E402
from tests.fake_model import FakeTrainableSteerModel  # noqa: E402


def test_znorm_weighted():
    torch.manual_seed(0)
    s, w = torch.randn(3, 64) * 4 + 2, torch.rand(3, 64)
    z = znorm(s, w)
    mu = (w * z).sum(1) / w.sum(1)
    var = (w * (z - mu[:, None]) ** 2).sum(1) / w.sum(1)
    assert torch.allclose(mu, torch.zeros(3), atol=1e-5) and torch.allclose(var, torch.ones(3), atol=1e-4)


def test_q_score_indicator():
    frac = torch.zeros(1, 16)
    frac[0, :4] = 1
    w = torch.ones(1, 16)
    assert torch.allclose(q_score(frac.clone(), frac, w), torch.ones(1))
    assert torch.allclose(q_score(torch.ones(1, 16), frac, w), torch.zeros(1))


def test_spec_loss_hard_uses_only_hardest_control():
    qp = torch.tensor([1.0, 1.0], requires_grad=True)
    qc = torch.tensor([[0.2, 0.9, 0.1], [0.8, 0.0, 0.3]], requires_grad=True)
    loss, hard = spec_loss(qp, qc, margin=0.5, mode="hard")
    assert hard.tolist() == [1, 0]
    loss.backward()
    g = qc.grad
    assert g[0, 1] != 0 and g[1, 0] != 0 and g[0, 0] == 0 and g[0, 2] == 0 and g[1, 1] == 0 and g[1, 2] == 0
    # margins 0.1 and 0.2 both below m=0.5 -> loss = mean(0.4, 0.3)
    assert abs(float(loss) - 0.35) < 1e-6


def test_spec_loss_mean_and_satisfied_margin():
    qp = torch.tensor([3.0])
    qc = torch.tensor([[0.0, 1.0]])
    assert float(spec_loss(qp, qc, 0.5, "mean")[0]) == 0.0
    assert float(spec_loss(qp, qc, 2.5, "hard")[0]) == pytest.approx(0.5)


def test_fake_trainable_forward_and_state_roundtrip():
    m = FakeTrainableSteerModel()
    names = m.select_trainable(train_head=False)
    assert names and not any(n.startswith("lin_seg_head") for n in names)
    x = torch.rand(2, 3, m.resolution, m.resolution)
    assert m.check_forward(x, "colon polyp") < 1e-6
    st = m.trainable_state()
    m2 = FakeTrainableSteerModel()
    with torch.no_grad():
        m2.model.connector.weight.add_(1.0)
    assert load_trainable(m2.model, st) == len(st)
    assert torch.allclose(m2.logits_grad(x, "colon polyp"), m.logits_grad(x, "colon polyp"))
    with pytest.raises(KeyError):
        load_trainable(m2.model, {"no.such.param": torch.zeros(1)})
