import pytest

torch = pytest.importorskip("torch")
from nwp.config import Config
from nwp.model import build_model


@pytest.mark.parametrize("arch", ["transformer", "lstm"])
def test_shapes_and_tying(arch):
    cfg = Config(arch=arch, d_model=32, n_layers=2, n_heads=4, d_ff=64, max_len=16)
    m = build_model(cfg, 50)
    out = m(torch.randint(0, 50, (3, 10)))
    assert out.shape == (3, 10, 50)
    assert m.head.weight.data_ptr() == (m.tok if arch == "transformer" else m.emb).weight.data_ptr()


@pytest.mark.parametrize("arch", ["transformer", "lstm"])
def test_causality(arch):
    """Changing a *later* token must not change earlier-position logits."""
    cfg = Config(arch=arch, d_model=32, n_layers=2, n_heads=4, d_ff=64, max_len=16, dropout=0.0)
    m = build_model(cfg, 50).eval()
    a = torch.randint(0, 50, (1, 10)); b = a.clone(); b[0, 7] = (b[0, 7] + 1) % 50
    with torch.no_grad():
        la, lb = m(a), m(b)
    assert torch.allclose(la[:, :7], lb[:, :7], atol=1e-5)
    assert not torch.allclose(la[:, 7:], lb[:, 7:])
