import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from medsteer.config import load_config  # noqa: E402
from medsteer.metrics import localisation, upsample  # noqa: E402
from semdiff_eval import znorm  # noqa: E402


def _case(seed=0, size=336, g=24):
    rng = np.random.default_rng(seed)
    pm = rng.normal(size=g * g) * 3 + 7
    mask = np.zeros((size, size), bool)
    mask[100:180, 120:200] = True
    valid = np.zeros((size, size), bool)
    valid[28:308] = True                       # padded image: top/bottom bands excluded
    frac = mask.reshape(g, size // g, g, size // g).mean((1, 3))
    vfrac = valid.reshape(g, size // g, g, size // g).mean((1, 3))
    return pm, mask, valid, frac, vfrac


def test_znorm_is_exact_on_pixels():
    pm, _, valid, _, _ = _case()
    z = znorm(pm, valid)
    pix = upsample(z, 336).reshape(-1)[valid.reshape(-1)]
    assert abs(pix.mean()) < 1e-6 and abs(pix.std() - 1) < 1e-6


def test_rank_metrics_invariant_and_q_linear():
    pm, mask, valid, frac, vfrac = _case(0)
    pc, *_ = _case(1)
    a, c = znorm(pm, valid), znorm(pc, valid)
    r_raw = localisation(pm, mask, frac, with_dice=False, valid=valid, valid_frac=vfrac)
    r_a = localisation(a, mask, frac, with_dice=False, valid=valid, valid_frac=vfrac)
    r_c = localisation(c, mask, frac, with_dice=False, valid=valid, valid_frac=vfrac)
    r_d = localisation(a - c, mask, frac, with_dice=False, valid=valid, valid_frac=vfrac)
    assert abs(r_raw["auroc"] - r_a["auroc"]) < 1e-6   # upsample() works in float32
    assert abs(r_d["css_diff"] - (r_a["css_diff"] - r_c["css_diff"])) < 1e-6


def test_config_paraphrase_one_matches_paper():
    _, new = load_config(ROOT / "configs/prompts_semdiff.yaml")
    _, old = load_config(ROOT / "configs/prompts_final2.yaml")
    for ds, cfg in new.items():
        cats = {}
        for p in cfg.prompts:
            cats.setdefault(p.category, []).append(p)
        assert set(cats) == {"pathology", "anatomy", "counterfactual", "wrong_medical", "irrelevant"}
        assert all(len(v) == 3 for v in cats.values()), ds
        prim = {p.id: p.text for p in cfg.prompts if p.primary}
        assert prim == {p.id: p.text for p in old[ds].prompts}, ds
