import numpy as np
import pytest
import torch

from medsteer import residuals, stats
from medsteer.data import mask_to_grid
from medsteer.metrics import css, localisation, otsu_threshold, patch_labels


def square_mask(size=112, lo=28, hi=56):
    m = np.zeros((size, size), bool)
    m[lo:hi, lo:hi] = True
    return m


def test_perfect_map_scores_one():
    mask = square_mask()
    frac = mask_to_grid(mask, 8)
    out = localisation(frac.reshape(-1), mask, frac)
    assert out["auroc"] > 0.97 and out["auroc_patch"] == pytest.approx(1.0)
    assert out["css_ratio"] > 1 and out["css_diff"] > 0
    assert out["dice_otsu"] > 0.8


def test_inverted_map_is_below_chance():
    mask = square_mask()
    frac = mask_to_grid(mask, 8)
    out = localisation(-frac.reshape(-1), mask, frac)
    assert out["auroc"] < 0.05 and out["css_diff"] < 0


def test_constant_map_is_chance():
    mask = square_mask()
    frac = mask_to_grid(mask, 8)
    out = localisation(np.ones(64), mask, frac)
    assert out["auroc"] == 0.5 and out["auprc_lift"] == 1.0


def test_tiny_lesion_still_has_patch_labels():
    mask = np.zeros((112, 112), bool)
    mask[30:33, 30:33] = True  # far below one 14px patch
    frac = mask_to_grid(mask, 8)
    assert patch_labels(frac).sum() == 1


def test_mask_grid_alignment():
    mask = square_mask(112, 28, 56)  # patches 2..3 in an 8x8 grid of 14 px
    frac = mask_to_grid(mask, 8)
    assert frac[2:4, 2:4].min() == 1.0 and frac.sum() == 4.0


def test_residual_maps():
    F0 = torch.randn(2, 64, 16)
    assert torch.allclose(residuals.csr_abs(F0, F0), torch.zeros(2, 64))
    Fp = F0.clone()
    Fp[:, 5] += 3.0
    r = residuals.csr_abs(Fp, F0)
    assert r[:, 5].min() > 0 and r[:, :5].abs().max() == 0
    assert torch.allclose(residuals.csr_cos(2 * F0, F0), torch.zeros(2, 64), atol=1e-6)


def test_pooled_residual_falls_back_for_empty_weights():
    dF = torch.randn(2, 64, 8)
    w = torch.zeros(2, 64)
    w[0, :4] = 1
    p = residuals.pooled_residual(dF, w)
    assert torch.allclose(p[0], dF[0, :4].mean(0), atol=1e-6)
    assert torch.allclose(p[1], dF[1].mean(0), atol=1e-6)


def test_holm():
    adj = stats.holm([0.01, 0.04, 0.03, np.nan])
    assert adj[0] == pytest.approx(0.03) and np.isnan(adj[3])
    assert adj[1] == pytest.approx(0.06) and adj[2] == pytest.approx(0.06)


def test_otsu_and_css():
    x = np.r_[np.zeros(100), np.ones(100)]
    assert 0 < otsu_threshold(x) < 1
    r, d = css(np.r_[np.ones(10) * 2, np.ones(90)], np.r_[np.ones(10, bool), np.zeros(90, bool)])
    assert r == pytest.approx(2.0, rel=1e-6) and d == pytest.approx(1.0)


def test_valid_region_removes_padding_keeps_dark_lesion():
    from medsteer.data import valid_region
    img = np.zeros((112, 112, 3), np.float32)
    img[20:90, :, :] = 0.5            # content band; rows 0-19 and 90-111 are padding
    img[40:60, 40:60, :] = 0.0        # a black (hypoechoic) lesion inside the content
    v = valid_region(img)
    assert not v[:20].any() and not v[90:].any()
    assert v[40:60, 40:60].all()


def test_localisation_ignores_padding():
    mask = square_mask()
    frac = mask_to_grid(mask, 8)
    valid = np.zeros_like(mask)
    valid[14:98] = True
    pmap = frac.reshape(-1).copy()
    pmap.reshape(8, 8)[0] = 5.0        # huge values in a padding row
    pmap.reshape(8, 8)[7] = 5.0
    with_pad = localisation(pmap, mask, frac)
    no_pad = localisation(pmap, mask, frac, valid=valid, valid_frac=mask_to_grid(valid, 8))
    assert no_pad["auroc"] > with_pad["auroc"] and no_pad["auroc_patch"] == pytest.approx(1.0)


def test_valid_region_round_head_on_black_background():
    from medsteer.data import valid_region
    yy, xx = np.mgrid[:112, :112]
    head = (yy - 56) ** 2 + (xx - 56) ** 2 < 40 ** 2
    img = np.zeros((112, 112, 3), np.float32)
    img[head] = 0.6
    img[50:60, 50:60] = 0.0           # dark tumour core inside the head
    v = valid_region(img)
    assert not v[0, 0] and not v[5, 56] and v[55, 55] and v[56, 56]
    assert 0.8 < v.sum() / head.sum() < 1.2
