"""Datasets: manifest building, image/mask loading and model-aligned preprocessing.

Every case is one row of ``data/manifest.csv`` with the columns in ``MANIFEST_COLUMNS``.
Images and masks are loaded, zero-padded to a square and resized identically, so the
mask stays aligned with the patch grid. Grayscale images are replicated to 3 channels.
"""
from __future__ import annotations

import csv
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

MANIFEST_COLUMNS = ["case_id", "dataset", "split", "label", "plane", "image_path", "mask_paths", "patient"]
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


@dataclass
class Case:
    case_id: str
    dataset: str
    split: str
    label: str
    plane: str
    image_path: str
    mask_paths: list[str]  # empty list = no mask (healthy image, or mask not available)
    patient: str = ""      # subject id when one subject contributes several images (Sunnybrook)

    @property
    def is_healthy(self) -> bool:
        return self.label in ("normal", "no_tumor")


# --------------------------------------------------------------------------- manifest IO

def read_manifest(path: str | Path, datasets: list[str] | None = None,
                  splits: list[str] | None = None) -> list[Case]:
    cases = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            if datasets and row["dataset"] not in datasets:
                continue
            if splits and row["split"] not in splits:
                continue
            masks = [m for m in row["mask_paths"].split("|") if m]
            cases.append(Case(row["case_id"], row["dataset"], row["split"], row["label"],
                              row.get("plane", ""), row["image_path"], masks,
                              row.get("patient", "") or ""))
    return cases


def write_manifest(cases: list[Case], path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(MANIFEST_COLUMNS)
        for c in cases:
            w.writerow([c.case_id, c.dataset, c.split, c.label, c.plane, c.image_path,
                        "|".join(c.mask_paths), c.patient])


def summarise(cases: list[Case]) -> str:
    lines = []
    by = defaultdict(Counter)
    masked = defaultdict(Counter)
    for c in cases:
        by[(c.dataset, c.split)][c.label] += 1
        if c.mask_paths:
            masked[(c.dataset, c.split)][c.label] += 1
    for key in sorted(by):
        lines.append(f"{key[0]:>6} {key[1]:>5}: " + ", ".join(
            f"{lab}={n} (masks {masked[key][lab]})" for lab, n in sorted(by[key].items())))
    return "\n".join(lines)


# --------------------------------------------------------------------------- BUSI

def build_busi(root: str | Path, seed: int = 0, fractions=(0.7, 0.1, 0.2)) -> list[Case]:
    """Dataset_BUSI_with_GT/{benign,malignant,normal}/'benign (1).png' + '..._mask[_k].png'.

    BUSI has no official split; a stratified split is made for bookkeeping only (the probing
    experiments need no training). Multiple masks per image are combined as a union.

    Also handles reorganised copies, e.g. Images/{benign,malignant}/x.png +
    Labels/{benign,malignant}/x[_mask].png + normal/x.png: the class comes from a class-named
    parent folder, a file is a mask if it sits under a folder called labels/masks/gt/
    annotations or has '_mask' in its name, and masks are matched to images by file stem.
    """
    root = Path(root)
    mask_dirs = {"labels", "label", "masks", "mask", "gt", "groundtruth", "ground_truth",
                 "annotations"}
    files = [p for p in sorted(root.rglob("*"))
             if p.suffix.lower() in IMAGE_EXTS and "__MACOSX" not in p.parts]
    if not files:
        raise FileNotFoundError(f"no BUSI images found under {root}")

    def label_of(p: Path) -> str:
        for part in reversed(p.parts[:-1]):
            if part.lower() in ("benign", "malignant", "normal"):
                return part.lower()
        m = re.match(r"(benign|malignant|normal)", p.stem.lower())
        return m.group(1) if m else ""

    def is_mask(p: Path) -> bool:
        return "_mask" in p.stem.lower() or any(x.lower() in mask_dirs for x in p.parts[:-1])

    masks = defaultdict(list)
    for p in files:
        if is_mask(p):
            stem = re.sub(r"_mask(_\d+)?$", "", p.stem, flags=re.I)
            masks[(label_of(p), stem)].append(str(p))
    cases, seen = [], set()
    for img in files:
        if is_mask(img):
            continue
        label = label_of(img)
        if not label:
            continue
        cid = "busi_" + re.sub(r"[^a-z0-9]+", "_", img.stem.lower()).strip("_")
        if cid in seen:
            continue
        seen.add(cid)
        # normal images have no (or empty) masks; they define "no lesion"
        cases.append(Case(cid, "busi", "", label, "", str(img), sorted(masks.get((label, img.stem), []))))
    _stratified_split(cases, seed, fractions)
    return cases


# --------------------------------------------------------------------------- BRISC 2025

BRISC_CODES = {"gl": "glioma", "me": "meningioma", "pi": "pituitary", "nt": "no_tumor",
               "no": "no_tumor"}
BRISC_LABELS = {"glioma", "meningioma", "pituitary", "no_tumor"}
BRISC_PLANES = {"ax": "axial", "co": "coronal", "sa": "sagittal"}


def _brisc_label(path: Path) -> str:
    for part in reversed(path.parts[:-1]):
        p = part.lower().replace(" ", "_").replace("-", "_")
        if p in BRISC_LABELS:
            return p
        if p in ("notumor", "no_tumour", "healthy"):
            return "no_tumor"
    for tok in path.stem.lower().split("_"):
        if tok in BRISC_CODES:
            return BRISC_CODES[tok]
    return ""


def _brisc_plane(path: Path) -> str:
    for tok in path.stem.lower().split("_"):
        if tok in BRISC_PLANES:
            return BRISC_PLANES[tok]
    return ""


def _brisc_split(path: Path) -> str:
    parts = [p.lower() for p in path.parts]
    for s in ("train", "test", "val"):
        if s in parts:
            return s
    return ""


def build_brisc(root: str | Path) -> list[Case]:
    """Walk a BRISC 2025 folder and pair images with masks by file stem.

    Handles the usual layout (classification_task/{train,test}/<class>/*.jpg and
    segmentation_task/{train,test}/{images,masks}/*) but only relies on: masks live in a
    folder called 'masks' (or have '_mask' in the name) and share the image's stem;
    the class comes from a class-named parent folder or the filename code (_gl_, _me_,
    _pi_, _nt_). The same image appearing in several folders is kept once.
    CHECK THE PRINTED COUNTS: tumour images without masks cannot be used for localisation.
    """
    root = Path(root)
    files = [p for p in root.rglob("*") if p.suffix.lower() in IMAGE_EXTS]
    is_mask = lambda p: "masks" in [x.lower() for x in p.parts] or "_mask" in p.stem.lower()
    masks = defaultdict(list)
    for p in files:
        if is_mask(p):
            stem = re.sub(r"_mask(_\d+)?$", "", p.stem)
            masks[stem].append(str(p))
    seen: dict[str, Case] = {}
    for p in sorted(files):
        if is_mask(p):
            continue
        label = _brisc_label(p)
        if not label:
            continue
        cid = "brisc_" + re.sub(r"[^a-z0-9]+", "_", p.stem.lower()).strip("_")
        m = sorted(masks.get(p.stem, []))
        case = Case(cid, "brisc", _brisc_split(p), label, _brisc_plane(p), str(p), m)
        prev = seen.get(cid)
        if prev is None or (m and not prev.mask_paths):
            seen[cid] = case
    return list(seen.values())


# --------------------------------------------------------------------------- Kvasir-SEG

def build_kvasir(root: str | Path, seed: int = 0, fractions=(0.7, 0.1, 0.2)) -> list[Case]:
    """Kvasir-SEG/{images,masks}/<id>.jpg: 1,000 polyp images, mask shares the file stem.

    One class (polyp), no healthy images, no official split (a random split is made for
    bookkeeping only)."""
    root = Path(root)
    files = [p for p in sorted(root.rglob("*"))
             if p.suffix.lower() in IMAGE_EXTS and "__MACOSX" not in p.parts]
    is_mask = lambda p: any(x.lower() in ("masks", "mask") for x in p.parts[:-1])
    masks = {p.stem: str(p) for p in files if is_mask(p)}
    cases = []
    for img in files:
        if is_mask(img) or not any(x.lower() == "images" for x in img.parts[:-1]):
            continue
        m = masks.get(img.stem)
        cases.append(Case("kvasir_" + re.sub(r"[^a-z0-9]+", "_", img.stem.lower()).strip("_"),
                          "kvasir", "", "polyp", "", str(img), [m] if m else []))
    if not cases:
        raise FileNotFoundError(f"no Kvasir-SEG images found under {root} (expected images/ and masks/)")
    _stratified_split(cases, seed, fractions)
    return cases


# --------------------------------------------------------------------------- Sunnybrook

def build_sunnybrook(root: str | Path, seed: int = 0, fractions=(0.7, 0.1, 0.2)) -> list[Case]:
    """Sunnybrook cardiac MRI (short axis), left-ventricle masks.

    images/<group>/<patient>_rawdcm_<frame>.png   (the .dcm copies are ignored)
    labels/<group>/<patient>gtmask<frame:04d>.png
    e.g. SC-N-02_rawdcm_020.png <-> SC-N-02gtmask0020.png. <group> (Healthy_normal,
    LV_hypertrophy, ...) is the patient's diagnosis, kept as the label. Every image shows the
    target structure, so no image counts as 'healthy' in the no-target sense.
    Several images per patient: the split is made per patient, and the patient id is stored
    so analyses can aggregate per patient (images of one patient are not independent)."""
    root = Path(root)
    imgs, masks = [], {}
    for p in sorted(root.rglob("*.png")):
        if "__MACOSX" in p.parts:
            continue
        parts = [x.lower() for x in p.parts[:-1]]
        m = re.match(r"(.+?)gtmask(\d+)$", p.stem, flags=re.I)
        if m and ("labels" in parts or "label" in parts or "masks" in parts):
            masks[(m.group(1), int(m.group(2)))] = str(p)
            continue
        m = re.match(r"(.+?)_rawdcm_(\d+)$", p.stem, flags=re.I)
        if m:
            imgs.append((p, m.group(1), int(m.group(2))))
    if not imgs:
        raise FileNotFoundError(f"no Sunnybrook images (*_rawdcm_*.png) found under {root}")
    cases = []
    for p, pat, frame in imgs:
        mk = masks.get((pat, frame))
        cases.append(Case(f"sunny_{re.sub(r'[^a-z0-9]+', '_', pat.lower())}_{frame:04d}", "sunnybrook",
                          "", p.parent.name.lower(), "short_axis", str(p), [mk] if mk else [], pat))
    # patient-level stratified split
    pats = {}
    for c in cases:
        pats.setdefault(c.patient, c.label)
    proxies = [Case(pid, "sunnybrook", "", lab, "", "", []) for pid, lab in sorted(pats.items())]
    _stratified_split(proxies, seed, fractions)
    split_of = {x.case_id: x.split for x in proxies}
    for c in cases:
        c.split = split_of[c.patient]
    return cases


def _stratified_split(cases: list[Case], seed: int, fractions) -> None:
    rng = random.Random(seed)
    by_label = defaultdict(list)
    for c in cases:
        by_label[c.label].append(c)
    for group in by_label.values():
        rng.shuffle(group)
        n = len(group)
        n_tr, n_va = int(round(fractions[0] * n)), int(round(fractions[1] * n))
        for i, c in enumerate(group):
            c.split = "train" if i < n_tr else ("val" if i < n_tr + n_va else "test")


# --------------------------------------------------------------------------- loading

def _pad_square(arr: np.ndarray) -> np.ndarray:
    h, w = arr.shape[:2]
    s = max(h, w)
    out = np.zeros((s, s) + arr.shape[2:], dtype=arr.dtype)
    top, left = (s - h) // 2, (s - w) // 2
    out[top:top + h, left:left + w] = arr
    return out


def load_image(path: str, size: int) -> np.ndarray:
    """-> float32 RGB array (size, size, 3) in [0, 1], zero-padded to square then resized."""
    img = Image.open(path)
    if img.mode not in ("L", "RGB"):
        img = img.convert("L") if img.mode in ("I;16", "I", "F", "LA", "P") else img.convert("RGB")
    arr = np.asarray(img)
    if arr.dtype == np.uint16 or arr.dtype == np.int32 or arr.dtype == np.float32:
        arr = arr.astype(np.float32)
        arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8) * 255.0
        arr = arr.astype(np.uint8)
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)  # grayscale -> 3 channels
    arr = _pad_square(arr)
    arr = np.asarray(Image.fromarray(arr).resize((size, size), Image.BICUBIC))
    return arr.astype(np.float32) / 255.0


def load_mask(paths: list[str], size: int, image_path: str) -> np.ndarray:
    """-> bool mask (size, size), union of all mask files, aligned with ``load_image``."""
    if not paths:
        return np.zeros((size, size), dtype=bool)
    union = None
    for p in paths:
        a = np.asarray(Image.open(p).convert("L"))
        m = a > (127 if a.max() > 1 else 0)   # 0/255 masks and 0/1 masks
        union = m if union is None else (union | m)
    union = _pad_square(union.astype(np.uint8) * 255)
    union = np.asarray(Image.fromarray(union).resize((size, size), Image.NEAREST)) > 127
    return union


VALID_REGION = "unpadded image (only our zero padding excluded; no intensity-based removal)"


def content_region(image_path: str, size: int) -> np.ndarray:
    """Bool (size, size): the pixels of the original, unpadded image (feedback 2026-10-28, pt 5).

    Defined from the image GEOMETRY only (width, height): never from the ground truth and never
    from intensities, so naturally dark regions (MRI background, dermoscopy vignetting, dark
    lumen, the Kvasir position overlay) stay in the evaluated region. Built with exactly the
    padding + nearest-neighbour resize of ``load_mask``, so every target pixel lies inside it."""
    with Image.open(image_path) as im:
        w, h = im.size
    ones = np.full((h, w), 255, dtype=np.uint8)
    reg = _pad_square(ones)
    return np.asarray(Image.fromarray(reg).resize((size, size), Image.NEAREST)) > 127


def valid_region(img: np.ndarray, thr: float = 0.06, min_area: float = 0.01) -> np.ndarray:
    """LEGACY (v2 runs only; replaced by ``content_region``).
    Bool (size, size): the image content (head, breast tissue), excluding our zero padding,
    black acquisition borders and black MRI background.

    tissue = brightness > thr -> opening (removes speckle and burnt-in text) -> keep
    components > ``min_area`` of the image -> convex hull. The hull keeps dark structures
    inside the anatomy (hypoechoic lesions, CSF, necrotic tumour cores)."""
    from scipy import ndimage
    from skimage.morphology import convex_hull_image

    gray = img.mean(axis=-1)
    fg = ndimage.binary_opening(gray > thr, structure=np.ones((5, 5)))
    lab, n = ndimage.label(fg)
    if n == 0:
        return np.ones_like(gray, dtype=bool)
    areas = ndimage.sum(fg, lab, index=np.arange(1, n + 1))
    # keep the anatomy: components that are large in absolute terms AND comparable to the
    # largest one (drops edge noise, rulers and text that would stretch the hull)
    big = (areas >= min_area * fg.size) & (areas >= 0.2 * areas.max())
    keep = np.isin(lab, 1 + np.flatnonzero(big))
    if not keep.any():
        return np.ones_like(gray, dtype=bool)
    return convex_hull_image(keep)


def refine_valid(img: np.ndarray, valid: np.ndarray, dataset: str, mask: np.ndarray | None = None,
                 thr: float = 0.06, min_area: float = 0.005) -> np.ndarray:
    """LEGACY (v2 runs only; replaced by ``content_region``). Used the ground truth (target
    pixels were never removed), which is why it was replaced.
    Dataset-specific clean-up of the content region.

    Kvasir-SEG: many images carry a black box (endoscope position overlay) in a corner; the
    convex hull keeps it inside the 'image'. Remove near-black components larger than
    ``min_area`` of the image. Only for Kvasir, because in other modalities dark regions inside
    the anatomy are real tissue (hypoechoic lesions, CSF) and must stay."""
    if dataset != "kvasir":
        return valid
    from scipy import ndimage
    dark = ndimage.binary_opening(img.mean(axis=-1) < thr, structure=np.ones((3, 3))) & valid
    lab, n = ndimage.label(dark)
    if n == 0:
        return valid
    areas = ndimage.sum(dark, lab, index=np.arange(1, n + 1))
    box = np.isin(lab, 1 + np.flatnonzero(areas >= min_area * dark.size))
    if mask is not None:
        box &= ~mask
    return valid & ~box


def mask_to_grid(mask: np.ndarray, grid: int) -> np.ndarray:
    """Per-patch lesion fraction (grid, grid) by area averaging."""
    s = mask.shape[0]
    assert s % grid == 0, "mask size must be a multiple of the patch grid"
    k = s // grid
    return mask.reshape(grid, k, grid, k).mean(axis=(1, 3)).astype(np.float32)
