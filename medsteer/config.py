"""Experiment configuration (prompts are frozen in configs/prompts.yaml before evaluation)."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path

import yaml

CATEGORIES = ["anatomy", "pathology", "class", "related_wrong", "irrelevant", "nuisance",
              "contrast_pos", "contrast_neg", "negation", "size_small", "size_large",
              "wrong_medical", "counterfactual"]


@dataclass
class Prompt:
    id: str
    text: str
    category: str
    label: str | None = None      # for class prompts: the diagnosis this prompt names
    primary: bool = True          # False = paraphrase variant (separate stability analysis)


@dataclass
class DatasetConfig:
    name: str
    prompts: list[Prompt]
    reference: str                # id of the control prompt used for specificity maps
    contrast_pairs: list[tuple[str, str]] = field(default_factory=list)

    def prompt(self, pid: str) -> Prompt:
        return next(p for p in self.prompts if p.id == pid)


def load_config(path: str | Path) -> tuple[dict, dict[str, DatasetConfig]]:
    raw = yaml.safe_load(Path(path).read_text())
    datasets = {}
    for name, d in raw["datasets"].items():
        prompts = [Prompt(**p) for p in d["prompts"]]
        ids = [p.id for p in prompts]
        assert len(ids) == len(set(ids)), f"duplicate prompt ids in {name}"
        for p in prompts:
            assert p.category in CATEGORIES, f"unknown category {p.category} ({p.id})"
        assert d["reference"] in ids, f"reference prompt {d['reference']} missing in {name}"
        pairs = [tuple(x) for x in d.get("contrast_pairs", [])]
        for a, b in pairs:
            assert a in ids and b in ids, f"contrast pair ({a}, {b}) not in prompts of {name}"
        datasets[name] = DatasetConfig(name, prompts, d["reference"], pairs)
    return raw, datasets


def config_hash(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:12]


def resolve_category(p: Prompt, case_label: str) -> str:
    """Class prompts become class_correct / class_wrong depending on the case's diagnosis."""
    if p.category == "class":
        return "class_correct" if p.label == case_label else "class_wrong"
    return p.category
