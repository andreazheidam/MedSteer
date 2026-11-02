"""MedSteer: probing clinical steerability of visual foundation models.

Pipeline: image -> F(x), F(x, p) -> Clinical Steering Residual dF -> steering-effect maps
-> localisation metrics (AUROC / AUPRC / CSS / Dice) -> paired statistics.
"""

__version__ = "0.1.0"
