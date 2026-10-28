"""Implemented pixel fidelity losses; pretrained feature objectives are optional later work."""
from losses.composite import CompositeLoss
from losses.l1_loss import CharbonnierLoss

__all__ = ["CompositeLoss", "CharbonnierLoss"]
