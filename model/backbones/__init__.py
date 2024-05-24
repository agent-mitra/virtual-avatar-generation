import torch
from .dino_v2.dinov2.hub.backbones import dinov2_vitb14_reg, dinov2_vitl14_reg


def create_backbone(model="base", device=None):
    if model == "base":
        dinov2_vitl14 = dinov2_vitb14_reg()
    elif model == "large":
        dinov2_vitl14 = dinov2_vitl14_reg()
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
    dinov2_vitl14 = dinov2_vitl14.to(device).eval()
    return dinov2_vitl14
