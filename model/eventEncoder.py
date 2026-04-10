import os
import torch
import torch.nn as nn
from .foundation_model import CLIPVisionTower, SigLIPVisionTower

def _resolve_event_tower_path(config):
    """Resolve event_tower path: if it's a relative path, resolve it relative to the model directory."""
    event_tower = config.event_tower
    if not os.path.isabs(event_tower):
        model_dir = getattr(config, "_name_or_path", "")
        if model_dir:
            event_tower = os.path.join(model_dir, event_tower)
    return event_tower

def build_event_tower(config):
    event_tower_type = getattr(config, "event_tower_type", "SigLIP")
    event_tower_path = _resolve_event_tower_path(config)
    pretrained_event_tower = getattr(config, "pretrained_event_tower", "")

    if event_tower_type == "CLIP":
        return CLIPVisionTower(event_tower_path, pretrained_event_tower)

    if event_tower_type == "SigLIP":
        return SigLIPVisionTower(event_tower_path, pretrained_event_tower)

    raise ValueError(f"Unsupported event tower: {event_tower_type}")






