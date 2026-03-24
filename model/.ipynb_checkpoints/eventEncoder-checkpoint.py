import torch
import torch.nn as nn
from .foundation_model import CLIPVisionTower, SigLIPVisionTower

def build_event_tower(config): 
    event_tower_type = getattr(config, "event_tower_type", "SigLIP")
    
    if event_tower_type == "CLIP":
        return CLIPVisionTower(config.event_tower, config.pretrained_event_tower)
    
    if event_tower_type == "SigLIP":
        return SigLIPVisionTower(config.event_tower, config.pretrained_event_tower)
    
    raise ValueError(f"Unsupported event tower: {event_tower_type}")






