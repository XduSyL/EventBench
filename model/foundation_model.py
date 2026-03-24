from transformers import CLIPVisionModel, CLIPImageProcessor, CLIPVisionConfig
from transformers import SiglipVisionModel, AutoImageProcessor
from transformers.utils import logging
from functools import reduce
import torch
import os
import os
import ctypes, os
import torch.nn as nn
import numpy as np

logger = logging.get_logger(__name__)

class SigLIPVisionTower(nn.Module):
    def __init__(self, vision_tower, pretrained_event_tower):
        super().__init__()

        self.is_loaded = False
        self.vision_tower_name = vision_tower
        self.pretrained_event_tower = pretrained_event_tower

        logger.info(f"Loading SigLIP vision tower: {vision_tower}")
        self.load_model()
        if isinstance(self.vision_tower, tuple):
            self.vision_tower = self.vision_tower[0]       
        self.event_processor = self.image_processor
            

    def load_model(self, device_map=None):
        self.image_processor = AutoImageProcessor.from_pretrained(self.vision_tower_name)
        self.vision_tower = SiglipVisionModel.from_pretrained(self.vision_tower_name, device_map=device_map)
               
        if self.pretrained_event_tower:
            pretrain_weights = torch.load(self.pretrained_event_tower, map_location="cpu")
            vision_model_state_dict = {
                k.replace("model.event_tower.vision_tower.vision_model.", ""): v
                for k, v in pretrain_weights.items()
                if k.startswith("model.event_tower.vision_tower.vision_model.")
            }

            self.vision_tower.vision_model.load_state_dict(vision_model_state_dict, strict=True)
            print("Pretrained event tower weights loaded successfully.")
            
        if isinstance(self.vision_tower, tuple):
            self.vision_tower = self.vision_tower[0]
        self.vision_tower.requires_grad_(False)
        self.is_loaded = True

    def forward(self, events):
        vision_tower = self.vision_tower
        if isinstance(vision_tower, tuple):
            vision_tower = vision_tower[0]

        if isinstance(events, list):
            event_features = []
            for event in events:
                feat = vision_tower(event.to(device=self.device, dtype=self.dtype).unsqueeze(0), output_hidden_states=True)
                event_features.append(feat)
            return event_features
        else:
            return vision_tower(events.to(device=self.device, dtype=self.dtype), output_hidden_states=True)

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        vision_tower = self.vision_tower
        if isinstance(vision_tower, tuple):
            vision_tower = vision_tower[0]
        return vision_tower.dtype

    @property
    def device(self):
        vision_tower = self.vision_tower
        if isinstance(vision_tower, tuple):
            vision_tower = vision_tower[0]
        return vision_tower.device

    @property
    def config(self):
        vision_tower = self.vision_tower
        if isinstance(vision_tower, tuple):
            vision_tower = vision_tower[0]
        if self.is_loaded:
            return vision_tower.config
        else:
            return self.cfg_only

    @property
    def hidden_size(self):
        return self.config.hidden_size

    @property
    def num_patches_per_side(self):
        return self.config.image_size // self.config.patch_size

    @property
    def num_patches(self):
        return self.num_patches_per_side ** 2

    @property
    def image_size(self):
        return self.config.image_size

class CLIPVisionTower(nn.Module):
    def __init__(self, vision_tower, pretrained_event_tower):
        super().__init__()

        self.is_loaded = False
        self.vision_tower_name = vision_tower
        self.pretrained_event_tower = pretrained_event_tower

        print(f"Loading event tower: {vision_tower}")
        print(f"pretrained_event_tower: {self.pretrained_event_tower}")
        self.load_model()
        if isinstance(self.vision_tower, tuple):
            self.vision_tower = self.vision_tower[0]
        self.event_processor = self.image_processor

    def load_model(self, device_map=None):
        self.image_processor = CLIPImageProcessor.from_pretrained(self.vision_tower_name)
        self.vision_tower = CLIPVisionModel.from_pretrained(self.vision_tower_name, device_map=device_map)
        if self.pretrained_event_tower:
            pretrain_weights = torch.load(self.pretrained_event_tower, map_location="cpu")
            vision_model_state_dict = {
                k.replace("model.event_tower.vision_tower.vision_model.", ""): v
                for k, v in pretrain_weights.items()
                if k.startswith("model.event_tower.vision_tower.vision_model.")
            }

            self.vision_tower.vision_model.load_state_dict(vision_model_state_dict, strict=True)
            print("Pretrained event tower weights loaded successfully.")
        
        if isinstance(self.vision_tower, tuple):
            self.vision_tower = self.vision_tower[0]
        self.vision_tower.requires_grad_(False)

        self.is_loaded = True

    def forward(self, events):
        vision_tower = self.vision_tower
        if isinstance(vision_tower, tuple):
            vision_tower = vision_tower[0]
        if type(events) is list:
            event_features = []
            for event in events:
                event_feature = vision_tower(event.to(device=self.device, dtype=self.dtype).unsqueeze(0), output_hidden_states=True)
                event_features.append(event_feature)
        else:
            event_features = vision_tower(events.to(device=self.device, dtype=self.dtype), output_hidden_states=True)

        return event_features
    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        vision_tower = self.vision_tower
        if isinstance(vision_tower, tuple):
            vision_tower = vision_tower[0]
        return vision_tower.dtype

    @property
    def device(self):
        vision_tower = self.vision_tower
        if isinstance(vision_tower, tuple):
            vision_tower = vision_tower[0]
        return vision_tower.device

    @property
    def config(self):
        vision_tower = self.vision_tower
        if isinstance(vision_tower, tuple):
            vision_tower = vision_tower[0]
        if self.is_loaded:
            return vision_tower.config
        else:
            return self.cfg_only

    @property
    def hidden_size(self):
        _hidden_size = self.config.hidden_size
        return _hidden_size

    @property
    def num_patches_per_side(self):
        return self.config.image_size // self.config.patch_size

    @property
    def num_patches(self):
        _num_patches = (self.config.image_size // self.config.patch_size) ** 2
        return _num_patches

    @property
    def image_size(self):
        return self.config.image_size
    




