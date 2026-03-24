import torch
import torch.nn as nn
import torch.nn.functional as F

def merge_token(feature, num_patches_per_side, pool_type='avg', stride=2):
    """
    Merge the feature into a single token.
    """
    num_event_bin, num_event_token, d_model = feature.shape
    feature = feature.view(num_event_bin, num_patches_per_side, num_patches_per_side, -1)
    feature = feature.permute(0, 3, 1, 2).contiguous()
    
    if pool_type == 'avg':
        feature = F.avg_pool2d(feature, stride)
    elif pool_type == 'max':
        feature = F.max_pool2d(feature, stride)
    else:
        raise ValueError(f"Invalid pool type: {pool_type}")

    feature = feature.permute(0, 2, 3, 1).contiguous()
    feature = feature.view(num_event_bin, -1, d_model)
    return feature




