import torch
import torch.nn as nn

class EventQFormer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_qformer = config.num_qformer
        self.event_encoder_hidden_size = config.event_encoder_hidden_size
        self.hidden_size = config.hidden_size
        self.query_tokens = nn.Parameter(torch.randn(1, self.num_qformer, self.event_encoder_hidden_size))
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=self.event_encoder_hidden_size,
            num_heads=getattr(config, 'nhead', 8),
            batch_first=True
        )
        self.projector = nn.Linear(self.event_encoder_hidden_size, self.hidden_size)

    def forward(self, x):
        batch_size = x.size(0)
        query_tokens = self.query_tokens.expand(batch_size, -1, -1)  # (batch, num_qformer, hidden)
        out, _ = self.cross_attn(query_tokens, x, x)  # (batch, num_qformer, hidden)
        out = self.projector(out)
        return out

def build_event_projector(config):
    projector_type = getattr(config, "event_projector_type", "mlp")
    
    if projector_type == "linera":
        return nn.Linear(config.event_tower_hidden_size, config.hidden_size)
    
    if projector_type == "mlp":
        return nn.Sequential(
            nn.Linear(config.event_tower_hidden_size, config.hidden_size),
            nn.GELU(),
            nn.Linear(config.hidden_size, config.hidden_size)
        )
    
    if projector_type == "transformer":
        projector_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.nhead,
            dim_feedforward = config.event_tower_hidden_size,
        )
        projector = nn.TransformerEncoder(
            projector_layer,
            num_layers=1  
        )
        return projector
    
    if projector_type == "Q-Former":
        return EventQFormer(config)
    
    raise ValueError(f"Invalid event projector type: {projector_type}")

def build_point_cloud_projector(config):
    projector_type = getattr(config, "point_cloud_projector_type", "mlp")
    
    if projector_type == "linera":
        return nn.Linear(config.point_cloud_hidden_size, config.hidden_size)
    
    if projector_type == "mlp":
        return nn.Sequential(
            nn.Linear(config.point_cloud_hidden_size, config.hidden_size),
            nn.GELU(),
            nn.Linear(config.hidden_size, config.hidden_size)
        )
    
    if projector_type == "transformer":
        projector_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.nhead,
            dim_feedforward = config.point_cloud_hidden_size,
        )
        projector = nn.TransformerEncoder(
            projector_layer,
            num_layers=1  
        )
        return projector
    
    if projector_type == "Q-Former":
        return EventQFormer(config)
    
    raise ValueError(f"Invalid event projector type: {projector_type}")
    
    
    