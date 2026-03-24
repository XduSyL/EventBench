from dataclasses import dataclass, field
from typing import Optional, List
import transformers

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="/path/vicuna-7b-v1.5")
    version: Optional[str] = field(default="eventgpt_v1")
    freeze_backbone: bool = field(default=False)
    tune_event_projector: bool = field(default=True)
    event_tower: Optional[str] = field(default=None)
    mm_use_ev_start_end: bool = field(default=False)
    mm_use_ev_patch_token: bool = field(default=True)
    output_mm_mlp_adapter: Optional[str] = field(default=None)
    hidden_size: int = field(default=2048)
    
    # new added
    llm_backbone: Optional[str] = field(default='llama')
    event_tower_type: Optional[str] = field(default='CLIP')
    event_projector_type: Optional[str] = field(default='mlp')
    pretrain_event_projector: Optional[str] = field(default=None)
    pretrained_event_tower: Optional[str] = field(default=None)
    tuning_target_module: Optional[str] = field(default=None, 
                                                metadata={"help":'"event_towen", "event_projector", "llm_backbone"' })    
    reward_funcs: list[str] = field(
        default_factory=lambda: ["distance", "abs_dis"],
        metadata={"help": "List of reward functions. Possible values: 'accuracy', 'format'"},
    )
    task_type: Optional[str] = field(
        default='rec',
        metadata={"help": "Choose task type: 'default', 'gui', ..."},
    )
    attn_implementation: Optional[str] = field(default='flash_attention_2')
    

@dataclass
class DataArguments:
    data_path: str = field(default='',
                           metadata={"help": "Path to the training data."})
    lazy_preprocess: bool = False
    is_multimodal: bool = False
    image_folder: Optional[str] = field(default=None)
    point_cloud_folder: Optional[str] = field(default=None)
    image_aspect_ratio: str = 'square'
    event_folder: Optional[str] = field(default=None)
    video_folder: Optional[str] = field(default=None)

    # new added
    num_bins_list: Optional[List[int]] = field(default_factory=lambda: [4, 8, 16, 32])
    event_size_cfg: Optional[str] = field(default='')
    use_npz: bool = False
    use_preprocess: bool = False
    frames_upbound : Optional[int] = 32
    
    use_spatial_preprocess: bool = False

    
@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    remove_unused_columns: bool = field(default=False)
    freeze_mm_mlp_adapter: bool = field(default=False)
    mpt_attn_impl: Optional[str] = field(default="triton")
    model_max_length: int = field(
        default=512,
        metadata={
            "help":
            "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    double_quant: bool = field(
        default=True,
        metadata={"help": "Compress the quantization statistics through double quantization."}
    )
    quant_type: str = field(
        default="nf4",
        metadata={"help": "Quantization data type to use. Should be one of `fp4` or `nf4`."}
    )
    bits: int = field(
        default=16,
        metadata={"help": "How many bits to use."}
    )
    lora_enable: bool = False
    lora_r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_weight_path: str = ""
    lora_bias: str = "none"
    mm_projector_lr: Optional[float] = 1e-4
    group_by_modality_length: bool = field(default=False)
    useLora: bool = field(default=False)
    save_optimizer: bool = False
    output_dir: Optional[str] = field(default=None)


