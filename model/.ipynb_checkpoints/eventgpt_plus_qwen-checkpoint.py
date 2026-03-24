import torch
import random
import torch.nn as nn
from typing import Optional, List, Union
from transformers.generation.utils import GenerateOutput
from model.eventProjector import build_event_projector
from model.eventEncoder import build_event_tower
from utils.token_merge import merge_token
from utils.constents import IGNORE_INDEX, EVENT_TOKEN_INDEX, DEFAULT_EVENT_PATCH_TOKEN, DEFAULT_EV_START_TOKEN, DEFAULT_EV_END_TOKEN, EVENT_PAD_INDEX
from transformers import AutoConfig, AutoModelForCausalLM
from transformers import Qwen2Config, Qwen2Model, Qwen2ForCausalLM


class EventGPTPlusOutputWrapper:
    def __init__(self, outputs, new_input_ids):
        self.outputs = outputs
        self.new_input_ids = new_input_ids

    def __getattr__(self, item):
        return getattr(self.outputs, item)

    def __getitem__(self, key):
        return self.outputs[key]

    def __iter__(self):
        return iter(self.outputs)

    def keys(self):
        return self.outputs.keys()

class EventGPTPlusQwenConfig(Qwen2Config):
    model_type = "eventgpt_plus_qwen" 
    
class EventGPTPlusQwenModel(Qwen2Model):
    config_class = EventGPTPlusQwenConfig

    def __init__(self, config: Qwen2Config):
        super(EventGPTPlusQwenModel, self).__init__(config)

        if hasattr(config, "event_tower"):          
            self.event_tower = build_event_tower(config)
            self.event_projector = build_event_projector(config).to(dtype=torch.bfloat16)       
    
    def get_event_tower(self):
        event_tower = getattr(self, 'event_tower', None)
        if type(event_tower) is list:
            event_tower = event_tower[0]
        return event_tower
    
    def get_point_cloud_encoder(self):
        point_cloud_encoder = getattr(self, 'point_cloud_encoder', None)
        return point_cloud_encoder

    def initialize_event_modules(self, model_args, fsdp=None):
        event_tower = model_args.event_tower
        self.config.event_tower = event_tower
        
        if model_args.ues_point_cloud:
            self.config.PointCloudEncoder = model_args.PointCloudEncoder
            
        # Build the event tower
        event_tower = build_event_tower(model_args) 
        self.config.event_tower_type = model_args.event_tower_type
        self.event_tower = event_tower
        
        # Build the point cloud encoder
        if model_args.ues_point_cloud:
            point_cloud_encoder = build_point_cloud_encoder(model_args)
            self.point_cloud_encoder = point_cloud_encoder
            self.config.PointCloudEncoder = model_args.PointCloudEncoder

            # Build the point cloud projector
            point_cloud_hidden_size = point_cloud_encoder.hidden_size
            model_args.point_cloud_hidden_size = point_cloud_hidden_size   
            self.point_cloud_projector = build_point_cloud_projector(model_args).to(dtype=torch.bfloat16)
            self.config.point_cloud_hidden_size = point_cloud_hidden_size
        
        # Build the event projector
        event_tower_hidden_size = event_tower.config.hidden_size
        model_args.event_tower_hidden_size = event_tower_hidden_size
        self.event_projector = build_event_projector(model_args).to(dtype=torch.bfloat16)
        self.config.event_tower_hidden_size = event_tower_hidden_size
        
        # TODO: revise this
        # Load pretrained weights for visual_projector if provided
        if model_args.pretrain_event_projector is not None:
            print("Loading event_projector pretrain weights...")
            pretrained_weights = torch.load(model_args.pretrain_event_projector)
            # Adjust keys to match model structure
            pretrained_weights = {k.replace("model.event_projector.", ""): v for k, v in pretrained_weights.items()}
            self.event_projector.load_state_dict(pretrained_weights, strict=True)
            print("Pretrained weights loaded successfully into visual_projector.")
            

class EventGPTPlusQwenCausalLM(Qwen2ForCausalLM):

    config_class = EventGPTPlusQwenConfig

    def __init__(self, config) -> None:
        # super(Qwen2ForCausalLM, self).__init__(config)       
        Qwen2ForCausalLM.__init__(self, config)
        config.model_type = "EventChat_Qwen"
        config.rope_scaling = None
        self.model = EventGPTPlusQwenModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
    def get_model(self):
        return self.model
    
    def get_event_tower(self):
        return self.get_model().event_tower
    
    def encode_event(self, event_tensor):
        event_features = self.get_model().get_event_tower()(event_tensor)
        event_features = event_features['last_hidden_state']
        event_features = event_features[:,1:,:]
        event_features = self.get_model().event_projector(event_features)
        return event_features

    
    def initialize_event_tokenizer(self, model_args, tokenizer):
        if model_args.mm_use_ev_patch_token:
            tokenizer.add_tokens([DEFAULT_EVENT_PATCH_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

        if model_args.mm_use_ev_start_end:
            num_new_tokens = tokenizer.add_tokens([DEFAULT_EV_START_TOKEN, DEFAULT_EV_END_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

            if num_new_tokens > 0:
                input_embeddings = self.get_input_embeddings().weight.data
                output_embeddings = self.get_output_embeddings().weight.data

                input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True)
                output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True)

                input_embeddings[-num_new_tokens:] = input_embeddings_avg
                output_embeddings[-num_new_tokens:] = output_embeddings_avg

            if model_args.tune_event_projector:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = True
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False

            if model_args.pretrain_event_projector:
                mm_projector_weights = torch.load(model_args.pretrain_event_projector, map_location='cpu')
                embed_tokens_weight = mm_projector_weights['model.embed_tokens.weight']
                assert num_new_tokens == 2
                if input_embeddings.shape == embed_tokens_weight.shape:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight[-num_new_tokens:]
                elif embed_tokens_weight.shape[0] == num_new_tokens:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight
                else:
                    raise ValueError(f"Unexpected embed_tokens_weight shape. Pretrained: {embed_tokens_weight.shape}. Current: {input_embeddings.shape}. Numer of new tokens: {num_new_tokens}.")
        elif model_args.mm_use_ev_patch_token:
            if model_args.tune_event_projector:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = False
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False
                    
    def forward(self, 
            event_tensors: Optional[torch.FloatTensor] = None,
            point_cloud_file: Optional[str] = None,
            input_ids: torch.LongTensor = None, 
            labels: Optional[torch.LongTensor] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            use_cache: Optional[bool] = None,
            event_image_sizes : Optional[List[List[int]]] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            cache_position: Optional[torch.LongTensor] = None,
            **kwargs):

        new_input_ids = None
        
        if inputs_embeds is None:
            (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels,
                new_input_ids 
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids, 
                position_ids,
                attention_mask,
                past_key_values,
                labels,
                event_tensors,
                point_cloud_file,
                event_image_sizes           
            )
            
        outputs = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )      


        torch.cuda.synchronize()
        return EventGPTPlusOutputWrapper(outputs, new_input_ids)
    
####非point_cloud_only############
    def prepare_inputs_labels_for_multimodal(self, input_ids, position_ids, attention_mask, 
                                            past_key_values, labels, event_tensors,
                                            point_cloud_file=None, event_bin_sizes=None):
        if event_tensors is not None and not isinstance(event_tensors, list):
            event_tensors = [event_tensors]
        if point_cloud_file is not None and not isinstance(point_cloud_file, list):
            point_cloud_file = [point_cloud_file]

        num_patches_per_side = self.get_event_tower().num_patches_per_side
        event_tower = self.get_event_tower()

        if event_tower is None or event_tensors is None or input_ids.shape[1] == 1:
            return input_ids, position_ids, attention_mask, past_key_values, None, labels, None

        pc_features_list = None
        if point_cloud_file:
            pc_features_list = []
            for pt_file in point_cloud_file:
                pc_feat = self.encoder_point_cloud(pt_file)
                # pc_feat = self.encode_point_cloud_moe(pt_file, moe_cfg=getattr(self.config,'pc_moe_cfg',{}))
                if pc_feat.dim() == 1:
                    pc_feat = pc_feat.unsqueeze(0)
                embed = self.get_model().embed_tokens
                pc_feat = pc_feat.to(device=embed.weight.device, dtype=embed.weight.dtype)
                pc_features_list.append(pc_feat)

        if isinstance(event_tensors, list):
            event_tensors = [x.unsqueeze(0) if x.ndim == 3 else x for x in event_tensors]
        event_tensors_list = []
        for event_tensor in event_tensors:
            event_tensors_list.append(event_tensor if event_tensor.ndim == 4 else event_tensor.unsqueeze(0))

        all_event_tensors = torch.cat([t for t in event_tensors_list], dim=0)
        split_idxs = [t.shape[0] for t in event_tensors_list]
        encoded_all_event_tensors = self.encode_event(all_event_tensors)
        encoded_all_event_tensors = torch.split(encoded_all_event_tensors, split_idxs)

        event_features = []
        for ev_feat in encoded_all_event_tensors:
            ev_feat = merge_token(ev_feat, num_patches_per_side)
            ev_feat = ev_feat.flatten(0, 1)
            event_features.append(ev_feat)

        _labels = labels
        _position_ids = position_ids
        _attention_mask = attention_mask
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        else:
            attention_mask = attention_mask.bool()
        if position_ids is None:
            position_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=input_ids.device)
        if labels is None:
            labels = torch.full_like(input_ids, IGNORE_INDEX)

        input_ids = [ids[mask] for ids, mask in zip(input_ids, attention_mask)]
        labels = [lab[mask] for lab, mask in zip(labels, attention_mask)]

        new_input_embeds, new_labels, new_input_ids_list = [], [], []
        cur_event_idx = 0
        embed = self.get_model().embed_tokens
        device = embed.weight.device
        dtype = embed.weight.dtype

        for batch_idx, cur_input_ids in enumerate(input_ids):
            num_event_bin = int((cur_input_ids == EVENT_TOKEN_INDEX).sum().item())

            if num_event_bin == 0:
                cur_input_embeds_1 = embed(cur_input_ids.to(device))
                new_input_embeds.append(cur_input_embeds_1)
                new_labels.append(labels[batch_idx].to(device))
                new_input_ids_list.append(cur_input_ids.to(device))
                continue

            event_token_pos = torch.where(cur_input_ids == EVENT_TOKEN_INDEX)[0].tolist()
            cur_labels = labels[batch_idx]

            seg_starts = [-1] + event_token_pos + [cur_input_ids.shape[0]]
            text_segments = []
            label_segments = []
            for i in range(len(seg_starts) - 1):
                text_segments.append(cur_input_ids[seg_starts[i] + 1 : seg_starts[i + 1]])
                label_segments.append(cur_labels[seg_starts[i] + 1 : seg_starts[i + 1]])

            text_lens = [seg.shape[0] for seg in text_segments]
            if sum(text_lens) > 0:
                text_all = embed(torch.cat(text_segments).to(device))
                text_splits = torch.split(text_all, text_lens, dim=0)
            else:
                text_splits = [torch.empty((0, embed.embedding_dim), device=device, dtype=dtype) for _ in text_segments]

            cur_new_input_embeds, cur_new_labels, cur_new_input_ids = [], [], []
            inserted_once = False

            for i in range(num_event_bin + 1):
                cur_new_input_embeds.append(text_splits[i])
                cur_new_labels.append(label_segments[i].to(device))
                cur_new_input_ids.append(text_segments[i].to(device))

                if i < num_event_bin:
                    if not inserted_once:
                        ev_feat = event_features[cur_event_idx].to(device=device, dtype=dtype)
                        if pc_features_list is not None and batch_idx < len(pc_features_list) and pc_features_list[batch_idx] is not None:
                            ev_feat = torch.cat([ev_feat, pc_features_list[batch_idx]], dim=0)
                        cur_new_input_embeds.append(ev_feat)
                        cur_new_labels.append(torch.full((ev_feat.shape[0],), IGNORE_INDEX, device=device, dtype=cur_labels.dtype))
                        cur_new_input_ids.append(torch.full((ev_feat.shape[0],), EVENT_PAD_INDEX, device=device, dtype=torch.long))
                        inserted_once = True
                        cur_event_idx += 1
                    else:
                        pass

            cur_new_input_embeds = torch.cat(cur_new_input_embeds, dim=0)
            cur_new_labels = torch.cat(cur_new_labels, dim=0)
            cur_new_input_ids = torch.cat(cur_new_input_ids, dim=0)

            new_input_embeds.append(cur_new_input_embeds)
            new_labels.append(cur_new_labels)
            new_input_ids_list.append(cur_new_input_ids)

        tokenizer_model_max_length = getattr(self.config, "tokenizer_model_max_length", None)
        if tokenizer_model_max_length is not None:
            new_input_embeds = [x[:tokenizer_model_max_length] for x in new_input_embeds]
            new_labels = [x[:tokenizer_model_max_length] for x in new_labels]
            new_input_ids_list = [x[:tokenizer_model_max_length] for x in new_input_ids_list]

        max_len = max(x.shape[0] for x in new_input_embeds)
        batch_size = len(new_input_embeds)

        new_input_embeds_padded = []
        new_labels_padded = torch.full((batch_size, max_len), IGNORE_INDEX, dtype=new_labels[0].dtype, device=new_labels[0].device)
        new_input_ids_padded = torch.full((batch_size, max_len), EVENT_PAD_INDEX, dtype=torch.long, device=new_input_ids_list[0].device)
        attention_mask = torch.zeros((batch_size, max_len), dtype=attention_mask.dtype, device=attention_mask.device)
        position_ids = torch.zeros((batch_size, max_len), dtype=position_ids.dtype, device=position_ids.device)

        for i, (emb_i, lab_i, ids_i) in enumerate(zip(new_input_embeds, new_labels, new_input_ids_list)):
            cur_len = emb_i.shape[0]
            if getattr(self.config, "tokenizer_padding_side", "right") == "left":
                new_input_embeds_padded.append(torch.cat((torch.zeros((max_len - cur_len, emb_i.shape[1]), dtype=emb_i.dtype, device=emb_i.device), emb_i), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, -cur_len:] = lab_i
                    new_input_ids_padded[i, -cur_len:] = ids_i
                    attention_mask[i, -cur_len:] = True
                    position_ids[i, -cur_len:] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)
            else:
                new_input_embeds_padded.append(torch.cat((emb_i, torch.zeros((max_len - cur_len, emb_i.shape[1]), dtype=emb_i.dtype, device=emb_i.device)), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, :cur_len] = lab_i
                    new_input_ids_padded[i, :cur_len] = ids_i
                    attention_mask[i, :cur_len] = True
                    position_ids[i, :cur_len] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)

        new_input_embeds = torch.stack(new_input_embeds_padded, dim=0)

        if _labels is None:
            new_labels = None
        else:
            new_labels = new_labels_padded

        if _attention_mask is None:
            attention_mask = None
        else:
            attention_mask = attention_mask.to(dtype=_attention_mask.dtype)

        if _position_ids is None:
            position_ids = None
        if getattr(self.config, "use_pos_skipping", False) and self.training:
            position_ids = torch.arange(new_input_embeds.size(1), device=new_input_embeds.device).unsqueeze(0).to(new_input_embeds.device)
            split_position = random.randint(0, new_input_embeds.size(1))
            left_add = random.randint(0, self.config.pos_skipping_range)
            right_add = random.randint(left_add, self.config.pos_skipping_range)
            position_ids[:, :split_position] += left_add
            position_ids[:, split_position:] += right_add

        return None, position_ids, attention_mask, past_key_values, new_input_embeds, new_labels, new_input_ids_padded
 
  
    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        event_tensors: Optional[torch.Tensor] = None,
        event_image_sizes: Optional[torch.Tensor] = None,
        event_data=None,
        event_feature = None,
        point_cloud_file = None,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")
        
        if event_tensors is not None or event_data is not None:
            (
                inputs,
                position_ids,
                attention_mask,
                _,
                inputs_embeds,
                _,
                new_input_ids
            ) = self.prepare_inputs_labels_for_multimodal(
                inputs,
                position_ids,
                attention_mask,
                None,
                None,
                event_tensors,
                point_cloud_file
            )
        else:
            raise NotImplementedError("please input Event")
        
        return super().generate(
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            **kwargs
        )

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None,
                                      inputs_embeds=None, **kwargs):
        event_tensors = kwargs.pop("event_tensors", None)
        inputs = super().prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs
        )
        if event_tensors is not None:
            inputs['event_tensors'] = event_tensors
        return inputs
    
              
AutoConfig.register("eventgpt_plus_qwen", EventGPTPlusQwenConfig)
AutoModelForCausalLM.register(EventGPTPlusQwenConfig, EventGPTPlusQwenCausalLM)