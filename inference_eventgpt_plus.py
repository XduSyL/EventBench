import torch
from model.eventgpt_plus_qwen import EventGPTPlusQwenConfig, EventGPTPlusQwenModel, EventGPTPlusQwenCausalLM
from utils.constents import DEFAULT_EVENT_PATCH_TOKEN, DEFAULT_EV_START_TOKEN, DEFAULT_EV_END_TOKEN, DEFAULT_EVENT_TOKEN, EVENT_TOKEN_INDEX
from transformers import AutoConfig, AutoTokenizer
from utils.bin_selector import event_bin_selector
from dataset.data_processor import generate_event_tensor
from dataset.conversation import conv_templates
import argparse
import numpy as np
import time
import cv2
import yaml
import os

def load_model(args):
    if args.model_type == "qwen":
        config = AutoConfig.from_pretrained(args.model_path)
        # if args.pretrained_event_tower:
        # config.pretrained_event_tower = args.pretrained_event_tower
        tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=False)
        model = EventGPTPlusQwenCausalLM.from_pretrained(args.model_path,
                                                       torch_dtype=torch.bfloat16, 
                                                       config=config)
    elif args.model_type == "llama":
        config = AutoConfig.from_pretrained(args.model_path)
        # if args.pretrained_event_tower:
        config.pretrained_event_tower = args.pretrained_event_tower
        tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=False)
        model = EventGPTPlusLLaMACausalLM.from_pretrained(args.model_path,
                                                       attn_implementation="flash_attention_2",
                                                       torch_dtype=torch.bfloat16, 
                                                       config=config)
    else:
        raise ValueError(f"Invalid model type: {model_type}")
   
    return model, tokenizer

def process_event_data_use_preprocess(event_data_path, event_processor, args):
    event_data_path = os.path.splitext(event_data_path)[0] + ".npz"
    event_npy = np.load(event_data_path, allow_pickle=True)
    event_bins = event_npy['event_bins']
    event_tensors = []
    i = 0
    for event_bin in event_bins:
        event_data_type = args.event_data_type
        with open(args.event_size_cfg, 'r') as f:
            config = yaml.safe_load(f)               
        ev_height = config['data_type'][event_data_type]['ev_height']
        ev_width = config['data_type'][event_data_type]['ev_width']   
        try:     
            event_tensor = generate_event_tensor(event_bin['x'], event_bin['y'], event_bin['p'], 
                                                ev_height, ev_width)
        except Exception as e:
            print(f"[ERROR] Failed to process file: {event_data_path}")
            print(f"        Reason: {e}")
            raise
        event_tensor = event_processor.preprocess(event_tensor, return_tensors="pt")["pixel_values"]
        event_tensors.append(event_tensor)
    event_tensors = torch.cat(event_tensors, dim=0)
    return event_tensors

def npz_to_npy(data_path):
    try:
        data = np.load(data_path)
    except:
        data = np.load(data_path, allow_pickle=True)
    if 'event_data' in data.files:
        arr = data['event_data']
        try:
            x, y, t, p = arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3]
        except:
            x, y, t, p = arr['x'], arr['y'], arr['t'], arr['p']           
    else:
        x, y, t, p = data['x'], data['y'], data['t'], data['p']

    event_dict = {
        'p': p.astype(np.uint8,  copy=False),
        'x': x.astype(np.uint16, copy=False),
        'y': y.astype(np.uint16, copy=False),
        't': t.astype(np.int64,  copy=False),
    }
    return event_dict

def tokenizer_event_token(prompt, tokenizer, event_token_index=EVENT_TOKEN_INDEX, return_tensors=None):
    prompt_chunks = [tokenizer(chunk).input_ids for chunk in prompt.split('<event>')]

    def insert_separator(X, sep):
        return [ele for sublist in zip(X, [sep] * len(X)) for ele in sublist][:-1]

    input_ids = []
    offset = 0
    if len(prompt_chunks) > 0 and len(prompt_chunks[0]) > 0 and prompt_chunks[0][0] == tokenizer.bos_token_id:
        offset = 1
        input_ids.append(prompt_chunks[0][0])

    for x in insert_separator(prompt_chunks, [event_token_index] * (offset + 1)):
        input_ids.extend(x[offset:])

    if return_tensors is not None:
        if return_tensors == 'pt':
            return torch.tensor(input_ids, dtype=torch.long)
        raise ValueError(f'Unsupported tensor type: {return_tensors}')
    return input_ids

def process_event_data(event_data, num_bins_list, event_processor, args):
    timestamps = event_data['t']
    t_min, t_max = timestamps.min(), timestamps.max()
    t_span = t_max - t_min
    event_bins = event_bin_selector(event_data, t_span, num_bins_list)
    event_tensors = []
    i = 0
    for event_bin in event_bins:
        event_data_type = event_data['data_type']
        with open(args.event_size_cfg, 'r') as f:
            config = yaml.safe_load(f)               
        ev_height = config['data_type'][event_data_type]['ev_height']
        ev_width = config['data_type'][event_data_type]['ev_width']              
        event_tensor = generate_event_tensor(event_bin['x'], event_bin['y'], event_bin['p'], 
                                            ev_height, ev_width)
        # save event_tensor as png
        # cv2.imwrite(f"/mnt/data2/SyL/EventGPT-V2/event_tensor/event_tensor_{i}.png", event_tensor)  
        # i += 1
        event_tensor = event_processor.preprocess(event_tensor, return_tensors="pt")["pixel_values"]
        event_tensors.append(event_tensor)
    event_tensors = torch.cat(event_tensors, dim=0)
    return event_tensors

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--model_type", type=str, required=True)
    parser.add_argument("--chat_template", type=str, required=True)
    parser.add_argument("--event_data", type=str, required=True)
    parser.add_argument("--compute_ttft", action="store_true", help="Enable TTFT computation")
    parser.add_argument("--event_data_type", type=str, required=True)
    parser.add_argument("--pretrain_event_projector", type=str, default='')
    parser.add_argument("--pretrained_event_tower", type=str, default='')
    parser.add_argument("--load_pretrain_event_projector", action="store_true", help="Load pretrain event_projector")
    parser.add_argument("--num_bins_list", type=list, default=[4, 8, 16, 32])
    parser.add_argument("--event_bin_size", type=int, default=[240, 320])
    parser.add_argument("--context_max_len", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument("--top_p", type=float, default=1)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--event_size_cfg", type=str, required=True)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--query", type=str, required=True)
    parser.add_argument("--use_npz", action="store_true", help="Use preprocess")
    parser.add_argument("--use_preprocess", action="store_true", help="Use preprocess")

    args = parser.parse_args()
    
    model, tokenizer = load_model(args)
    event_processor = None
    
    if args.load_pretrain_event_projector:
        pretrain_event_projector = args.pretrain_event_projector
        print("Loading event_projector pretrain weights...")
        pretrained_weights = torch.load(pretrain_event_projector)
        # Adjust keys to match model structure
        pretrained_weights = {k.replace("model.event_projector.", ""): v for k, v in pretrained_weights.items()}
        model.get_model().event_projector.load_state_dict(pretrained_weights, strict=True)
        print("Pretrained weights loaded successfully into visual_projector.")

    mm_use_ev_start_end = getattr(model.config, "mm_use_ev_start_end", False)
    mm_use_ev_patch_token = getattr(model.config, "mm_use_ev_patch_token", True)
    if mm_use_ev_patch_token:
        tokenizer.add_tokens([DEFAULT_EVENT_PATCH_TOKEN], special_tokens=True)
    if mm_use_ev_start_end:
        tokenizer.add_tokens([DEFAULT_EV_START_TOKEN, DEFAULT_EV_END_TOKEN], special_tokens=True)
    
    if mm_use_ev_patch_token or mm_use_ev_start_end:
        model.resize_token_embeddings(len(tokenizer))
    
    event_tower = model.get_event_tower()
    event_processor = event_tower.event_processor
    
    context_max_len = args.context_max_len 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    qs = args.query
    event_token_se = DEFAULT_EV_START_TOKEN + DEFAULT_EVENT_TOKEN + DEFAULT_EV_END_TOKEN
    qs = DEFAULT_EVENT_TOKEN + "\n" + qs
    
    chat_template = args.chat_template
    conv = conv_templates[chat_template].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    
    if args.use_preprocess:
        event_tensors = process_event_data_use_preprocess(args.event_data, event_processor, args)
    else:
        if args.use_npz:
            event_data = npz_to_npy(args.event_data)
        else:
            event_npy = np.load(args.event_data, allow_pickle=True)
            event_data = event_npy.item()
        event_data['data_type'] = args.event_data_type
        # event_bin_size = args.event_bin_size
        event_tensors = process_event_data(event_data, args.num_bins_list, event_processor, args)

    input_ids = tokenizer_event_token(prompt, tokenizer, EVENT_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(model.device)

    if args.compute_ttft == True:
        args.max_new_tokens = 1
    start_time = time.time()
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            event_tensors=event_tensors,
            do_sample=True if args.temperature > 0 else False,
            temperature=args.temperature,
            top_p=args.top_p,
            num_beams=args.num_beams,
            max_new_tokens=args.max_new_tokens,
            use_cache=True
        )
    end_time = time.time()
    if args.compute_ttft:
        TTFT = end_time - start_time
        print(f"Time taken for inference: {TTFT:.2f} seconds")

    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
    print(outputs)  

