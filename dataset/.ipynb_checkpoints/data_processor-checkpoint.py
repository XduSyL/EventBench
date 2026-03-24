import numpy as np
from typing import Dict, Sequence
from utils.constents import DEFAULT_EVENT_TOKEN, DEFAULT_EV_START_TOKEN, DEFAULT_EV_END_TOKEN
from argument import DataArguments
import dataset.conversation as conversation_lib

def split_event_by_time(event_npy, time_interval=25000):
    # Extract data from the event_npy dictionary
    p = event_npy['p']
    t = event_npy['t']
    x = event_npy['x']
    y = event_npy['y']

    # Calculate the time bin for each timestamp based on the given time interval
    time_bins = (t // time_interval) * time_interval

    # Get the unique time bins (intervals)
    unique_bins = np.unique(time_bins)

    # Split the data according to the time bins
    split_data = [
        {
            'p': p[time_bins == bin],
            't': t[time_bins == bin],
            'x': x[time_bins == bin],
            'y': y[time_bins == bin]
        }
        for bin in unique_bins
    ]

    return split_data


def split_event_by_n_segments(event_npy, n_segments):
    p = event_npy['p']
    t = event_npy['t']
    x = event_npy['x']
    y = event_npy['y']

    t_min = t.min()
    t_max = t.max()

    time_bounds = np.linspace(t_min, t_max, n_segments + 1)

    split_data = []
    for i in range(n_segments):
        mask = (t >= time_bounds[i]) & (t < time_bounds[i + 1])
        split_data.append({
            'p': p[mask],
            't': t[mask],
            'x': x[mask],
            'y': y[mask]
        })

    return split_data

def generate_event_tensor(x, y, p, height, width):  
    event_tensor = np.ones((height, width, 3), dtype=np.uint8) * 255

    blue_mask = p == 0
    red_mask = p == 1

    event_tensor[y[blue_mask], x[blue_mask]] = np.array([0, 0, 255])  # blue
    event_tensor[y[red_mask], x[red_mask]] = np.array([255, 0, 0])  # red

    return event_tensor

def get_event_tensor_list(event_npy, n, height, width):
    x, y, p, t = event_npy['x'], event_npy['y'], event_npy['p'], event_npy['t']

    total_events = len(t)
    events_per_image = total_events // n  

    event_tensor_list = [] 

    for i in range(n):
        start_idx = i * events_per_image
        end_idx = (i + 1) * events_per_image if i < n - 1 else total_events  

        x_part = x[start_idx:end_idx]
        y_part = y[start_idx:end_idx]
        p_part = p[start_idx:end_idx]

        event_tensor = generate_event_tensor(x_part, y_part, p_part, height, width)
        
        event_tensor_list.append(event_tensor) 

    return event_tensor_list  

def npz_to_npy(data_path):
    data = np.load(data_path)
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

def preprocess_multimodal(
    sources: Sequence[str],
    data_args: DataArguments
) -> Dict:
    is_multimodal = data_args.is_multimodal
    if not is_multimodal:
        return sources

    for source in sources:
        for sentence in source:
            if DEFAULT_EVENT_TOKEN in sentence['value']:
                sentence['value'] = sentence['value'].replace(DEFAULT_EVENT_TOKEN, '').strip()
                sentence['value'] = DEFAULT_EVENT_TOKEN + '\n' + sentence['value']
                sentence['value'] = sentence['value'].strip()
                if "mmtag" in conversation_lib.default_conversation.version:
                    sentence['value'] = sentence['value'].replace(DEFAULT_EVENT_TOKEN, '<Event>' + DEFAULT_EVENT_TOKEN + '</Event>')
            replace_token = DEFAULT_EVENT_TOKEN
            if data_args.mm_use_ev_start_end: 
                replace_token = DEFAULT_EV_START_TOKEN + replace_token + DEFAULT_EV_END_TOKEN
            sentence["value"] = sentence["value"].replace(DEFAULT_EVENT_TOKEN, replace_token)

    return sources

