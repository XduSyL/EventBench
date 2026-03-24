import numpy as np
from dataset.data_processor import split_event_by_n_segments
from decord import VideoReader, cpu
import decord.logging
decord.logging.set_level(decord.logging.ERROR)
import numpy as np

"""fuction: Dynamic event bin selector
Input: raw event stream
Output: event stream with fixed bins
"""

def event_bin_selector(event_npy, t_span, num_bins_list):
    if t_span <= 100000:
        num_bins = num_bins_list[0]
        event_bins = split_event_by_n_segments(event_npy, num_bins)
    elif t_span >100000 and t_span <= 4000000:
        num_bins = num_bins_list[1]
        event_bins = dynamic_event_bin_selection(event_npy, num_bins, 25)
    elif t_span > 4000000 and t_span <= 16000000:
        num_bins = num_bins_list[2]
        event_bins = dynamic_event_bin_selection(event_npy, num_bins, 25)
    elif t_span > 16000000:
        num_bins = num_bins_list[3]
        event_bins = dynamic_event_bin_selection(event_npy, num_bins, 25)        
    else:
        raise ValueError("Invalid t_span")
    
    return event_bins
    

def to_structured_array(event_dict):
    n = len(event_dict['t'])
    dtype = [('p', np.uint8), ('t', event_dict['t'].dtype),
             ('x', np.uint16), ('y', np.uint16)]
    structured_arr = np.empty(n, dtype=dtype)
    structured_arr['p'] = event_dict['p']
    structured_arr['t'] = event_dict['t']
    structured_arr['x'] = event_dict['x']
    structured_arr['y'] = event_dict['y']
    return structured_arr


def fast_find_dynamic_window(times, window_ms):
    t0 = times[0]
    times_ms = ((times - t0) / 1000).astype(np.int64)  
    hist_len = times_ms[-1] + 1
    hist = np.zeros(hist_len + 1, dtype=np.int32)
    np.add.at(hist, times_ms, 1)
    kernel = np.ones(window_ms, dtype=np.int32)
    conv = np.convolve(hist, kernel, mode='valid')
    start_idx_ms = np.argmax(conv)
    mask = (times_ms >= start_idx_ms) & (times_ms < start_idx_ms + window_ms)
    return mask


def dynamic_event_bin_selection(event_stream, num_bins=16, window_ms=25):
    event_stream = to_structured_array(event_stream)
    times_all = event_stream['t']
    sorted_idx = np.argsort(times_all)
    event_stream = event_stream[sorted_idx]
    times_all = times_all[sorted_idx]
    
    t_min, t_max = times_all[0], times_all[-1]
    total_duration = t_max - t_min
    fixed_bins = []

    for k in range(num_bins):
        bin_start = t_min + k * total_duration / num_bins
        bin_end = t_min + (k + 1) * total_duration / num_bins

        left_idx = np.searchsorted(times_all, bin_start, side='left')
        right_idx = np.searchsorted(times_all, bin_end, side='right')

        if right_idx <= left_idx:
            fixed_bins.append(np.empty(0, dtype=event_stream.dtype))
            continue

        events_bin = event_stream[left_idx:right_idx]
        times_bin = events_bin['t']

        if len(times_bin) == 0:
            fixed_bins.append(np.empty(0, dtype=event_stream.dtype))
            continue

        mask = fast_find_dynamic_window(times_bin, window_ms)
        selected_events = events_bin[mask]
        fixed_bins.append(selected_events)

    return fixed_bins


def process_video_with_decord(video_file, data_args):
    vr = VideoReader(video_file, ctx=cpu(0), num_threads=1)
    total_frame_num = len(vr)
    fps = vr.get_avg_fps()
    video_time = total_frame_num / fps

    num_frames_to_sample = data_args.frames_upbound if data_args.frames_upbound > 0 else 32
    frame_idx = np.linspace(0, total_frame_num - 1, num_frames_to_sample, dtype=int).tolist()
    frame_time = [i / fps for i in frame_idx]

    video = vr.get_batch(frame_idx).asnumpy()
    frame_time_str = ",".join([f"{t:.2f}s" for t in frame_time])

    vr.seek(0)
    return video, video_time, frame_time_str, num_frames_to_sample