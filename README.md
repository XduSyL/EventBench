# EventBench

EventBench: Towards Comprehensive Benchmarking of Event-based MLLMs

[[Paper](https://arxiv.org/abs/2511.18448)]


## Model & Dataset

- **Model**: Download from [EventGPT-Plus-2B](https://huggingface.co/XduSyL/EventGPT-Plus-2B)
- **Benchmark Dataset**: Download from [EventBench](https://huggingface.co/datasets/XduSyL/EventBench)

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

### Running Inference

Edit [script/predict.sh](script/predict.sh) and run:

```bash
bash script/predict.sh
```

### Required Parameters

| Parameter | Description | Example |
|-----------|-------------|---------|
| `--model_path` | Path to the EventGPT-Plus model | `/path/to/EventGPT-Plus-2B` |
| `--model_type` | Model backbone type | `qwen` or `llama` |
| `--chat_template` | Chat template to use | `eventgpt_qwen` |
| `--event_data` | Path to event data file (.npz) | `/path/to/event_data.npz` |
| `--event_data_type` | Type of event data | `v2e` |
| `--event_size_cfg` | Path to event size config YAML | `/path/to/event_size_type.yaml` |
| `--query` | Question to ask the model | `"How does the child move across the tiled floor?"` |

### Optional Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--use_npz` | Use npz format for event data | False |
| `--use_preprocess` | Use preprocessed event data | False |
| `--compute_ttft` | Compute Time to First Token | False |
| `--temperature` | Sampling temperature | 0.3 |
| `--top_p` | Top-p sampling threshold | 1.0 |
| `--num_beams` | Number of beams for beam search | 1 |
| `--max_new_tokens` | Maximum tokens to generate | 512 |
| `--context_max_len` | Maximum context length | 1024 |
| `--num_bins_list` | List of event bin counts | [4, 8, 16, 32] |

### Example Command

```bash
python inference_eventgpt_plus.py \
    --model_path /path/to/EventGPT-Plus-2B \
    --model_type qwen \
    --use_npz \
    --use_preprocess \
    --event_data_type v2e \
    --chat_template eventgpt_qwen \
    --event_data /path/to/event_data.npz \
    --event_size_cfg /path/to/event_size_type.yaml \
    --query "How does the child move across the tiled floor?"
```

## Roadmap

- 🔜 **Coming Soon**: Task-specific evaluation scripts
- 🚀 **Mar 2026**: Codebase released
- 📦 **Nov 2025**: EventGPT-Plus-2B model released
- 📊 **Nov 2025**: EventBench benchmark dataset released
- 📄 **Nov 2025**: Paper released on arXiv


## Citation

If you use EventBench in your research, please cite:

```bibtex
@article{liu2025eventbench,
  title={EventBench: Towards Comprehensive Benchmarking of Event-based MLLMs},
  author={Liu, Shaoyu and Li, Jianing and Zhao, Guanghui and Zhang, Yunjian and Ji, Xiangyang},
  journal={arXiv preprint arXiv:2511.18448},
  year={2025}
}
```
