export CUDA_VISIBLE_DEVICES=0

python inference_eventgpt_plus.py \
    --model_path /root/private_data/SyL/EventGPT-V2/checkpoints/total/eventgptv2-7b-sft \
    --model_type qwen \
    --use_npz \
    --use_preprocess \
    --event_data_type v2e  \
    --chat_template eventgpt_qwen \
    --event_data /root/private_data/EventBench/sample/5SjV5j8f6rw_000009_000019.npz \
    --event_size_cfg /root/private_data/SyL/EventGPT-V2/script/event_size_type.yaml \
    --query "How does the child move across the tiled floor?"

    # --compute_ttft \