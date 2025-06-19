accelerate launch --main_process_port 29521 inference.py \
    --num_inference_steps 120 \
    --output_dir "./output" \
    --data_dir "./data" \
    --test_batch_size 8 --guidance_scale 2.0 \
    --txt_name "example" \
    --pretrained_model_name_or_path "LLanv/AssetDropper" \
    --seed 42