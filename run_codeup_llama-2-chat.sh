python finetune.py \
    --base_model='meta-llama/Llama-2-7b-chat-hf' \
    --data_path='data/codeup_19k.json' \
    --num_epochs=15 \
    --cutoff_len=512 \
    --group_by_length \
    --output_dir='./codeup-peft-llama-2-chat' \
    --lora_target_modules='[q_proj,k_proj,v_proj,o_proj]' \
    --lora_r=16 \
    --micro_batch_size=4 