python finetune.py \
    --base_model='decapoda-research/llama-7b-hf' \
    --data_path='data/codeup_19k.json' \
    --num_epochs=10 \
    --cutoff_len=512 \
    --group_by_length \
    --output_dir='./codeup-peft' \
    --lora_target_modules='[q_proj,k_proj,v_proj,o_proj]' \
    --lora_r=16 \
    --micro_batch_size=16

python export_checkpoint.py \
    --base_model='decapoda-research/llama-7b-hf' \
    --lora_weights='codeup-peft' \
    --lora_target_modules='[q_proj,k_proj,v_proj,o_proj]' \
    --export_dir='export_checkpoint/7b' \
    --checkpoint_type='hf'