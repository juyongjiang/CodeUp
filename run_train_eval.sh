# Fine-tuning
## multi-gpus
WORLD_SIZE=8 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 --master_port=3166 finetune.py \
    --base_model='/hpc2hdd/home/jjiang472/OpenSource/Models/Meta-Llama-3-8B' \
    --data_path='data/codeup_19k.json' \
    --output_dir='codeup-peft-llama-3-8b' \
    --batch_size=128 \
    --micro_batch_size=4 \
    --num_epochs=10 \
    --learning_rate=3e-4 \
    --val_set_size=2000 \
    --cutoff_len=1024 \
    --lora_r=16 \
    --lora_alpha=32 \
    --lora_dropout=0.05 \
    --lora_target_modules='[q_proj,k_proj,v_proj,o_proj]' \
    --train_on_inputs \
    --group_by_length \
    --resume_from_checkpoint='/hpc2hdd/home/jjiang472/OpenSource/Models/Meta-Llama-3-8B' \
    --prompt_template_name='alpaca' 2>&1 | tee train.log
  
## single gpu
python finetune.py \
    --base_model='/hpc2hdd/home/jjiang472/OpenSource/Models/Meta-Llama-3-8B' \
    --data_path='data/codeup_19k.json' \
    --output_dir='codeup-peft-llama-3-8b' \
    --batch_size=128 \
    --micro_batch_size=4 \
    --num_epochs=10 \
    --learning_rate=3e-4 \
    --val_set_size=2000 \
    --cutoff_len=1024 \
    --lora_r=16 \
    --lora_alpha=32 \
    --lora_dropout=0.05 \
    --lora_target_modules='[q_proj,k_proj,v_proj,o_proj]' \
    --train_on_inputs \
    --group_by_length \
    --resume_from_checkpoint='/hpc2hdd/home/jjiang472/OpenSource/Models/Meta-Llama-3-8B' \
    --prompt_template_name='alpaca' 2>&1 | tee train.log


# Evaluation
cd bigcode-evaluation-harness
pip install -e .

CUDA_VISIBLE_DEVICES=0 accelerate launch  main.py \
  --model /hpc2hdd/home/jjiang472/CodeUp/codeup-peft-llama-3-8b/merged \
  --tasks multiple-py \
  --max_length_generation 1024 \
  --temperature 0.8 \
  --do_sample True \
  --n_samples 200 \
  --batch_size 200 \
  --generation_only \
  --save_generations \
  --save_generations_path generations_multiple-py.json 2>&1 | tee ../eval.log


CUDA_VISIBLE_DEVICES=0 accelerate launch  main.py \
  --model /hpc2hdd/home/jjiang472/CodeUp/codeup-peft-llama-3-8b/merged \
  --max_length_generation 1024 \
  --tasks mbpp \
  --temperature 0.1 \
  --n_samples 15 \
  --batch_size 10 \
  --allow_code_execution \
  --save_generations \
  --save_generations_path generations_mbpp.json 2>&1 | tee ../eval.log