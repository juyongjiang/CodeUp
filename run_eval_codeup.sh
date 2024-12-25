cd evaluation

# HumanEval and HumanEval+
MODEL="./models/codeup_llama3_lora_sft_code"
DATASET=humaneval 
SAVE_PATH=evalplus-$(basename $MODEL)-$DATASET.jsonl
CUDA_VISIBLE_DEVICES=0 python -m evaluation.text2code_vllm \
    --model_key $MODEL \
    --dataset $DATASET \
    --save_path $SAVE_PATH \
    --n_samples_per_problem 1 \
    --max_new_tokens 1024 \
    --top_p 1.0 \
    --temperature 0.0 

python -m evalplus.evaluate --dataset $DATASET --samples $SAVE_PATH 2>&1 | tee evalplus-$(basename $MODEL)-$DATASET.log


# MBPP and MBPP+
MODEL="./models/codeup_llama3_lora_sft_code"
DATASET=mbpp 
SAVE_PATH=evalplus-$(basename $MODEL)-$DATASET.jsonl
CUDA_VISIBLE_DEVICES=0 python -m evaluation.text2code_vllm \
    --model_key $MODEL \
    --dataset $DATASET \
    --save_path $SAVE_PATH \
    --n_samples_per_problem 1 \
    --max_new_tokens 1024 \
    --top_p 1.0 \
    --temperature 0.0 

python -m evalplus.evaluate --dataset $DATASET --samples $SAVE_PATH 2>&1 | tee evalplus-$(basename $MODEL)-$DATASET.log

