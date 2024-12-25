# instruction
llamafactory-cli train codeup/llama3_lora_sft_instruct.yaml
llamafactory-cli export codeup/llama3_lora_sft_merge_instruct.yaml

# math reasoning
llamafactory-cli train codeup/llama3_lora_sft_math.yaml
llamafactory-cli export codeup/llama3_lora_sft_merge_math.yaml

# code reasoning
llamafactory-cli train codeup/llama3_lora_sft_code.yaml
llamafactory-cli export codeup/llama3_lora_sft_merge_code.yaml