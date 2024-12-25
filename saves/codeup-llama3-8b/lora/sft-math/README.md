---
library_name: peft
license: other
base_model: models/codeup_llama3_lora_sft_instruct
tags:
- llama-factory
- lora
- generated_from_trainer
model-index:
- name: sft-math
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# sft-math

This model is a fine-tuned version of [models/codeup_llama3_lora_sft_instruct](https://huggingface.co/models/codeup_llama3_lora_sft_instruct) on the metamathqa dataset.
It achieves the following results on the evaluation set:
- Loss: 0.1839

## Model description

More information needed

## Intended uses & limitations

More information needed

## Training and evaluation data

More information needed

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 0.0001
- train_batch_size: 4
- eval_batch_size: 4
- seed: 42
- distributed_type: multi-GPU
- num_devices: 8
- gradient_accumulation_steps: 2
- total_train_batch_size: 64
- total_eval_batch_size: 32
- optimizer: Use OptimizerNames.ADAMW_TORCH with betas=(0.9,0.999) and epsilon=1e-08 and optimizer_args=No additional optimizer arguments
- lr_scheduler_type: cosine
- lr_scheduler_warmup_ratio: 0.1
- num_epochs: 3.0

### Training results

| Training Loss | Epoch  | Step  | Validation Loss |
|:-------------:|:------:|:-----:|:---------------:|
| 0.2582        | 0.0900 | 500   | 0.2655          |
| 0.2403        | 0.1800 | 1000  | 0.2422          |
| 0.237         | 0.2700 | 1500  | 0.2313          |
| 0.2206        | 0.3600 | 2000  | 0.2237          |
| 0.2176        | 0.4500 | 2500  | 0.2170          |
| 0.2123        | 0.5401 | 3000  | 0.2129          |
| 0.2043        | 0.6301 | 3500  | 0.2094          |
| 0.2049        | 0.7201 | 4000  | 0.2067          |
| 0.2037        | 0.8101 | 4500  | 0.2035          |
| 0.1989        | 0.9001 | 5000  | 0.2016          |
| 0.1978        | 0.9901 | 5500  | 0.1993          |
| 0.1799        | 1.0801 | 6000  | 0.1984          |
| 0.1847        | 1.1701 | 6500  | 0.1971          |
| 0.1786        | 1.2601 | 7000  | 0.1959          |
| 0.176         | 1.3501 | 7500  | 0.1940          |
| 0.1753        | 1.4401 | 8000  | 0.1926          |
| 0.1761        | 1.5302 | 8500  | 0.1909          |
| 0.1765        | 1.6202 | 9000  | 0.1896          |
| 0.1688        | 1.7102 | 9500  | 0.1881          |
| 0.1754        | 1.8002 | 10000 | 0.1868          |
| 0.1743        | 1.8902 | 10500 | 0.1855          |
| 0.1749        | 1.9802 | 11000 | 0.1844          |
| 0.153         | 2.0702 | 11500 | 0.1874          |
| 0.1571        | 2.1602 | 12000 | 0.1874          |
| 0.1524        | 2.2502 | 12500 | 0.1863          |
| 0.1532        | 2.3402 | 13000 | 0.1860          |
| 0.1524        | 2.4302 | 13500 | 0.1854          |
| 0.1538        | 2.5203 | 14000 | 0.1848          |
| 0.1462        | 2.6103 | 14500 | 0.1844          |
| 0.1525        | 2.7003 | 15000 | 0.1842          |
| 0.1524        | 2.7903 | 15500 | 0.1840          |
| 0.1551        | 2.8803 | 16000 | 0.1839          |
| 0.1496        | 2.9703 | 16500 | 0.1839          |


### Framework versions

- PEFT 0.12.0
- Transformers 4.46.1
- Pytorch 2.5.1+cu118
- Datasets 3.1.0
- Tokenizers 0.20.3