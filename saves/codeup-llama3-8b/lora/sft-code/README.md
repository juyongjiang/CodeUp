---
library_name: peft
license: other
base_model: models/codeup_llama3_lora_sft_math
tags:
- llama-factory
- lora
- generated_from_trainer
model-index:
- name: sft-code
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# sft-code

This model is a fine-tuned version of [models/codeup_llama3_lora_sft_math](https://huggingface.co/models/codeup_llama3_lora_sft_math) on the evol_codealpaca dataset.
It achieves the following results on the evaluation set:
- Loss: 0.9684

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

| Training Loss | Epoch  | Step | Validation Loss |
|:-------------:|:------:|:----:|:---------------:|
| 1.0211        | 0.3195 | 500  | 0.9954          |
| 0.9573        | 0.6390 | 1000 | 0.9791          |
| 0.9683        | 0.9585 | 1500 | 0.9708          |
| 0.9564        | 1.2780 | 2000 | 0.9695          |
| 0.9047        | 1.5974 | 2500 | 0.9657          |
| 0.8991        | 1.9169 | 3000 | 0.9616          |
| 0.8625        | 2.2364 | 3500 | 0.9700          |
| 0.8846        | 2.5559 | 4000 | 0.9687          |
| 0.872         | 2.8754 | 4500 | 0.9685          |


### Framework versions

- PEFT 0.12.0
- Transformers 4.46.1
- Pytorch 2.5.1+cu118
- Datasets 3.1.0
- Tokenizers 0.20.3