---
library_name: peft
license: other
base_model: /hpc2hdd/home/jjiang472/OpenSource/Models/Meta-Llama-3-8B
tags:
- llama-factory
- lora
- generated_from_trainer
model-index:
- name: sft-instruct
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# sft-instruct

This model is a fine-tuned version of [/hpc2hdd/home/jjiang472/OpenSource/Models/Meta-Llama-3-8B](https://huggingface.co//hpc2hdd/home/jjiang472/OpenSource/Models/Meta-Llama-3-8B) on the wizardlm_evol_instruct dataset.
It achieves the following results on the evaluation set:
- Loss: 0.6121

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
| 0.6206        | 0.2486 | 500  | 0.6291          |
| 0.616         | 0.4973 | 1000 | 0.6191          |
| 0.6021        | 0.7459 | 1500 | 0.6129          |
| 0.587         | 0.9945 | 2000 | 0.6091          |
| 0.558         | 1.2432 | 2500 | 0.6109          |
| 0.5517        | 1.4918 | 3000 | 0.6085          |
| 0.5544        | 1.7404 | 3500 | 0.6066          |
| 0.5549        | 1.9891 | 4000 | 0.6037          |
| 0.5117        | 2.2377 | 4500 | 0.6129          |
| 0.5226        | 2.4863 | 5000 | 0.6128          |
| 0.5173        | 2.7350 | 5500 | 0.6119          |
| 0.4824        | 2.9836 | 6000 | 0.6121          |


### Framework versions

- PEFT 0.12.0
- Transformers 4.46.1
- Pytorch 2.5.1+cu118
- Datasets 3.1.0
- Tokenizers 0.20.3