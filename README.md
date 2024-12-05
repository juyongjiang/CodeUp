<p align="center" width="100%">
<img src="assets/codeup_logo.jpeg" alt="HKUST CodeUp" style="width: 50%; min-width: 250px; display: block; margin: auto;">
</p>

# CodeUp: A Multilingual Code Generation Llama-X Model with Parameter-Efficient Instruction-Tuning

[![Code License](https://img.shields.io/badge/Code%20License-Apache_2.0-green.svg)](https://github.com/juyongjiang/CodeUp/blob/master/LICENSE)
[![Data License](https://img.shields.io/badge/Data%20License-CC%20By%20NC%204.0-red.svg)](https://github.com/juyongjiang/CodeUp/blob/master/data/DATA_LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/release/python-390/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

## Table of Contents
- [CodeUp: A Multilingual Code Generation Llama-X Model with Parameter-Efficient Instruction-Tuning](#codeup-a-multilingual-code-generation-llama-x-model-with-parameter-efficient-instruction-tuning)
  - [Table of Contents](#table-of-contents)
  - [Introduction](#introduction)
  - [Prompt Template](#prompt-template)
  - [Training (`finetune.py`)](#training-finetunepy)
  - [Inference (`generate.py`)](#inference-generatepy)
  - [Evaluation](#evaluation)
    - [Setup](#setup)
    - [Usage](#usage)
  - [Useful Resources](#useful-resources)
    - [LLMs](#llms)
    - [CPU Running](#cpu-running)
    - [Interface](#interface)
    - [Dataset](#dataset)
    - [Evaluation](#evaluation-1)
    - [Hugging Face](#hugging-face)
    - [Papers](#papers)

> [!IMPORTANT]
> 
> If you use the data or code in this repo, please cite the repo.

```
@misc{codeup,
  author = {Juyong Jiang and Sunghun Kim},
  title = {CodeUp: A Multilingual Code Generation Llama-X Model with Parameter-Efficient Instruction-Tuning},
  year = {2023},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/juyongjiang/CodeUp}},
}

@article{jiang2024survey,
  title={A Survey on Large Language Models for Code Generation},
  author={Jiang, Juyong and Wang, Fan and Shen, Jiasi and Kim, Sungju and Kim, Sunghun},
  journal={arXiv preprint arXiv:2406.00515},
  year={2024}
}
```

## Introduction
In recent years, large language models (LLMs) have demonstrated exceptional capabilities across a wide range of applications, largely due to their remarkable emergent abilities. To better align these models with human preferences, techniques such as instruction-tuning and reinforcement learning from human feedback (RLHF) have been developed for chat-based LLMs, including models like ChatGPT and GPT-4. However, except for Codex, these general-purpose LLMs primarily focus on general domains and are not specifically optimized for coding tasks. Codex, while a viable option, is a closed-source model developed by OpenAI. This underscores the need for developing open-source, instruction-following LLMs tailored to the code domain.
The development of such models, however, faces significant challenges due to the extensive number of parameters (≥ 7 billion) and the vast datasets required for training. These factors demand substantial computational resources, which can hinder training and inference on consumer hardware.

To address these challenges, our project leverages the latest powerful foundation model, `Llama` with version `X`, termed `Llama-X`, to construct high-quality instruction-following datasets for code generation tasks. We propose the development of an instruction-following multilingual code generation model based on Llama-X. 
To ensure that our approach is feasible within an academic budget and can be executed on consumer hardware, such as a single RTX 3090, we are inspired by Alpaca-LoRA to integrate advanced parameter-efficient fine-tuning (PEFT) methods like `LoRA` for the code domain. 
These methods facilitate the efficient adaptation of pre-trained language models (PLMs, also known as foundation models) to various downstream applications without the need to fine-tune the entire model's parameters. 
<!-- Our overall training strategy is outlined as follows. -->


## Prompt Template
We follow the previous work to use the following prompts template `templates/alpaca.json` for instruction-tuning the model. However, during inference (e.g., for the web demo), we use the user instruction with an empty input field (second option).

- Non-empty input field:
 ```
 Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.
 
 ### Instruction:
 {instruction}
 
 ### Input:
 {input}
 
 ### Response:
 ```
- Empty input field:
 ```
 Below is an instruction that describes a task. Write a response that appropriately completes the request.
 
 ### Instruction:
 {instruction}
 
 ### Response:
 ```

## Training (`finetune.py`)
To access `Llama-X` model, please follow the [Download Guide](https://github.com/facebookresearch/llama/tree/main#download) and the difference between two versions of LLaMA can be found in [Model Card](https://github.com/facebookresearch/llama/blob/main/MODEL_CARD.md).

To reproduce our fine-tuning runs for CodeUp, first, install the dependencies.

```bash
pip install -r requirements.txt
```

The `finetune.py` file contains a straightforward application of [PEFT](https://github.com/huggingface/peft) to the Llama-X model, as well as some code related to prompt construction and tokenization.

```bash
python finetune.py \
    --base_model='meta-llama/Meta-Llama-3-8B' \
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
    --resume_from_checkpoint='meta-llama/Meta-Llama-3-8B' \
    --prompt_template_name='alpaca'

```

## Inference (`generate.py`)
This file reads the foundation model (i.e., Llama2 7B) from the Hugging Face model hub and the LoRA weights from `codeup-peft-llama-2/7b`, and runs a `Gradio interface` for inference on a specified input. Users should treat this as an example code for using the model and modify it as needed.

```bash
python generate.py \
    --load_8bit \
    --base_model 'meta-llama/Meta-Llama-3-8B' \
    --lora_weights 'codeup-peft-llama-3-8b'
```

## Evaluation
We use the open-source framework [Code Generation LM Evaluation Harness](https://github.com/bigcode-project/bigcode-evaluation-harness) developed by BigCode team to evaluate our CodeUp performance.

### Setup
```bash
git clone https://github.com/bigcode-project/bigcode-evaluation-harness.git
cd bigcode-evaluation-harness
pip install -e .
```
Also make sure you have `git-lfs`` installed (above guide) and are logged in the Hub
```bash
huggingface-cli login
```
### Usage
You can use this evaluation framework to generate text solutions to code benchmarks with any autoregressive model available on [Hugging Face hub](https://huggingface.co/), to evaluate (and execute) the solutions or to do both. While it is better to use GPUs for the generation, the evaluation only requires CPUs. So it might be beneficial to separate these two steps (i.e., `--generation_only` or `--load_generations_path`). By default both generation and evaluation are performed.

For more details on how to evaluate on the various tasks (i.e., benchmark), please refer to the documentation in [`bigcode-evaluation-harness/docs/README.md`](https://github.com/bigcode-project/bigcode-evaluation-harness/blob/main/docs/README.md). 

Below is an example of CodeUp to `generate and evaluate` on a `multiple-py` task (i.e., benchmark), which denotes `HumanEval` benchmark is translated into 18 programming languages.

```bash
accelerate launch  main.py \
  --model deepse/CodeUp-Llama-2-7b-hf \
  --tasks multiple-py \
  --max_length_generation 650 \
  --temperature 0.8 \
  --do_sample True \
  --n_samples 200 \
  --batch_size 200 \
  --allow_code_execution \
  --save_generations
```

* `--model` can be any autoregressive model available on [Hugging Face hub](https://huggingface.co/) can be used, but we recommend using code generation models trained specifically on Code such as [SantaCoder](https://huggingface.co/bigcode/santacoder), [InCoder](https://huggingface.co/facebook/incoder-6B) and [CodeGen](https://huggingface.co/Salesforce/codegen-16B-mono).
* `--tasks` denotes a variety of benchmarks as follows:
    ```bash
    'codexglue_code_to_text-go', 'codexglue_code_to_text-java', 'codexglue_code_to_text-javascript', 'codexglue_code_to_text-php', 'codexglue_code_to_text-python', 'codexglue_code_to_text-python-left', 'codexglue_code_to_text-ruby', 'codexglue_text_to_text-da_en', 'codexglue_text_to_text-lv_en', 'codexglue_text_to_text-no_en', 'codexglue_text_to_text-zh_en', 
    'conala', 
    'concode', 
    'ds1000-all-completion', 'ds1000-all-insertion', 'ds1000-matplotlib-completion', 'ds1000-matplotlib-insertion', 'ds1000-numpy-completion', 'ds1000-numpy-insertion', 'ds1000-pandas-completion', 'ds1000-pandas-insertion', 'ds1000-pytorch-completion', 'ds1000-pytorch-insertion', 'ds1000-scipy-completion', 'ds1000-scipy-insertion', 'ds1000-sklearn-completion', 'ds1000-sklearn-insertion', 'ds1000-tensorflow-completion', 'ds1000-tensorflow-insertion', 
    'humaneval', 'instruct-humaneval', 'instruct-humaneval-nocontext', 
    'mbpp', 
    'multiple-cpp', 'multiple-cs', 'multiple-d', 'multiple-go', 'multiple-java', 'multiple-jl', 'multiple-js', 'multiple-lua', 'multiple-php', 'multiple-pl', 'multiple-py', 'multiple-r', 'multiple-rb', 'multiple-rkt', 'multiple-rs', 'multiple-scala', 'multiple-sh', 'multiple-swift', 'multiple-ts', 
    'pal-gsm8k-greedy', 'pal-gsm8k-majority_voting', 'pal-gsmhard-greedy', 'pal-gsmhard-majority_voting']
    ```
* `--limit` represents the number of problems to solve, if it's not provided, all problems in the benchmark are selected. 
* `--allow_code_execution` is for executing the generated code: it is off by default, read the displayed warning before calling it to enable execution. 
* Some models with custom code on the HF hub like [SantaCoder](https://huggingface.co/bigcode/santacoder) require calling `--trust_remote_code`, for private models add `--use_auth_token`.
* `--save_generations` saves the post-processed generations in a json file at `--save_generations_path` (by default `generations.json`). You can also save references by calling `--save_references`
* `--max_length_generation` is the maximum token length of generation including the input token length. The default is 512, but for some tasks like GSM8K and GSM-Hard, the complete prompt with 8 shot examples (as used in [PAL](https://github.com/reasoning-machines/pal)) take up `~1500` tokens, hence the value should be greater than that and the recommended value of `--max_length_generation` is `2048` for these tasks.
* For APPS tasks, you can use `--n_samples=1` for strict and average accuracies (from the original APPS paper) and `n_samples>1` for `pass@k` metrics.

Note that some tasks (i.e., benchmarks) don't require code execution (i.e., don't specify `--allow_code_execution`) due to text generation task or lacking unit tests, such as
`codexglue_code_to_text-<LANGUAGE>`/`codexglue_code_to_text-python-left`/`conala`/`concode` that use BLEU evaluation. 
In addition, we generate one candidate solution for each problem in these tasks, so use `--n_samples=1` and `--batch_size=1`. (Note that `batch_size` should always be equal or less than `n_samples`).

## Useful Resources
### LLMs
- [LLaMA](https://github.com/facebookresearch/llama), inference code for LLaMA models
- [Llama 2](https://ai.meta.com/research/publications/llama-2-open-foundation-and-fine-tuned-chat-models/), open foundation and fine-tuned chat models
- [Stanford Alpaca](https://github.com/tatsu-lab/stanford_alpaca), an instruction-following LLaMA model
- [Alpaca-Lora](https://github.com/tloen/alpaca-lora), instruct-tune LLaMA on consumer hardware
- [FastChat](https://github.com/lm-sys/FastChat), an open platform for training, serving, and evaluating large language models. Release repo for Vicuna and Chatbot Arena.
- [GPT Code UI](https://github.com/ricklamers/gpt-code-ui), an open source implementation of OpenAI's ChatGPT Code interpreter
- [PEFT](https://github.com/huggingface/peft), state-of-the-art parameter-efficient fine-tuning (PEFT) methods
- [Codex](https://github.com/openai/human-eval), an evaluation harness for the HumanEval problem solving dataset
- [Code Alpaca](https://github.com/sahil280114/codealpaca), an instruction-following LLaMA model trained on code generation instructions
- [WizardLM](https://github.com/nlpxucan/WizardLM), an instruction-following LLM using Evol-Instruct
- [Self-Instruct](https://github.com/yizhongw/self-instruct), aligning pretrained language models with instruction data generated by themselves.
- [StackLLaMA](https://huggingface.co/blog/stackllama), a hands-on guide to train LLaMA with RLHF
- [StarCoder](https://github.com/bigcode-project/starcoder), a language model (LM) trained on source code and natural language text.
- [CodeGeeX](https://github.com/THUDM/CodeGeeX), a multilingual code generation model
- [CodeGen](https://github.com/salesforce/CodeGen), an open large language model for code with multi-turn program synthesis
- [InCoder](https://github.com/dpfried/incoder), a generative model for code infilling and synthesis 
- [CodeT5+](https://github.com/salesforce/CodeT5), a standard Transformer framework for code understanding and generation
- [CodeBERT](https://github.com/microsoft/CodeBERT), a pre-trained language model for programming and natural languages

### CPU Running
- [llama.cpp](https://github.com/ggerganov/llama.cpp), a native client for running LLaMA models on the CPU
- [alpaca.cpp](https://github.com/antimatter15/alpaca.cpp), a native client for running Alpaca models on the CPU

### Interface
- [Alpaca-LoRA-Serve](https://github.com/deep-diver/Alpaca-LoRA-Serve), a ChatGPT-style interface for Alpaca models

### Dataset
- [AlpacaDataCleaned](https://github.com/gururise/AlpacaDataCleaned), a project to improve the quality of the Alpaca dataset
- [GPT-4 Alpaca Data](https://github.com/Instruction-Tuning-with-GPT-4/GPT-4-LLM), a project to port synthetic data creation to GPT-4
- [Code Alpaca Data](https://github.com/sahil280114/codealpaca/tree/master/data), a project for code generation
- [CodeXGLUE](https://github.com/microsoft/CodeXGLUE), a machine learning benchmark dataset for code understanding and generation
- [HumanEval](https://github.com/openai/human-eval), [APPS](https://huggingface.co/datasets/codeparrot/apps), [HumanEval+](https://github.com/evalplus/evalplus), [MBPP](https://huggingface.co/datasets/mbpp), and [DS-1000](https://github.com/HKUNLP/DS-1000)

### Evaluation
- [Multilingual Code Models Evaluation Leadboard](https://huggingface.co/spaces/bigcode/multilingual-code-evals)
- [Code Generation LM Evaluation Harness](https://github.com/bigcode-project/bigcode-evaluation-harness)

### Hugging Face
- [https://huggingface.co/decapoda-research/llama-7b-hf](https://huggingface.co/decapoda-research/llama-7b-hf)
- [https://huggingface.co/tloen/alpaca-lora-7b](https://huggingface.co/tloen/alpaca-lora-7b)
- [https://huggingface.co/meta-llama/Llama-2-7b-hf](https://huggingface.co/meta-llama/Llama-2-7b-hf)
- [https://huggingface.co/chansung/gpt4-alpaca-lora-7b](https://huggingface.co/chansung/gpt4-alpaca-lora-7b)

### Papers
- [A Survey on Large Language Models for Code Generation](https://arxiv.org/abs/2406.00515)
- [A Survey of Large Language Models](https://arxiv.org/abs/2303.18223)
- [Codex: Evaluating Large Language Models Trained on Code](https://arxiv.org/pdf/2107.03374)
- [LLaMA: Open and Efficient Foundation Language Models](https://arxiv.org/abs/2302.13971v1)
- [Llama 2: Open Foundation and Fine-Tuned Chat Models](https://scontent-nrt1-1.xx.fbcdn.net/v/t39.2365-6/10000000_662098952474184_2584067087619170692_n.pdf?_nc_cat=105&ccb=1-7&_nc_sid=3c67a6&_nc_ohc=ByL78P2ckIMAX8bqkga&_nc_ht=scontent-nrt1-1.xx&oh=00_AfDo3G6cAzYUombvtIceZm9x3NY0jg5mT_L4yEnXodk40w&oe=64C84A3F)
- [Self-Instruct: Aligning Language Model with Self Generated Instructions](https://arxiv.org/abs/2212.10560)
- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
- [InstructGPT: Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155)
- [Emergent Abilities of Large Language Models](https://arxiv.org/abs/2206.07682)
- [StarCoder: may the source be with you!](https://arxiv.org/abs/2305.06161)
- [WizardCoder: Empowering Code Large Language Models with Evol-Instruct](https://arxiv.org/abs/2306.08568)
- [CodeGeeX: A Pre-Trained Model for Code Generation with Multilingual Evaluations on HumanEval-X](https://arxiv.org/abs/2303.17568)
- [CodeT5+: Open Code Large Language Models for Code Understanding and Generation](https://arxiv.org/abs/2305.07922)
- [CodeGen: An Open Large Language Model for Code with Multi-Turn Program Synthesis](https://arxiv.org/abs/2203.13474)
- [InCoder: A Generative Model for Code Infilling and Synthesis](https://arxiv.org/abs/2204.05999)
- [CodeBERT: A Pre-Trained Model for Programming and Natural Languages](https://arxiv.org/abs/2002.08155)
- [CodeXGLUE: A Machine Learning Benchmark Dataset for Code Understanding and Generation](https://arxiv.org/abs/2102.04664)
