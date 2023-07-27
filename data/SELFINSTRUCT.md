## Self-instruct Code Data Generation

We built on the `100K` code instruction data generation pipeline from [self-instruct](https://github.com/yizhongw/self-instruct) and [stanford-alpaca](https://github.com/tatsu-lab/stanford_alpaca#data-generation-process). The following modifications is borrowed from [stanford-alpaca](https://github.com/tatsu-lab/stanford_alpaca#data-generation-process):

- Using `text-davinci-003` to generate the instruction data instead of `davinci`.
- Writing a new prompt (`prompt.txt`) that explicitly gave the requirement of instruction generation to `text-davinci-003`. 
- Adopting much more aggressive batch decoding, i.e., generating `20 instructions at once`, which significantly reduced the cost of data generation.
- Simplifying the data generation pipeline by discarding the difference between classification and non-classification instructions.
- Only generating a `single instance` for each instruction, instead of 2 to 3 instances as in [1].

### Generation Details [1]

* **Step 1 (Seed Tasks)**: Utilizing the 175 human-written task seed sets of `seed_tasks.jsonl` as initialization instruction samples. Note that each task contains an instruction and an input-output instance, and some task instances have an empty input field. The examples are as follows:

```
[
    {
        "instruction": "Give three tips for staying healthy.",
        "input": "",
        "output": "1.Eat a balanced diet and make sure to include plenty of fruits and vegetables. \n2. Exercise regularly to keep your body active and strong. \n3. Get enough sleep and maintain a consistent sleep schedule."
    },
    {
        "instruction": "Use the given data to calculate the median.",
        "input": "[2, 3, 7, 8, 10]",
        "output": "The median of the given data is 7."
    },
    {
        "instruction": "Translate the following sentence into Spanish.",
        "input": "The blue sky is so beautiful.",
        "output": "El cielo azul es tan hermoso."
    },
    ......
]
```

* **Step 2 (Build Prompt)**: Building a prompt and call the `text-davinci-003` interface to generate instruction data in batches. The prompt is composed of two parts: 1) the requirement of instruction generation (as shown in `prompt.txt`); 2) 3 samples randomly sampled from the 175 human-written task seed sets which denotes the in-context examples in the prompt. A prompt example is as follows:

```
You are asked to come up with a set of 20 diverse task instructions. These task instructions will be given to a GPT model and we will evaluate the GPT model for completing the instructions.

Here are the requirements:
1. Try not to repeat the verb for each instruction to maximize diversity.
2. The language used for the instruction also should be diverse. For example, you should combine questions with imperative instrucitons.
3. The type of instructions should be diverse. The list should include diverse types of tasks like open-ended generation, classification, editing, etc.
2. A GPT language model should be able to complete the instruction. For example, do not ask the assistant to create any visual or audio output. For another example, do not ask the assistant to wake you up at 5pm or set a reminder because it cannot perform any action.
3. The instructions should be in English.
4. The instructions should be 1 to 2 sentences long. Either an imperative sentence or a question is permitted.
5. You should generate an appropriate input to the instruction. The input field should contain a specific example provided for the instruction. It should involve realistic data and should not contain simple placeholders. The input should provide substantial content to make the instruction challenging but should ideally not exceed 100 words.
6. Not all instructions require input. For example, when a instruction asks about some general information, "what is the highest peak in the world", it is not necssary to provide a specific context. In this case, we simply put "<noinput>" in the input field.
7. The output should be an appropriate response to the instruction and the input. Make sure the output is less than 100 words.

List of 20 tasks:

###
1. Instruction: 
Give three tips for staying healthy.
1. Input: 
<noinput>
1. Output: 
1.Eat a balanced diet and make sure to include plenty of fruits and vegetables. \n2. Exercise regularly to keep your body active and strong. \n3. Get enough sleep and maintain a consistent sleep schedule.
###
2. Instruction: 
Use the given data to calculate the median.
2. Input: 
[2, 3, 7, 8, 10]
2. Output: 
The median of the given data is 7.
###
3. Instruction: 
Translate the following sentence into Spanish.
3. Input: 
The blue sky is so beautiful.
3. Output: 
El cielo azul es tan hermoso.
###
4. Instruction:
```

* **Step 3 (Instruction Filter)**: Performing `ROUGE-L` similar measure on the newly `generated instruction data` and the `existing instruction data`. If the `ROUGE-L` value is greater than `0.7`, the newly generated instruction data will be filtered to ensure the `diversity` of instruction data.

[1]: 开源大模型微调和训练-指令遵循语言模型 Alpaca. https://zhuanlan.zhihu.com/p/618423685

### Running

1. Set environment variables `OPENAI_API_KEY` to your OpenAI API key.
2. Install the dependencies with `pip install -r requirements.txt`.
3. Run `python -m generate_instruction generate_instruction_following_data` to generate the data.