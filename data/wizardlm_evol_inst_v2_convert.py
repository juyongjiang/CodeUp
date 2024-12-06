import json

# 读取 A 格式的 JSON 文件
with open('/hpc2hdd/home/jjiang472/OpenSource/Datasets/WizardLM_evol_instruct_V2_196k/WizardLM_evol_instruct_V2_143k.json', 'r', encoding='utf-8') as f:
    a_data = json.load(f)

# 转换为 B 格式
b_data = []
num_data = 0
for item in a_data:
    conversation = item['conversations']
    instruction = ""
    output = ""
    for conv in conversation:
        if conv['from'] == 'human':
            instruction = conv['value']
        elif conv['from'] == 'gpt':
            output = conv['value']
        else:
            raise ValueError(f"Human or value is empty in item: {conv}")
    b_data.append({
        "instruction": instruction,
        "input": "",
        "output": output
    })
    num_data += 1

# 保存为 B 格式的 JSON 文件
with open('wizardlm_evol_instruct_v2_143k.json', 'w', encoding='utf-8') as f:
    json.dump(b_data, f, ensure_ascii=False, indent=4)

print(f"Conversion completed {num_data} successfully!") # 143k