import json

# 读取 A 格式的 JSON 文件
input_file = '/hpc2hdd/home/jjiang472/OpenSource/Datasets/MetaMathQA/MetaMathQA-395K.json'
output_file = 'metamathqa_395k.json'

b_data = []

num_data = 0
with open(input_file, 'r', encoding='utf-8') as f:
    a_data = json.load(f)
    for item in a_data:
        instruction = item.get('query', '')
        output = item.get('response', '')
        
        # 检查 instruction 和 output 是否为空
        if not instruction or not output:
            # raise ValueError(f"Instruction or output is empty in item: {item}")
            continue
        
        b_data.append({
            "instruction": instruction,
            "input": "",
            "output": output
        })
        num_data += 1

# 保存为 B 格式的 JSON 文件
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(b_data, f, ensure_ascii=False, indent=4)

print(f"Conversion completed {num_data} successfully!") # 394,996