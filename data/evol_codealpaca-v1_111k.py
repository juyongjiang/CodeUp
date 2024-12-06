import json

# 读取 A 格式的 JSONL 文件
input_file = '/hpc2hdd/home/jjiang472/OpenSource/Datasets/evol-codealpaca-v1/train.jsonl'
output_file = 'evol_codealpaca_v1_111k.json'

b_data = []
num_data = 0
with open(input_file, 'r', encoding='utf-8') as f:
    for line in f:
        item = json.loads(line)
        instruction = item.get('instruction', '')
        output = item.get('output', '')
        
        # 检查 instruction 和 output 是否为空
        if not instruction or not output:
            raise ValueError(f"Instruction or output is empty in item: {item}")
        
        b_data.append({
            "instruction": instruction,
            "input": "",
            "output": output
        })
        num_data += 1

# 保存为 B 格式的 JSON 文件
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(b_data, f, ensure_ascii=False, indent=4)

print(f"Conversion completed {num_data} successfully!") # 111,272