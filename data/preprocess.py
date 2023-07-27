import json
import copy
import numpy as np 
from guesslang import Guess # Refer to https://guesslang.readthedocs.io/en/latest/ for more details


# enumerate commonly used PL
pl_name_search = ['JavaScript', 'Java', 'shell', 'Python', 'C++', 'C#', ' C ', 'Bash', 'bash', 
           'HTML', 'SQL', 'JSON', 'CSS', 'JSX', 'Swift', 
           'Ruby', 'PHP', 'Go', 'Kotlin', ' R ', 'MATLAB', 'TypeScript',
           'Scala', 'Haskell', 'Perl', 'Rust']
pl_name = ['JavaScript', 'Java', 'shell', 'Python', 'C++/C', 'C#', 
           'HTML', 'SQL', 'JSON', 'CSS', 'JSX', 'Swift', 
           'Ruby', 'PHP', 'Go', 'Kotlin', 'R', 'MATLAB', 'TypeScript', 
           'Scala', 'Haskell', 'Perl', 'Rust', 'Others']
per_pl_num = dict.fromkeys(pl_name, 0)

# read raw code corpus
with open('./code_alpaca_20k.json', 'r') as f, open('./new_codealpaca.json', 'r') as f_new:
    json_data = json.load(f)
    json_data_new = json.load(f_new)
    print(len(json_data), len(json_data_new))
    json_data.extend(json_data_new)
all_sample = len(json_data)
print("all instruction num: ", all_sample)

unknown_pl_data = []
preprocessed_data = []

for sample in json_data:
    for pl in pl_name_search:
        if pl in sample['instruction']:
            if pl == ' C ' or pl == 'C++':
                pl = 'C++/C'
            elif pl == 'Bash' or pl == 'bash':
                pl = 'shell'
            elif pl == ' R ':
                pl = 'R'
            per_pl_num[pl] += 1
            unknown = False
            preprocessed_data.append(sample)
            break
        else:
            unknown = True
    if unknown:
        unknown_pl_data.append(sample)

# print the statistic of PL
per_pl_num['Others'] = len(unknown_pl_data)
print(per_pl_num)
print(sum(per_pl_num.values()), len(unknown_pl_data))
print(f"unknown PL num: {len(unknown_pl_data)}")


# set the python as default PL for unknown PL data
python_default = copy.deepcopy(per_pl_num)
guess_pl = Guess()
python_marker = ['def ', 'import ',]
for sample in copy.deepcopy(unknown_pl_data):
    pl_name_in = guess_pl.language_name(sample['input']) if sample['input'] != "" else None
    pl_name_out = guess_pl.language_name(sample['output'])
    if pl_name_in == 'Python' or pl_name_out == 'Python' or any(marker in sample['output'] for marker in python_marker):
        python_default['Python'] += 1
        unknown_pl_data.remove(sample)
        preprocessed_data.append(sample)
    # else:
    #     print(sample)
    #     print(pl_name_in, pl_name_out)
    #     input('check')

# print the statistic of PL
python_default['Others'] = len(unknown_pl_data)
print(python_default)
print(sum(python_default.values()), len(unknown_pl_data))
print(f"unknown PL num: {len(unknown_pl_data)}")
print('before Python num: ', per_pl_num['Python'], ', after Python num: ', python_default['Python'])

# record unknown PL data
print(unknown_pl_data[:5])
json_data_object = json.dumps(unknown_pl_data, indent=6)
with open("unknown_pl_data.json", "w") as outfile:
    outfile.write(json_data_object)

# record clean up PL data
print(len(preprocessed_data))
all_len = len(preprocessed_data)//1000
print(f"Clean up PL corpus: {all_len}k")
clean_json_data = json.dumps(preprocessed_data, indent=6)
with open(f'codeup_{all_len}k.json', 'w') as f:
    f.write(clean_json_data)

# save the number of each PL
with open('pl_raw.json', 'w') as f_raw, open('pl_clean.json', 'w') as f_clean:
    json.dump(per_pl_num, f_raw)
    json.dump(python_default, f_clean)
