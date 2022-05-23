import json

# 图文
path_2 = 'project/submmision/title_merge.txt'

# 属性
path_1 = 'project/submmision/attr_merge.txt'

out_file = 'project/submmision/results.txt'

with open(path_2, 'r', encoding='utf-8') as f:
    lines_2 = f.readlines()

data_list = []
with open(path_1, 'r', encoding='utf-8') as f:
    lines = f.readlines()
    for line, line_2 in zip(lines, lines_2):
        data = json.loads(line)
        data_2 = json.loads(line_2)
        data['match']['图文'] = data_2['match']['图文']
       
        if data['match']['图文'] == 1:
            for key in data['match']:
                data['match'][key] = 1
        data_list.append(json.dumps(data, ensure_ascii=False)+'\n')

with open(out_file, 'w', encoding='utf-8') as f:
    f.writelines(data_list)
print("ok")