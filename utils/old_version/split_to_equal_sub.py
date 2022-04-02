# fine 
file = '../../data/split_word/fine45000.txt'
save_file = '../../data/final_processed_data/fine45000.txt'

word_dict = {}
rets = []
with open(file, 'r') as f:
    for i, line in enumerate(tqdm(f)):
        item = json.loads(line)
        title = item['title']
        title_split = item['title_split']
        
        # 替换title_split和title
        for n, word in enumerate(title_split):
            if word in equal_dict: # 等号替换
                title_split[n] = equal_dict[word]
            elif word in ['拉链', '系带'] and '裤' in title: # 拉链系带替换
                title_split[n] = ''.join(['裤', word])
            elif word in ['拉链', '系带'] and '鞋' in title: # 拉链系带替换
                title_split[n] = ''.join(['鞋', word])
        new_title = ''.join(title_split)
        item['title'] = new_title
        item['title_split'] = title_split
        
        # 替换key_attr
        key_attr = item['key_attr']
        for query, attr in key_attr.items():
            if attr in equal_dict:
                key_attr[query] = equal_dict[attr]
                
            
        # 统计word_dict
        for word in title_split:
            if word in word_dict:
                word_dict[word] += 1
            else:
                word_dict[word] = 1
        
        rets.append(json.dumps(item, ensure_ascii=False)+'\n')
        
        # if i>500:
        #     break
        # i += 1
        
with open(save_file, 'w') as f:
    f.writelines(rets)