## 数据预处理
代码在dataset/data_process/equal_process_data.ipynb里。
equal_process_data.ipynb主要实现了属性替换，年份删除，统一大写，数据划分。属性替换包括相等属性替换和特殊属性替换。
1. 生成替换后的字典
2. 对fine进行内部的替换处理，fine的属性替换是根据提供的key_attr选取的。
3. 对coarse实现属性替换，然后提取coarse的属性，就是创建它的key_attr，这样coarse也有关键属性了。

属性替换的原则：长名替换为短名，最终名字互相不包含，互相不重复。基于此原则有一些特殊的替换。
比如中长款替换为中款，裤子的拉链替换为拉链裤等等。其中替换的坑很多，就不一一列出了。

替换的细节如有兴趣请自行研究代码或与作者联系。

## 数据分词
分词工具使用的是哈工大LTP的base模型，以及我给这个模型添加了一些人工筛选的词。

代码在dataset/split_word里。
equal_split_word.py是将fine45000和coarse89588分词并统计词表word_dict，最终的词表是基于这个词表，而不统计验证集的词。
equal_split_word_val.py是验证集fine5000和coarse10412的分词。
分词的结果存放在data/split_word文件下，多了title_split这个分值。这个分词结果是中间数据，并不是最终使用的数据。

通过process_split_word.ipynb进行词表处理，记为vocab_dict。这个词表为精简和合并的词表，最终的词表vocab.txt是基于这个文件的txt版，通过get_vocab.ipynb得到。process_split_word.ipynb还对data/split_word下的文件进行二次处理，只将vocab_dict中存在的词留下，放到vocab_split这个分值中。这个数据是最终使用的分词数据。

分词的细节如有兴趣请自行研究代码或与作者联系。

## 预训练和finetune
预训练文件为pretrain_split_word.py。使用的模型在model/pretrain_splitbert.py。这个bert是自己实现的，代码是基于transformers库改的，在split_bert中存放。和原始bert的区别为基于自己词表的tokenizer， 新定义word_embedding，不添加位置编码和token_type_ids。

属性finetune文件为finetune_attribute.py。

训练的细节如有兴趣请自行研究代码或与作者联系。