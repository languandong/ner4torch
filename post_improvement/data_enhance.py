from utils import convert, resolve_tag_for_train
from ast import literal_eval
from collections import defaultdict, Counter
import pandas as pd
import random
import copy
from sklearn.utils import shuffle


def get_entity(data_path, tag_type=2, the_specified=None):
    """
    返回实体列表
    @param the_specified: 只返回指定类别实体
    @param data_path: 文件路径
    @param tag_type: 0：接受标签形式为['O'\n, 'O'\n, 'S-...'\n, 'B-...'\n]的文件
                     1：接受标签形式为['实体类别-开始索引-实体', ...]的文件
                     2：接受标签形式为[['实体类别', '开始索引', '实体'], ...]的文件
    """
    if tag_type == 0:
        df = convert(data_path)
    else:
        df = pd.read_csv(data_path)
        df['tag'] = df['tag'].apply(literal_eval)
    tag_list = df['tag'].to_list()
    entity_tag_dict = defaultdict(list)
    for i, tag_list in enumerate(tag_list):
        for tag in tag_list:
            if len(tag) == 0:
                continue
            else:
                if tag_type == 1:
                    entity_tag_dict[tag.split('-')[2]].append(tag.split('-')[0])
                elif tag_type == 2:
                    entity_tag_dict[tag[2]].append(tag[0])
    all_entity = []
    for entity in entity_tag_dict.keys():
        type_count = dict(Counter(entity_tag_dict[entity]))
        type_count = list(type_count.items())
        type_count = sorted(type_count, key=lambda x: -x[1])
        all_entity.append((entity, type_count))

    all_entity = sorted(all_entity, key=lambda x: -x[1][0][1])
    if the_specified is None:
        entity_list = [i[0] for i in all_entity]
        entity_len_list = [len(i[0]) for i in all_entity]
    else:
        entity_list = [i[0] for i in all_entity if i[1][0][0] == the_specified]
        entity_len_list = [len(i[0]) for i in all_entity if i[1][0][0] == the_specified]
    return entity_list, max(entity_len_list)


def add_symbol(entity_num, seq, tag_lst):
    symbols = ['+', '$', '#', '￥', '&', '@', '=', '【', '】', '「', '」', '{', '}', '《', '》', '（', '）', '`',
               '|', '(', ')', '？', '?', '!', '！', '<', '>', '；', ';', '『', '』', '-', ]
    sequence = copy.deepcopy(seq)
    tag_list = copy.deepcopy(tag_lst)
    if entity_num == 1:
        # 实体个数只有1个
        symbol = random.choice(symbols)
        entity, entity_start = list(tag_list[0][2]), int(tag_list[0][1])
        entity_len = len(entity)  # 实体长度
        if entity_len == 1:
            # 实体长度为1，将符号插入实体后面
            tag_list[0][2] = ''.join(entity)
            sequence.insert(entity_start + 1, symbol)
        else:
            # 实体长度大于1，将符号插入实体中间
            entity.insert(int(entity_len % 2) + 1, symbol)
            tag_list[0][2] = ''.join(entity)
            sequence.insert(entity_start + int(entity_len % 2) + 1, symbol)
    else:
        # 实体个数大于1，要考虑插入符号后实体的位置变化
        entity1, entity2 = list(tag_list[0][2]), list(tag_list[1][2])
        entity1_start, entity2_start = int(tag_list[0][1]), int(tag_list[1][1])
        entity1_len, entity2_len = len(entity1), len(entity2)
        symbol1, symbol2 = random.choice(symbols), random.choice(symbols)
        if entity_num == 2:
            # 将符号插入第一个实体前面，插入第二个实体中间
            tag_list[0][2] = ''.join(entity1)
            sequence.insert(entity1_start, symbol1)
            tag_list[0][1] = str(entity1_start + 1)
            insert_index = 1 if entity2_len == 1 else int(entity2_len % 2) + 1
            if entity2_len > 1:
                entity2.insert(insert_index, symbol2)
            tag_list[1][2] = ''.join(entity2)
            entity2_start = entity2_start + 1
            sequence.insert(entity2_start + insert_index, symbol2)
            tag_list[1][1] = str(entity2_start)
        else:
            # 将符号插入第一个实体前面，插入第二个实体后面，插入除前二个实体外所有实体中间
            tag_list[0][2] = ''.join(entity1)
            sequence.insert(entity1_start, symbol1)
            tag_list[0][1] = str(entity1_start + 1)
            tag_list[1][2] = ''.join(entity2)
            entity2_start = entity2_start + 1
            sequence.insert(entity2_start + entity2_len, symbol2)
            tag_list[1][1] = str(entity2_start)
            for n in range(2, entity_num):
                entity, entity_start = list(tag_list[n][2]), int(tag_list[n][1])
                symbol = random.choice(symbols)
                entity_len = len(entity)
                insert_index = 1 if entity_len == 1 else int(entity_len % 2) + 1
                if entity_len > 1:
                    entity.insert(insert_index, symbol)
                tag_list[n][2] = ''.join(entity)
                tag_list[n][1] = str(entity_start + n)
                sequence.insert(int(tag_list[n][1]) + insert_index, symbol)
    return sequence, tag_list


def replace_entity(entity_num, seq, tag_lst, entity_dict):
    sequence = copy.deepcopy(seq)
    tag_list = copy.deepcopy(tag_lst)
    move_back = 0  # 下一个实体的开始位置需要往后移动多少位
    for n in range(entity_num):
        entity_type, entity_start, entity = tag_list[n][0], int(tag_list[n][1]), tag_list[n][2]
        if entity_type == 'O' or len(entity) == entity_dict[entity_type]['max_len']:
            continue
        flag, new_entity = True, None
        while flag:
            new_entity = random.choice(entity_dict[entity_type]['entity_list'])
            if len(new_entity) >= len(entity) and new_entity != entity:
                flag = False
        new_entity_start = entity_start + move_back
        tag_list[n][1] = str(new_entity_start)
        tag_list[n][2] = new_entity
        for i, idx in enumerate(range(new_entity_start, new_entity_start + len(new_entity))):
            if i < len(entity):
                sequence[idx] = new_entity[i]
            else:
                sequence.insert(idx, new_entity[i])
        move_back += (len(new_entity) - len(entity))
    return sequence, tag_list


def random_insert_sentence(seq, tag_lst, text_data, tag_data):
    sequence = copy.deepcopy(seq)
    tag_list = copy.deepcopy(tag_lst)
    sequence_len = len(sequence)
    insert_index = random.randint(0, len(text_data))
    new_sequence = sequence + list(copy.deepcopy(text_data[insert_index]))
    insert_tag_list = copy.deepcopy(tag_data[insert_index])
    for idx in range(len(insert_tag_list)):
        insert_tag_list[idx][1] = str(int(insert_tag_list[idx][1]) + sequence_len)
    new_tag_list = tag_list + insert_tag_list
    return new_sequence, new_tag_list


def main(data_path):
    df = pd.read_csv(data_path)
    text_data = df['text'].to_list()
    tag_data = df['tag'].apply(literal_eval).to_list()
    loc_entity_list, loc_max_len = get_entity(data_path, the_specified='LOC')
    org_entity_list, org_max_len = get_entity(data_path, the_specified='ORG')
    gpe_entity_list, gpe_max_len = get_entity(data_path, the_specified='GPE')
    per_entity_list, per_max_len = get_entity(data_path, the_specified='PER')
    entity_dict = {'LOC': {'entity_list': loc_entity_list, 'max_len': loc_max_len},
                   'ORG': {'entity_list': org_entity_list, 'max_len': org_max_len},
                   'GPE': {'entity_list': gpe_entity_list, 'max_len': gpe_max_len},
                   'PER': {'entity_list': per_entity_list, 'max_len': per_max_len}, }
    enhance_text_data, enhance_tag_data = [], []
    # enhance_methods = ['add_symbol', 'replace_entity']
    enhance_methods = ['replace_entity']
    random.seed(0)
    for sequence, tag_list in zip(text_data, tag_data):
        enhance_text_data.append(sequence)
        enhance_tag_data.append(copy.deepcopy(tag_list))
        sequence = list(sequence)
        entity_num = len(tag_list)
        if len(sequence) > 4 and entity_num > 0:
            # 加入噪声符号
            if 'add_symbol' in enhance_methods:
                seq, tag_lst = add_symbol(entity_num, sequence, tag_list)
                enhance_text_data.append(''.join(seq))
                enhance_tag_data.append(tag_lst)
            # 替换同类型实体
            if 'replace_entity' in enhance_methods:
                seq, tag_lst = replace_entity(entity_num, sequence, tag_list, entity_dict)
                enhance_text_data.append(''.join(seq))
                enhance_tag_data.append(tag_lst)
        # 随机插入语义无关句子
        if 'random_insert_sentence' in enhance_methods:
            seq, tag_lst = random_insert_sentence(sequence, tag_list, text_data, tag_data)
            enhance_text_data.append(''.join(seq))
            enhance_tag_data.append(tag_lst)

    enhance_train = pd.DataFrame({'text': enhance_text_data, 'tag': enhance_tag_data})
    enhance_train = shuffle(enhance_train, random_state=0)
    resolve_tag_for_train(dataframe=enhance_train).to_csv('./data/enhance_train.csv', index=False)


if __name__ == '__main__':
    main('./data/correct_char_ner_train.csv')
