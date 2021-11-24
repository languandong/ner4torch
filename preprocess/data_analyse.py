import pandas as pd
from collections import defaultdict
import numpy as np
def get_len(data: pd.DataFrame):
    data['sentence_len'] = data['text'].apply(len)
    print(f'数据长度：{len(data)}')
    print(f'最长句长：{max(data["sentence_len"])}')
    print(f'最短句长：{min(data["sentence_len"])}')

    def func(x):
        if x < 32:
            return 32
        elif x < 64:
            return 64
        elif x < 128:
            return 128
        elif x < 256:
            return 256
        elif x < 1024:
            return 1024
        else:
            return 1025

    data['len2'] = data['sentence_len'].apply(func)
    print(data['len2'].value_counts())


def get_label_distribute(data: pd.DataFrame):
    result = data['class'].value_counts()
    count = []
    for i in range(len(result.index)):
        count.append(result.loc[i])
    count = np.array(count)
    bili = [i/count.sum() for i in count]

    return bili


def convert(data: pd.DataFrame):
    """
    将标签['O', 'O', 'S', 'B']转换为[['实体类别','开始索引','实体'], ...]形式
    """
    texts, tags = data['text'].to_list(), data['BIO_anno'].apply(lambda x: x.strip().split()).to_list()
    total_tag_with_span = []
    for text_list, tag_list in zip(texts, tags):
        assert len(text_list) == len(tag_list)
        text_len = len(text_list)
        start_idx, end_idx = 0, 1
        tag_with_span = []
        while start_idx < text_len:
            tag = tag_list[start_idx]
            bio = tag.split('-')[0]
            if bio != 'B':
                start_idx += 1
                end_idx += 1
            else:
                entity_type = tag.split('-')[1]
                while end_idx < text_len and tag_list[end_idx].split('-')[0] == 'I':
                    end_idx += 1
                tag_with_span.append(
                    [entity_type] + [str(start_idx)] + [''.join(text_list[start_idx:end_idx])])
                start_idx = end_idx
                end_idx += 1
        total_tag_with_span.append(tag_with_span)

    data['span'] = total_tag_with_span
    return data


def get_entity(data: pd.DataFrame, the_specified: str = None):
    data = convert(data)
    entity_dicts = defaultdict(set)
    for tags in data['span']:
        for tag in tags:
            entity = tag[2]
            entity_type = tag[0]
            entity_dicts[entity_type].add(entity)
    if the_specified:
        return entity_dicts[the_specified]
    else:
        return entity_dicts


if __name__ == '__main__':
    # train_data = pd.read_csv('../data/train_data_public.csv')
    # result = pd.read_csv('../data/result.csv')
    # # test_data = pd.read_csv('../data/test_public.csv')
    # # valid_data = pd.read_csv('../data/valid.csv')
    # # print(len(get_entity(valid_data, the_specified='COMMENTS_ADJ')))
    # o  = get_label_distribute(result)
    data = pd.read_csv('../data/train_data_public.csv')
    print(get_label_distribute(data))