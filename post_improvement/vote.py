import numpy as np
import pandas as pd
from collections import Counter, defaultdict
from sklearn.utils import shuffle
from ast import literal_eval
from tqdm import tqdm
import re


def convert(data_path, fold=-1):
    """
    将标签['O'\n, 'O'\n, 'S-...'\n, 'B-...'\n]转换为['实体类别-开始索引-实体', ...]形式
    @param data_path: 文件路径
    @param fold: 第几折；-1表示对原始训练集或预测结果进行格式转换
    @return: DataFrame
    """
    texts, tags = [], []
    with open(data_path, 'r') as f:
        text, tag = [], []
        for i in f.readlines()[1:]:
            if len(i) > 2:
                if '","' in i:
                    text.append('@')
                    temp_tag = i.strip('\n').split(',')[-1]
                    if '""""' in temp_tag:
                        tag.append('O')
                    else:
                        tag.append(temp_tag)
                elif '""""' in i:
                    text.append('@')
                    temp_tag = i.strip('\n').split(',')[-1]
                    if '""""' in temp_tag:
                        tag.append('O')
                    else:
                        tag.append(temp_tag)
                else:
                    text.append(i.strip('\n').split(',')[0])
                    tag.append(i.strip('\n').split(',')[1])
            else:
                texts.append(text)
                tags.append(tag)
                text, tag = [], []
    total_tag_with_span = []
    total_sentence_list = []
    for te_list, ta_list in zip(texts, tags):
        if fold == -1: total_sentence_list.append(''.join(te_list))
        start_idx, end_idx = -1, -1
        tag_with_span = []
        for idx, tag in enumerate(ta_list):
            if tag == 'O':
                continue
            bmes = tag.split('-')[0]
            entity_type = tag.split('-')[1]
            if bmes == 'S':
                tag_with_span.append(entity_type + '-' + str(idx) + '-' + ''.join(te_list[idx:idx + 1]))
            else:
                if bmes == 'B':
                    start_idx = idx
                elif bmes == 'E':
                    if start_idx != -1:
                        end_idx = idx
                        tag_with_span.append(
                            entity_type + '-' + str(start_idx) + '-' + ''.join(te_list[start_idx:end_idx + 1]))
                        start_idx, end_idx = -1, -1
        total_tag_with_span.append(tag_with_span)
    if fold == -1:
        return pd.DataFrame({'text': total_sentence_list, 'tag': total_tag_with_span})
    else:
        return pd.DataFrame({'span': total_tag_with_span})


def vote_v0(select_vote, left_vote):
    """
    没有投票值大于3的实体，才抽取小于3的实体
    """
    if len(select_vote) == 0 and len(left_vote) != 0:
        # 将实体拆分为字符串的形式  eg：实体类型 - 实体开始位置 - 实体 - 出现的次数
        left_vote = [key + "-" + str(value) for i in left_vote for key, value in i.items()]
        left_vote = sorted(left_vote, key=lambda x: (-int(x.split("-")[-1]), -len(x.split("-")[2])))
        select_vote.append(left_vote.pop(0))
        position_select_vote = [int(i.split("-")[1]) for i in select_vote]
        latent_entity = []
        for left_entity in left_vote:
            entity_position = int(left_entity.split("-")[1])
            flag = 0
            for i in position_select_vote:
                if abs(entity_position - i) <= 3:
                    flag = 1
            if flag == 0 and int(left_entity.split("-")[-1]) >= 2:
                latent_entity.append(left_entity)
        select_vote.extend(latent_entity)
        select_vote = [{"-".join(i.split("-")[:-1]): int(i.split("-")[-1])} for i in select_vote]
        return select_vote
    else:
        return select_vote


def vote_v1(select_vote, left_vote, text, more_recall=False):
    """
    @param select_vote: 投票值>=3的实体字典, e.g. {'实体类型 - 实体开始位置 - 实体' : 出现的次数}
    @param left_vote: 投票值<3的实体字典
    @param text: 相应句子
    @param more_recall: 是否召回更多实体，如果是，则抽取全部无重叠实体，否则只在没有投票值>=3的实体的情况下，才抽取小于3的实体。
    @return: 句子最终包含的实体字典, e.g. {'实体类型 - 实体开始位置 - 实体' : 出现的次数}
    """
    if len(left_vote) != 0:
        if not more_recall and len(select_vote) > 0:
            return select_vote
        # 将实体拆分为字符串的形式  eg：实体类型 - 实体开始位置 - 实体 - 出现的次数
        left_vote = [key + "-" + str(value) for i in left_vote for key, value in i.items()]
        select_vote = [key + "-" + str(value) for i in select_vote for key, value in i.items()]
        left_vote = sorted(left_vote, key=lambda x: (-int(x.split("-")[-1]), -len(x.split("-")[2])))
        # 取所有已选择实体的开始位置，长度
        position_select_vote = [[int(i.split("-")[1]), len(i.split("-")[2])] for i in select_vote]
        # 构造实体选择列表(1表示实体，0表示其他)
        have_been_chosen = np.array([0] * len(text))
        for i in position_select_vote:
            have_been_chosen[i[0]:(i[0] + i[1])] = 1
        # 遍历所有剩下实体，如果和已选择实体无重叠，则添加
        latent_entity = []
        for left_entity in left_vote:
            span = np.array([0] * len(text))
            entity_start, entity_len = int(left_entity.split("-")[1]), len(left_entity.split("-")[2])
            span[entity_start:(entity_start + entity_len)] = 1
            if not np.any(span * have_been_chosen) and int(left_entity.split("-")[-1]) >= 2:
                latent_entity.append(left_entity)
                have_been_chosen[entity_start:(entity_start + entity_len)] = 1
        select_vote.extend(latent_entity)
        select_vote = [{"-".join(i.split("-")[:-1]): int(i.split("-")[-1])} for i in select_vote]
        return select_vote
    else:
        return select_vote


def resolve_tag_for_train(data_path):
    """
    将标签[['实体类别', '开始索引', '实体'], [...], ...]转换为['O', 'O', 'S-...', 'B-...']形式
    """
    data = pd.read_csv(data_path)
    text_list = data['text'].to_list()
    data['tag'] = data['tag'].apply(literal_eval)
    tag_list = data['tag'].to_list()
    resolve_tag_list = []
    for i, tags in enumerate(tag_list):
        res_tag = ['O'] * len(text_list[i])
        if len(tags) == 0:
            resolve_tag_list.append(res_tag)
        else:
            for tag in tags:
                if len(tag[-1]) == 1:
                    res_tag[int(tag[1])] = ('S-' + tag[0]) if tag[0] != 'O' else 'O'
                else:
                    entity_len = len(tag[-1])
                    entity_type = tag[0]
                    entity_start = int(tag[1])
                    for idx in range(entity_start, entity_start + entity_len):
                        if idx == entity_start:
                            res_tag[idx] = ('B-' + entity_type) if entity_type != 'O' else 'O'
                        elif idx == entity_start + entity_len - 1:
                            res_tag[idx] = ('E-' + entity_type) if entity_type != 'O' else 'O'
                        else:
                            res_tag[idx] = ('M-' + entity_type) if entity_type != 'O' else 'O'
            resolve_tag_list.append(res_tag)

    pd.DataFrame({'text': text_list, 'tag': resolve_tag_list}).to_csv(data_path, index=False)


def resolve_tag_for_submit(data_path, train=False):
    """
    将标签['O', 'O', 'S-...', 'B-...']转换为['O'\n, 'O'\n, 'S-...'\n, 'B-...'\n]形式
    """
    if not train: resolve_tag_for_train(data_path)
    data = pd.read_csv(data_path)
    text_list = data['text'].to_list()
    data['tag'] = data['tag'].apply(literal_eval)
    tag_list = data['tag'].to_list()
    ids, tags, chars = [], [], []
    i = 0
    for text, label in zip(text_list, tag_list):
        assert len(text) == len(label)
        for index, char in enumerate(text):
            ids.append(i)
            chars.append(char)
            tags.append(label[index])
            i += 1

        ids.append('')
        tags.append('')
        chars.append('')

    result = pd.DataFrame({'id': ids, 'tag': tags})
    if train:
        tag_with_char = pd.DataFrame({'chars': chars, 'target': tags})
        tag_with_char.to_csv(data_path[:-4] + '.conll', index=False)
    else:
        result_with_char = pd.DataFrame({'char': chars, 'tag': tags})
        result.to_csv(f"./data/{data_path.split('/')[2]}", index=False)
        result_with_char.to_csv(f"./data/with_char_{data_path.split('/')[2]}", index=False)


def resolve_tag_for_enhance(data_path):
    """
    将标签['O', 'O', 'S-...', 'B-...']转换为[['实体类别', '开始索引', '实体'], [...], ...]形式
    """
    resolve_tag_for_submit(data_path, train=True)
    df = convert(data_path[:-4] + '.conll')
    tag = df['tag'].to_list()
    text = df['text'].to_list()
    new_tag = []
    for t in tag:
        new_tag.append([[span.split('-')[0]] + [span.split('-')[1]] + [span.split('-')[2]] for span in t])
    pd.DataFrame({'text': text, 'tag': new_tag}).to_csv(data_path[:-4] + '.enhance', index=False)


def vote_fold(more_recall=False):
    """
    5z投票
    """
    total_fold5_tag_list = []
    for i in range(1, 6):
        file_path = f"./data/fold{i}_with_char.csv"
        df = convert(file_path, fold=i)
        each_fold_tag_list = df["span"].tolist()
        total_fold5_tag_list.append(each_fold_tag_list)
    new_tag_list = []
    for index in range(len(total_fold5_tag_list[0])):
        new_tag_list.append(total_fold5_tag_list[0][index] + \
                            total_fold5_tag_list[1][index] + \
                            total_fold5_tag_list[2][index] + \
                            total_fold5_tag_list[3][index] + \
                            total_fold5_tag_list[4][index])

    vote_result_list = []
    text_list = pd.read_csv('./data/test.csv')['text']
    for i, j in zip(new_tag_list, text_list):
        select_vote, left_vote = [], []
        for key, value in Counter(i).items():
            if value >= 3:
                select_vote.append({key: value})
            if value <= 2:
                left_vote.append({key: value})

        select_vote = vote_v1(select_vote, left_vote, j, more_recall=more_recall)

        select_vote = [[key.split("-")[0]] + [key.split("-")[1]] + [key.split("-")[2]] for i in select_vote for
                       key, value in i.items()]

        vote_result_list.append(select_vote)
    vote_result = pd.DataFrame({'text': text_list, 'tag': vote_result_list})
    vote_result.to_csv('./data/fold5_vote_result.csv', index=False)
    resolve_tag_for_submit('./data/fold5_vote_result.csv')


def vote_model(more_recall=False):
    """
    不同模型投票
    """
    total_model_tag_list = []
    for model_name in ['bert', 'wwm', 'base']:
        file_path = f"./data/{model_name}_with_char_correct_result.csv"
        df = convert(file_path, fold=1)
        print(model_name)
        each_fold_tag_list = df["span"].tolist()
        total_model_tag_list.append(each_fold_tag_list)
    new_tag_list = []
    for index in range(len(total_model_tag_list[0])):
        new_tag_list.append(total_model_tag_list[0][index] + \
                            total_model_tag_list[1][index] + \
                            total_model_tag_list[2][index])

    vote_result_list = []
    text_list = pd.read_csv('./data/test.csv')['text']
    for i, j in zip(new_tag_list, text_list):
        select_vote, left_vote = [], []
        for key, value in Counter(i).items():
            if value >= 2:
                select_vote.append({key: value})
            if value <= 1:
                left_vote.append({key: value})

        select_vote = vote_v1(select_vote, left_vote, j, more_recall=more_recall)

        select_vote = [[key.split("-")[0]] + [key.split("-")[1]] + [key.split("-")[2]] for i in select_vote for
                       key, value in i.items()]

        vote_result_list.append(select_vote)
    vote_result = pd.DataFrame({'text': text_list, 'tag': vote_result_list})
    vote_result.to_csv('./data/vote_model_result.csv', index=False)
    resolve_tag_for_submit('./data/vote_model_result.csv')


def vote_train(more_recall=False):
    """
    对原始训练集的预测结果进行投票，根据投票结果对训练集进行纠正。
    """
    total_fold5_tag_list = []
    for i in range(1, 6):
        file_path = f"./data/fold{i}_with_char.csv"
        df = convert(file_path, fold=i)
        each_fold_tag_list = df["span"].tolist()
        total_fold5_tag_list.append(each_fold_tag_list)
    new_tag_list = []
    for index in range(len(total_fold5_tag_list[0])):
        new_tag_list.append(total_fold5_tag_list[0][index] + \
                            total_fold5_tag_list[1][index] + \
                            total_fold5_tag_list[2][index] + \
                            total_fold5_tag_list[3][index] + \
                            total_fold5_tag_list[4][index])

    vote_result_list = []
    text_list = pd.read_csv('./data/correct_all.csv')['text']
    for i, j in zip(new_tag_list, text_list):
        select_vote, left_vote = [], []
        for key, value in Counter(i).items():
            if value >= 3:
                select_vote.append({key: value})
            if value <= 2:
                left_vote.append({key: value})

        select_vote = vote_v1(select_vote, left_vote, j, more_recall=more_recall)

        select_vote = [[key.split("-")[0]] + [key.split("-")[1]] + [key.split("-")[2]] for i in select_vote for
                       key, value in i.items()]

        select_vote = sorted(select_vote, key=lambda x: int(x[1]))

        vote_result_list.append(select_vote)
    vote_result = pd.DataFrame({'text': text_list, 'tag': vote_result_list})
    vote_result.to_csv('./data/vote_train.csv', index=False)


def show_contradictory_entity(data_path, tag_type=0):
    """
    显示多类别实体（潜在矛盾实体）
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
                if tag_type == 0 or 1:
                    entity_tag_dict[tag.split('-')[2]].append(tag.split('-')[0])
                elif tag_type == 2:
                    entity_tag_dict[tag[2]].append(tag[0])
    contradictory_entity = []
    for entity in entity_tag_dict.keys():
        type_count = dict(Counter(entity_tag_dict[entity]))
        type_count = list(type_count.items())
        type_count = sorted(type_count, key=lambda x: -x[1])
        if len(type_count) > 1:
            contradictory_entity.append((entity, type_count))
        start_found = re.match("[+$#￥&@=【】「」{}《》（）`|()？?!！<>；;『』-]", entity)
        end_found = re.match("[+$#￥&@=【】「」{}《》（）`|()？?!！<>；;『』-]", entity[::-1])
        if start_found or end_found:
            contradictory_entity.append((entity, type_count))
        if '百分之' in entity:
            contradictory_entity.append((entity, type_count))

    contradictory_entity = sorted(contradictory_entity, key=lambda x: -x[1][0][1])
    for i in contradictory_entity:
        print(i)


def show_all_entity(data_path, tag_type=0, the_specified=None):
    """
    显示所有实体
    @param the_specified: 只显示指定类别实体
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
                if tag_type == 0 or 1:
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
        for i in all_entity:
            print(i)
    else:
        for i in all_entity:
            if i[1][0][0] == the_specified:
                print(i)


def get_correct_entity_tag_map(df, train=False, test=False):
    """
    返回纠正字典
    @param df: 经convert转换过的DataFrame
    @param train: 训练集更正
    @param test: 测试集更正
    @return: 纠正字典
    """
    tag_list = df['tag'].to_list()
    entity_tag_dict = defaultdict(list)
    for i, tag_list in enumerate(tag_list):
        for tag in tag_list:
            if len(tag) == 0:
                continue
            else:
                entity_tag_dict[tag.split('-')[2]].append(tag.split('-')[0])
    correct_entity_map = {}
    remove_start_char_entity, remove_end_char_entity, remove_se_char_entity = [], [], []
    for entity in entity_tag_dict.keys():
        type_count = dict(Counter(entity_tag_dict[entity]))
        type_count = list(type_count.items())
        type_count = sorted(type_count, key=lambda x: -x[1])
        if len(type_count) > 1:
            correct_entity_map[entity] = type_count[0][0]
        start_found = re.match("[+$#￥&@=…*【】「」{}《》（）`|/()？?!！<>；;『』〈〉、~—-]", entity)
        end_found = re.match("[+$#￥&@=…*【】「」{}《》（）`|/()？?!！<>；;『』〈〉、~—-]", entity[::-1])
        if start_found and end_found:
            remove_se_char_entity.append(entity)
        elif start_found:
            remove_start_char_entity.append(entity)
        elif end_found:
            remove_end_char_entity.append(entity)

    correct_entity_map['９号'] = 'O'
    correct_entity_map['非洲'] = 'LOC'
    correct_entity_map['檀香山'] = 'LOC'
    correct_entity_map['今天'] = 'O'
    correct_entity_map['远东'] = 'LOC'
    correct_entity_map['珠海'] = 'GPE'
    correct_entity_map['中山'] = 'GPE'
    correct_entity_map['大同'] = 'GPE'
    correct_entity_map['卢旺达'] = 'GPE'
    correct_entity_map['维吉尼亚州'] = 'GPE'
    correct_entity_map['田纳西州的杰克斯保瑞'] = 'GPE'
    correct_entity_map['田纳西州'] = 'GPE'
    correct_entity_map['印度政府'] = 'GPE'
    correct_entity_map['中国政府'] = 'GPE'
    correct_entity_map['加利福尼亚州'] = 'GPE'
    correct_entity_map['汉廷顿'] = 'GPE'
    correct_entity_map['纽约市'] = 'GPE'
    correct_entity_map['第一'] = 'O'
    correct_entity_map['波黑'] = 'GPE'
    correct_entity_map['丁宜洲'] = 'PER'
    correct_entity_map['８５岁'] = 'O'
    correct_entity_map['非洲裔'] = 'O'
    correct_entity_map['亚洲人'] = 'O'
    correct_entity_map['连云港'] = 'GPE'
    correct_entity_map['加沙市'] = 'GPE'
    correct_entity_map['银川市'] = 'GPE'
    correct_entity_map['山西省'] = 'GPE'
    correct_entity_map['武乡县'] = 'GPE'
    correct_entity_map['同湖县'] = 'GPE'
    correct_entity_map['林'] = 'PER'
    correct_entity_map['三'] = 'O'

    if train:
        del correct_entity_map['青']  # '青':[('ORG', 1), ('GPE', 1)]
        del correct_entity_map['华']  # '华':[('GPE', 9), ('PER', 1)]
        del correct_entity_map['金']  # '金':[('GPE', 7), ('PER', 2)]
        del correct_entity_map['妈祖']  # '妈祖':[('GPE', 6), ('PER', 2)]
        del correct_entity_map['韩']  # '韩':[('GPE', 8), ('PER', 6)]
        del correct_entity_map['河北']  # '河北':[('GPE', 2), ('LOC', 1)]
        del correct_entity_map['美']  # ('美', [('GPE', 20), ('LOC', 4)])
        del correct_entity_map['苏']  # ('苏', [('PER', 3), ('GPE', 1)])
        del correct_entity_map['澎湖']  # ('澎湖', [('ORG', 2), ('GPE', 1), ('LOC', 1)])
        del correct_entity_map['汤']  # ('汤', [('GPE', 1), ('PER', 1)])
    elif test:
        correct_entity_map['韩国队'] = 'ORG'
        correct_entity_map['中国队'] = 'ORG'
        correct_entity_map['香港队'] = 'ORG'
        correct_entity_map['俄罗斯队'] = 'ORG'
        correct_entity_map['无印良品'] = 'ORG'
        correct_entity_map['南斯拉夫联盟'] = 'GPE'
        correct_entity_map['苏维埃社会主义共和国联盟'] = 'GPE'
        correct_entity_map['日本政府'] = 'GPE'
        correct_entity_map['厦门港'] = 'LOC'
        correct_entity_map['南粤'] = 'LOC'
        correct_entity_map['内陆'] = 'LOC'
        correct_entity_map['丑'] = 'O'
        correct_entity_map['美商'] = 'O'
        correct_entity_map['台商'] = 'O'
        correct_entity_map['台资'] = 'O'
        correct_entity_map['台胞'] = 'O'
        correct_entity_map['台独'] = 'O'
        correct_entity_map['福建人'] = 'O'
        correct_entity_map['外交部长'] = 'O'
        correct_entity_map['广西台'] = 'ORG'

        for char in ['华', '韩', '德', '金', '萨尔瓦多', '墨', '苏', '巴', '巴拉克', '福特', '奥尔布赖特', '乐山', '芝山岩',
                     '戈尔', '五角', '布莱尔', '怒江', '秦王', '中华', '迪斯尼', '布鲁堡', '吉安', '汤', '妈祖', '马', '万里',
                     '澎湖', '丹东', '东海', '华府', '洋', '仁爱', '高', '斯普林菲尔德', '杜邦', '丁宜洲', '洋山', '康定', '美']:
            if char in correct_entity_map.keys():
                del correct_entity_map[char]

    return correct_entity_map, remove_start_char_entity, remove_end_char_entity, remove_se_char_entity


def correct(data_path, train=False, test=False):
    """
    纠错
    接受标签形式：['O'\n, 'O'\n, 'S-...'\n, 'B-...'\n]
    @param train: 对训练集进行纠正
    @param data_path:
    @param test: 对测试集进行纠正
    """
    print('文件格式转换...')
    df = convert(data_path)
    print('返回纠正字典...')
    correct_entity_map, remove_start_char_entity, remove_end_char_entity, remove_se_char_entity = get_correct_entity_tag_map(
        df, train=train, test=test)
    total_tag_list = df['tag'].to_list()
    new_total_tag_list = []
    for tag_list in total_tag_list:
        if len(tag_list) == 0:
            new_total_tag_list.append([])
        else:
            new_total_tag_list.append(
                [[tag.split('-')[0]] + [tag.split('-')[1]] + [tag.split('-')[2]] for tag in tag_list])

    new_total_tag_list2 = []
    for tag_list in tqdm(new_total_tag_list, desc='处理中'):
        if len(tag_list) == 0:
            new_total_tag_list2.append([])
        else:
            temp = []
            for tag in tag_list:
                # 更正实体标签
                if tag[2] in correct_entity_map.keys():
                    tag[0] = correct_entity_map[tag[2]]
                # 去除实体头尾部噪声符号
                if tag[2] in remove_se_char_entity:
                    tag[2] = tag[2][1:-1]
                    tag[1] = str(int(tag[1]) + 1)
                # 去除实体头部噪声符号
                elif tag[2] in remove_start_char_entity:
                    tag[2] = tag[2][1:]
                    tag[1] = str(int(tag[1]) + 1)
                # 去除实体尾部噪声符号
                elif tag[2] in remove_end_char_entity:
                    tag[2] = tag[2][:-1]
                if '百分之' in tag[2]:
                    tag[0] = 'O'
                temp.append(tag)
            new_total_tag_list2.append(temp)

    if test:
        pd.DataFrame({'text': df['text'], 'tag': new_total_tag_list2}).to_csv('./data/correct_result.csv',
                                                                              index=False)
    elif train:
        all_train_data = pd.DataFrame({'text': df['text'], 'tag': new_total_tag_list2})
        all_train_data.to_csv('./data/correct_char_ner_train.csv', index=False)

        all_train_data = shuffle(all_train_data, random_state=0)
        all_train_data[:int(len(all_train_data) * 0.75)].to_csv('./data/correct_train.csv', index=False)
        all_train_data[int(len(all_train_data) * 0.75):].to_csv('./data/correct_valid.csv', index=False)

        resolve_tag_for_train('./data/correct_train.csv')
        resolve_tag_for_train('./data/correct_valid.csv')


if __name__ == '__main__':
    # vote_fold(more_recall=False)
    # correct('./data/char_ner_train.csv', train=True)
    # resolve_tag_for_submit('./data/correct_char_ner_train.csv')

    # show_contradictory_entity('./data/3model_merge/with_char_vote_model_result.csv')

    # correct('./data/with_char_vote_model_result.csv', test=True)
    # resolve_tag_for_submit('./data/correct_result.csv')

    # resolve_tag_for_submit('./data/correct_train.csv', train=True)
    # resolve_tag_for_submit('./data/correct_valid.csv', train=True)

    # show_all_entity('./data/correct_train_data_seed0/correct_train.enhance', tag_type=2, the_specified='LOC')

    # vote_model(more_recall=False)

    # resolve_tag_for_enhance('./data/correct_train_data_seed0/correct_train.csv')

    correct('./data/char_ner_train.csv', train=True)
