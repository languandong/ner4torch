import pandas as pd


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text, label):
        self.guid = guid
        self.text = text
        self.label = label


def load_data(file_path, is_predict=False):
    """读取数据"""
    df = pd.read_csv(file_path)
    lines = []
    if is_predict:
        for _, item in df.iterrows():
            text = item['text']
            label = ['O' for _ in range(len(text))]
            lines.append((text, label))
    else:
        for _, item in df.iterrows():
            text = item['text']
            label = item['BIO_anno'].strip().split(' ')
            assert len(text) == len(label)

            lines.append((text, label))

    return lines


def load_classify_data(file_path, is_predict=False):
    """读取数据"""
    df = pd.read_csv(file_path)
    lines = []
    if is_predict:
        for _, item in df.iterrows():
            text = item['text']
            label = 3
            lines.append((text, label))
    else:
        for _, item in df.iterrows():
            text = item['text']
            label = int(item['class'])
            lines.append((text, label))



    return lines

# 创建ner任务的数据对象
def create_example(file_path, is_predict=False):
    """put data into example """
    example = [] # 存储样本对象
    lines = []
    for path in file_path:
        lines += load_data(path, is_predict) # 读入数据[ ('',[]), ('',[])...]
    file_type = file_path[0].split('/')[-1].split('.')[0]

    for index, content in enumerate(lines):
        # 解析数据
        guid = "{0}_{1}".format(file_type, str(index))
        text = content[0]
        label = content[1]
        # 生成样本对象存入列表
        example.append(InputExample(guid=guid, text=text, label=label))

    return example

# 创建分类任务的数据对象
def create_classify_example(file_path, is_predict=False):
    example = []
    lines = []
    for path in file_path:
        lines += load_classify_data(path, is_predict)
    file_type = file_path[0].split('/')[-1].split('.')[0]

    for index, content in enumerate(lines):
        guid = "{0}_{1}".format(file_type, str(index))
        text = content[0]
        label = content[1]
        example.append(InputExample(guid=guid, text=text, label=label))

    return example
