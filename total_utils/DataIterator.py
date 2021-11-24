from total_utils.DataLoader import create_example, create_classify_example
import numpy as np
import torch
from transformers import BertTokenizer
from args import pretrain_model_path

# ner数据迭代器
class DataIterator(object):
    def __init__(self, args, data_file, is_test=False, is_predict=False):
        # args 参数传递
        self.device = args.device
        self.batch_size = args.batch_size
        self.sequence_length = args.sequence_length
        self.tokenizer = BertTokenizer(
            vocab_file=pretrain_model_path[args.pre_model_type] + "vocab.txt",
            do_lower_case=args.do_lower_case)
        self.label_type = args.special_label + [j + "-" + i
                                                for i in args.entity_label
                                                for j in args.bio_label]
        self.label_map = {value: index for index, value in enumerate(self.label_type)}

        # 数据的操作
        self.data = create_example(data_file, is_predict=is_predict) # 样本对象列表 [Object, Object.....]
        self.num_records = len(self.data)  # 数据的个数
        self.id_count = 0  # index
        self.all_idx = list(range(self.num_records))  # 全体数据索引
        self.is_test = is_test

        if not self.is_test:
            self.shuffle()
        print("标签个数：", len(self.label_map))
        print("样本个数：", self.num_records)

    def get_text_label(self, example_idx):
        text_list = list(self.data[example_idx].text.replace(" ", "§"))
        label_list = self.data[example_idx].label
        return text_list, label_list

    def convert_single_example(self, example_idx):
        text_list, label_list = self.get_text_label(example_idx)
        # 超过长度的截断
        if len(text_list) > self.sequence_length - 2:
            text_list = text_list[:(self.sequence_length - 2)]
            label_list = label_list[:(self.sequence_length - 2)]
        # 组织输入bert的数据
        char_list, segment_ids, label_ids = [], [], []
        for char, label in zip(text_list, label_list):
            temp_char = self.tokenizer.tokenize(char.lower())
            if temp_char:
                char_list.append(temp_char[0])
            else:
                # print("unknown character{},use § replace !!!".format(char))
                char_list.append("§")
            segment_ids.append(0)
            label_ids.append(self.label_map[label])
        # 组织输入bert的数据
        char_list = ["[CLS]"] + char_list + ["[SEP]"]
        segment_ids = [0] + segment_ids + [0]
        label_ids = [0] + label_ids + [0]
        input_ids = self.tokenizer.convert_tokens_to_ids(char_list)
        input_mask = [1] * len(input_ids)
        # padding到指定长度
        while len(input_ids) < self.sequence_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)
            # label_ids.append(self.label_map["[PAD]"])
            label_ids.append(0)
            char_list.append("*NULL*")

        assert len(input_ids) == self.sequence_length
        assert len(input_mask) == self.sequence_length
        assert len(segment_ids) == self.sequence_length
        assert len(label_ids) == self.sequence_length

        return input_ids, input_mask, segment_ids, label_ids, char_list

    def shuffle(self):
        np.random.shuffle(self.all_idx)

    def __iter__(self):
        return self

    def __next__(self):
        if self.id_count >= self.num_records:  # Stop iteration condition
            self.id_count = 0
            if not self.is_test:
                self.shuffle()
            raise StopIteration

        input_ids_list, input_mask_list, segment_ids_list, label_ids_list, char_lists = [], [], [], [], []
        batch_count = 0
        # 组织一个batch的数据
        while batch_count < self.batch_size:
            idx = self.all_idx[self.id_count]
            input_ids, input_mask, segment_ids, label_ids, char_list = self.convert_single_example(idx)
            # 写入batch
            input_ids_list.append(input_ids)
            input_mask_list.append(input_mask)
            segment_ids_list.append(segment_ids)
            label_ids_list.append(label_ids)
            char_lists.append(char_list)
            batch_count += 1
            self.id_count += 1

            if self.id_count >= self.num_records:
                break
        # 转为torch数据形式
        input_ids_list = torch.tensor([i for i in input_ids_list], dtype=torch.long).to(self.device)
        input_mask_list = torch.tensor([i for i in input_mask_list], dtype=torch.uint8).to(self.device)
        segment_ids_list = torch.tensor([i for i in segment_ids_list], dtype=torch.long).to(self.device)
        label_ids_list = torch.tensor([i for i in label_ids_list], dtype=torch.long).to(self.device)

        return input_ids_list, input_mask_list, segment_ids_list, label_ids_list, char_lists

    def __len__(self):
        if len(self.data) % self.batch_size == 0:
            return len(self.data) // self.batch_size
        else:
            return len(self.data) // self.batch_size + 1

# 分类任务数据迭代器
class DataIterator4Classify(object):
    """
    data iterator
    """
    def __init__(self, args, data_file, is_test=False, is_predict=False):
        # args 参数传递
        self.device = args.device
        self.batch_size = args.batch_size
        self.sequence_length = args.sequence_length
        self.tokenizer = BertTokenizer(
            vocab_file=pretrain_model_path[args.pre_model_type] + "vocab.txt",
            do_lower_case=args.do_lower_case)
        # 数据的操作
        self.data = create_classify_example(data_file, is_predict=is_predict)
        self.num_records = len(self.data)  # 数据的个数
        self.id_count = 0  # index
        self.all_idx = list(range(self.num_records))  # 全体数据索引
        self.is_test = is_test
        # 训练集打乱数据
        if not self.is_test:
            self.shuffle()

        print("标签个数：", args.class_num)
        print("样本个数：", self.num_records)
    # 获取样本的文本 标签
    def get_text_label(self, example_idx):
        text_list = list(self.data[example_idx].text.replace(" ", "§"))
        label = self.data[example_idx].label
        return text_list, label
    # 得到单个样本的bert输入
    def convert_single_example(self, example_idx):
        text_list, label = self.get_text_label(example_idx)

        if len(text_list) > self.sequence_length - 2:
            text_list = text_list[:(self.sequence_length - 2)]

        char_list, segment_ids = [], []
        for char in text_list:
            temp_char = self.tokenizer.tokenize(char.lower())
            if temp_char:
                char_list.append(temp_char[0])
            else:
                # print("unknown character{},use § replace !!!".format(char))
                char_list.append("§")

            segment_ids.append(0)

        char_list = ["[CLS]"] + char_list + ["[SEP]"]
        segment_ids = [0] + segment_ids + [0]
        input_ids = self.tokenizer.convert_tokens_to_ids(char_list)
        input_mask = [1] * len(input_ids)

        while len(input_ids) < self.sequence_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)
            char_list.append("*NULL*")

        assert len(input_ids) == self.sequence_length
        assert len(input_mask) == self.sequence_length
        assert len(segment_ids) == self.sequence_length

        return input_ids, input_mask, segment_ids, label

    def shuffle(self):
        np.random.shuffle(self.all_idx)

    def __iter__(self):
        return self

    def __next__(self):
        # 组织batch数据
        if self.id_count >= self.num_records:  # Stop iteration condition
            self.id_count = 0
            if not self.is_test:
                self.shuffle()
            raise StopIteration

        input_ids_list, input_mask_list, segment_ids_list, label_list = [], [], [], []
        batch_count = 0
        while batch_count < self.batch_size:
            idx = self.all_idx[self.id_count]
            input_ids, input_mask, segment_ids, label = self.convert_single_example(idx)

            input_ids_list.append(input_ids)
            input_mask_list.append(input_mask)
            segment_ids_list.append(segment_ids)
            label_list.append(label)
            batch_count += 1
            self.id_count += 1

            if self.id_count >= self.num_records:
                break

        input_ids_list = torch.tensor([i for i in input_ids_list], dtype=torch.long).to(self.device)
        input_mask_list = torch.tensor([i for i in input_mask_list], dtype=torch.uint8).to(self.device)
        segment_ids_list = torch.tensor([i for i in segment_ids_list], dtype=torch.long).to(self.device)
        label_list = torch.tensor([i for i in label_list], dtype=torch.long).to(self.device)

        return input_ids_list, input_mask_list, segment_ids_list, label_list

    def __len__(self):

        if len(self.data) % self.batch_size == 0:
            return len(self.data) // self.batch_size
        else:
            return len(self.data) // self.batch_size + 1
