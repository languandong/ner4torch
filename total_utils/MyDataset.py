from torch.utils.data import Dataset
import torch
from ast import literal_eval
import numpy as np
import random
from transformers import BertTokenizer


class NeZhaDataset(Dataset):

    def __init__(self, corpus, seq_len: int, args):
        self.seq_len = seq_len
        self.args = args
        self.lines = corpus
        self.corpus_lines = self.lines.shape[0]
        self.tokenizer = BertTokenizer(
            vocab_file=args.pretrain_model_path + "vocab.txt",
            do_lower_case=args.do_lower_case)
        self.label_type = args.special_label + [j + "-" + i
                                                for i in args.entity_label
                                                for j in args.bio_label]
        self.label_map = {value: index
                          for index, value in enumerate(self.label_type)}
        print('样本数：', self.corpus_lines)

    def __len__(self):
        return self.corpus_lines

    def __getitem__(self, idx):
        text_list, label_list = self.get_sentence_and_label(idx)
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

        char_list = ["[CLS]"] + char_list + ["[SEP]"]
        segment_ids = [0] + segment_ids + [0]
        # label_ids = [self.label_map["[CLS]"]] + label_ids + [self.label_map["[SEP]"]]
        label_ids = [0] + label_ids + [0]
        token_ids = self.tokenizer.convert_tokens_to_ids(char_list)

        padding = [0 for _ in range(self.seq_len - len(label_ids))]
        attention_mask = len(token_ids) * [1] + len(padding) * [0]
        label_ids.extend(padding), token_ids.extend(padding)
        segment_ids.extend(padding), char_list.extend(['*NULL*'] * len(padding))
        attention_mask = torch.tensor(attention_mask, dtype=torch.uint8).to(self.args.device)
        token_ids = torch.tensor(token_ids, dtype=torch.long).to(self.args.device)
        segment_ids = torch.tensor(segment_ids, dtype=torch.long).to(self.args.device)
        label_ids = torch.tensor(label_ids, dtype=torch.long).to(self.args.device)
        char_list = np.array(char_list)

        assert len(token_ids) == self.seq_len
        assert len(attention_mask) == self.seq_len
        assert len(segment_ids) == self.seq_len
        assert len(label_ids) == self.seq_len

        output = {"input_ids_list": token_ids,
                  "segment_ids_list": segment_ids,
                  "input_mask_list": attention_mask,
                  "label_ids_list": label_ids,
                  "char_lists": char_list}

        return output

    def get_sentence_and_label(self, idx):

        text, label = self.lines.iloc[idx].values
        text = list(text.replace(" ", "§"))
        label = literal_eval(label)
        if len(text) > self.seq_len - 2:
            text = text[:(self.seq_len - 2)]
            label = label[:(self.seq_len - 2)]

        return text, label


class PretrainDataset(Dataset):

    def __init__(self, corpus, seq_len: int):
        self.seq_len = seq_len - 2
        self.lines = corpus
        self.corpus_lines = len(self.lines)
        self.tokenizer = BertTokenizer(
            vocab_file='/Notebook/data_801/chenhy/MODEL/nezha-cn-base/vocab.txt',
            do_lower_case=True)
        self.vocab = {index: token for token, index in self.tokenizer.get_vocab().items()}

    def __len__(self):
        return self.corpus_lines

    def __getitem__(self, idx):
        text = self.get_sentence(idx)
        text_pair = ['CLS'] + text + ['[SEP]']
        text_pair, output_ids = self.create_masked_lm_predictions(text_pair)
        token_ids = []
        for token in text_pair:
            temp_char = self.tokenizer.tokenize(token.lower())
            if temp_char:
                token_ids.append(self.tokenizer.convert_tokens_to_ids(temp_char[0]))
            else:
                token_ids.append(self.tokenizer.convert_tokens_to_ids("§"))
        segment_ids = [0] * len(token_ids)

        padding = [0 for _ in range(self.seq_len + 2 - len(token_ids))]
        padding_label = [-100 for _ in range(self.seq_len + 2 - len(token_ids))]
        attention_mask = len(token_ids) * [1] + len(padding) * [0]
        token_ids.extend(padding), output_ids.extend(padding_label), segment_ids.extend(padding)
        attention_mask = np.array(attention_mask)
        token_ids = np.array(token_ids)
        segment_ids = np.array(segment_ids)
        output_ids = np.array(output_ids)
        output = {"input_ids": token_ids,
                  "token_type_ids": segment_ids,
                  'attention_mask': attention_mask,
                  "output_ids": output_ids}
        return output

    def get_sentence(self, idx):

        text, _ = self.lines[idx]
        text = text[:self.seq_len]

        return text

    def create_masked_lm_predictions(self, text, masked_lm_prob=0.15, max_predictions_per_seq=21,
                                     rng=random.Random()):
        cand_indexes = []
        for (i, token) in enumerate(text):
            if token == '[CLS]' or token == '[SEP]':
                continue
            cand_indexes.append([i])

        output_tokens = text
        output_tokens_copy = output_tokens.copy()
        output_labels = [-100] * len(text)

        num_to_predict = min(max_predictions_per_seq,
                             max(1, int(round(len(text) * masked_lm_prob))))

        # 不同 gram 的比例  **(改为3)**
        ngrams = np.arange(1, 3 + 1, dtype=np.int64)
        pvals = 1. / np.arange(1, 3 + 1)
        pvals /= pvals.sum(keepdims=True)

        # 每个 token 对应的三个 ngram
        ngram_indexes = []
        for idx in range(len(cand_indexes)):
            ngram_index = []
            for n in ngrams:
                ngram_index.append(cand_indexes[idx:idx + n])
            ngram_indexes.append(ngram_index)
        rng.shuffle(ngram_indexes)

        masked_lms = set()
        # 获取 masked tokens
        # cand_index_set 其实就是每个 token 的三个 ngram
        # 比如：[[[13]], [[13], [14]], [[13], [14], [15]]]
        for cand_index_set in ngram_indexes:
            if len(masked_lms) >= num_to_predict:
                break
            # 根据 cand_index_set 不同长度 choice
            n = np.random.choice(
                ngrams[:len(cand_index_set)],
                p=pvals[:len(cand_index_set)] / pvals[:len(cand_index_set)].sum(keepdims=True))
            # [16, 17] = sum([[16], [17]], [])
            index_set = sum(cand_index_set[n - 1], [])
            # 处理选定的 ngram index ：80% MASK，10% 是原来的，10% 随机替换一个
            for index in index_set:
                masked_token = None
                if rng.random() < 0.8:
                    masked_token = '[MASK]'
                else:
                    if rng.random() < 0.5:
                        masked_token = text[index]
                    else:
                        masked_token = self.vocab[
                            rng.randint(0, self.tokenizer.vocab_size - (106 + 1))]  # 取不到特殊字符
                temp_char = self.tokenizer.tokenize(output_tokens[index].lower())
                if temp_char:
                    output_labels[index] = self.tokenizer.convert_tokens_to_ids(temp_char[0])
                else:
                    output_labels[index] = self.tokenizer.convert_tokens_to_ids("§")
                output_tokens_copy[index] = masked_token
                masked_lms.add(index)

        return output_tokens_copy, output_labels
