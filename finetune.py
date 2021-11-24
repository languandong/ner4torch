from args import args
from total_utils.DataIterator import DataIterator, DataIterator4Classify
from total_utils.train_utils import train, classify_train
import os
import datetime
import numpy as np
import torch
import random


def get_save_path():
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H_%M_%S")

    args.model_save_path = args.model_save_path + "{}/".format(timestamp)

    if not os.path.exists(args.model_save_path):
        os.makedirs(args.model_save_path)

    argsDict = args.__dict__
    # 写入超参数设置文件
    with open(args.model_save_path + 'args.txt', 'w') as f:
        f.writelines('------------------ start ------------------' + '\n')
        for eachArg, value in argsDict.items():
            f.writelines(eachArg + ' : ' + str(value) + '\n')
        f.writelines('------------------- end -------------------')


def train_init():
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    get_save_path()


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    train_init()
    if args.do_ner:
        train_iter = DataIterator(args, data_file=args.train_data_path, is_test=False)
        valid_iter = DataIterator(args, data_file=args.valid_data_path, is_test=True)
        train(args, train_iter, valid_iter)
    elif args.do_classify:
        train_iter = DataIterator4Classify(args, data_file=args.train_data_path, is_test=False)
        valid_iter = DataIterator4Classify(args, data_file=args.valid_data_path, is_test=True)
        classify_train(args, train_iter, valid_iter)
