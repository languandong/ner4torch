from args import args
from total_utils.DataIterator import DataIterator, DataIterator4Classify
from total_utils.train_utils import train, classify_train
from total_utils.common import train_init
import os

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    train_init(args)
    if args.do_ner:
        train_iter = DataIterator(args, data_file=args.train_data_path, is_test=False)
        valid_iter = DataIterator(args, data_file=args.valid_data_path, is_test=True)
        train(args, train_iter, valid_iter)
    elif args.do_classify:
        train_iter = DataIterator4Classify(args, data_file=args.train_data_path, is_test=False)
        valid_iter = DataIterator4Classify(args, data_file=args.valid_data_path, is_test=True)
        classify_train(args, train_iter, valid_iter)
