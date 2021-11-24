from args import args
from total_utils.DataIterator import DataIterator, DataIterator4Classify
from total_utils.predict_utils import predict, classify_predict, creadte_fake_label, creadte_fake_label_topK,classify_predict_with_can
import os


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    if args.do_ner:
        test_iter = DataIterator(args, data_file=args.test_data_path, is_test=True, is_predict=True)
        predict(args, test_iter)
    elif args.do_classify:
        test_iter = DataIterator4Classify(args, data_file=args.test_data_path, is_test=True, is_predict=True)
        # classify_predict(args, test_iter)
        classify_predict_with_can(args, test_iter)
        # creadte_fake_label(args,test_iter,prob=0.95)
        # creadte_fake_label_topK(args,test_iter,k=2200)
