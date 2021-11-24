import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--gpu_id', default=3, type=int)
parser.add_argument('--pre_model_type', default='ernie_gram', type=str,
                    choices=['roberta', 'nezha_wwm', 'nezha_base', 'ernie_gram'])
parser.add_argument('--struc', default='cls', type=str,
                    choices=['cls', 'bilstm', 'bigru', 'idcnn'],
                    help="下接结构")
parser.add_argument('--epoch', default=10, type=int)
parser.add_argument('--extra_epoch', default=0, type=int, help='延缓学习率衰减')
parser.add_argument('--batch_size', default=32, type=int)
parser.add_argument('--sequence_length', default=128, type=int)
parser.add_argument('--seed', default=42, type=int)
parser.add_argument('--dropout_rate', default=0.2, type=float)
parser.add_argument("--dropout_num", default=0, type=int,help="dropout数量")
parser.add_argument('--warmup_prop', default=0.1, type=float)
parser.add_argument('--use_dynamic_fusion', action='store_true', default=False)
parser.add_argument('--use_avgpool', action='store_true', default=False)
parser.add_argument('--use_crf', default=True, type=bool)
parser.add_argument('--crf_learning_rate', default=1e-3, type=float, help='CRF的学习率')
parser.add_argument('--bert_learning_rate', default=2e-5, type=float, help='BERT的微调学习率')
parser.add_argument('--bert_hidden_size', default=768, type=int)
parser.add_argument("--lstm_dim", default=256, type=int,help="lstm隐藏状态维度")
parser.add_argument("--gru_dim", default=256, type=int, help="gru隐藏状态维度")
parser.add_argument('--entity_label', default=['BANK', 'PRODUCT', 'COMMENTS_N', 'COMMENTS_ADJ'], type=list)
parser.add_argument('--special_label', default=['O'], type=list)
parser.add_argument('--bio_label', default=["B", "I"], type=list)
parser.add_argument('--relation_num', default=9, type=int)
parser.add_argument('--class_num', default=3, type=int)
parser.add_argument('--origin_data_dir', default="./data/", type=str)
parser.add_argument('--model_save_path', default="./model_save/", type=str)
parser.add_argument('--do_lower_case', default=True, type=bool)
parser.add_argument('--best_model', default="/home/wangzhili/LGD/ner4torch/model_save/2021-11-20_11_01_32/model_0.6061.bin", type=str)
parser.add_argument('--train_data_path', default=[
                                                     "./data/train.csv",
                                                  #"./data/train_data_public.csv",
                                                  './data/fake_train_0.95.csv'
                                                  # './data/fake_ernie_gram_top2200.csv',
                                                  #  './data/fake_ernie_gram_0.95.csv',
                                                  #"/home/wangzhili/LGD/ner4torch/data/fake_ernie_gram_top2200.csv"
                                                  ], type=list)
parser.add_argument('--valid_data_path', default=["./data/valid.csv",
                                                  # './data/fake_ernie_gram_top2200.csv'
                                                  ], type=list)
parser.add_argument('--test_data_path', default=["./data/test_public.csv"], type=list)
parser.add_argument('--result_data_path', default="./data/result.csv", type=str)
parser.add_argument('--result_with_char_data_path', default="./data/result_with_char.csv", type=str)
parser.add_argument('--device', default="cuda", type=str)
parser.add_argument("--use_fgm", action='store_true', default=True)
parser.add_argument('--fgm_epsilon', default=0.3, type=float)
parser.add_argument("--do_ner", action='store_true', default=False)
parser.add_argument("--do_classify", action='store_true', default=True)

args = parser.parse_args()

pretrain_model_path = {'nezha_base': '/home/wangzhili/YangYang/pretrainModel/nezha-cn-base/',
                       'nezha_wwm': '/home/wangzhili/YangYang/pretrainModel/nezha-cn-wwm/',
                       'roberta': '/home/wangzhili/YangYang/pretrainModel/chinese_roberta_wwm_ext_pytorch/',
                       'ernie_gram': '/home/wangzhili/YangYang/pretrainModel/ernie-gram/'}
