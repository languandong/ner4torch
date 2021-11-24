import pandas as pd
from total_utils.DataIterator import DataIterator, DataIterator4Classify
from total_utils.predict_utils import predict, classify_predict
from tqdm import tqdm
from args import args
import os
import numpy as np
import torch
import random
from models.model_classify import Model as CModel

def train_init():
  random.seed(args.seed)
  np.random.seed(args.seed)
  torch.manual_seed(args.seed)
  torch.cuda.manual_seed_all(args.seed)

os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
valid_iter = DataIterator4Classify(args, data_file=args.valid_data_path, is_test=True)

model = CModel(args).to(args.device)
best_model = "/home/wangzhili/LGD/ner4torch/model_save/2021-11-15_01_13_16/model_0.6061.bin"
model.load_state_dict(torch.load(best_model))
model.eval()

true, preds = [], []
pbar = tqdm(valid_iter, dynamic_ncols=True)
with torch.no_grad():
  for input_ids_list, input_mask_list, segment_ids_list, label_list in pbar:
    outputs = model.forward(input_ids_list, input_mask_list, segment_ids_list)
    pred = np.argmax(outputs.cpu().numpy(), axis=-1)
    preds.extend(pred)
    true.extend(label_list.cpu().numpy())

index = list(map(lambda x: x[0] != x[1], zip(preds, true)))
dev_csv = pd.read_csv('./data/valid.csv', header=0)
dev_csv.insert(3, 'pred', preds)
dev_csv[index].to_csv(f'{best_model.split(".")[-2]}_badcase.csv',index = False, columns=['text','class','pred'])