import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold
import os

if __name__ == '__main__':
  # 加载数据集
  dataset = pd.read_csv('../data/train_data_public.csv')
  # fake_dataset = pd.read_csv('../data/fake_train_0.9.csv')
  # fake_dataset.insert(2,'BIO_anno',[0] * fake_dataset.shape[0])
  # dataset = dataset.append(fake_dataset)
  kfold_data_path = '../data/KFold_dataset/'
  kf = KFold(n_splits=5, shuffle=True, random_state=30)

  i = 1
  for train_index, valid_index in kf.split(dataset.iloc[:,1:3],dataset.iloc[:,3]):
    train = dataset.iloc[train_index]
    val = dataset.iloc[valid_index]
    save_path = kfold_data_path + 'fold' + str(i) + '/'
    if not os.path.exists(save_path):
      os.mkdir(save_path)
    train.to_csv(save_path + 'train.csv', index=False)
    val.to_csv(save_path + 'val.csv', index=False)
    i += 1
