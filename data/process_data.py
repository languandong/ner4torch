import pandas as pd
import numpy as np
from ast import literal_eval
import re
#
for i in range(1,6):
  data_path = './KFold_dataset/'
  if i == 1:
    val = pd.read_csv(data_path + f'fold{i}/val_result.csv')
  else:
    val_ = pd.read_csv(data_path + f'fold{i}/val_result.csv')
    val = pd.concat([val,val_],axis=0,ignore_index=True)
val.to_csv('train4.csv',index=False)


data_path = './KFold_dataset/'
test1 = pd.read_csv(data_path + f'fold1/test_result.csv')
test2 = pd.read_csv(data_path + f'fold2/test_result.csv')
test3 = pd.read_csv(data_path + f'fold3/test_result.csv')
test4 = pd.read_csv(data_path + f'fold4/test_result.csv')
test5 = pd.read_csv(data_path + f'fold5/test_result.csv')

def convert(x):
  x = re.findall("\d.\d+", x)
  return x
def convert1(x):
  x = re.findall(r"\d.\d+e-\d+", x)
  return x
def str2list(x):
  if 'e' in x:
    x = np.array(convert1(x))
  else:
    x = np.array(convert(x))
  return x
label_list = []
a = [0,0,0]
for i in range(test1.shape[0]):
  a1 = test1.loc[i,'class']
  a1 = str2list(a1)

  a2 = test2.loc[i,'class']
  a2 = str2list(a2)

  a3 = test3.loc[i, 'class']
  a3 = str2list(a3)

  a4 = test4.loc[i, 'class']
  a4 = str2list(a4)

  a5 = test5.loc[i, 'class']
  a5 = str2list(a5)

  for i in range(len(a1)):
    a[i] = float(a1[i]) + float(a2[i]) + float(a3[i]) + float(a4[i]) + float(a5[i])
  # a = a1 + a2 + a3 + a4 + a5
  # a = np.sum([a1,a2,a3,a4,a5],axis=0)
  label = np.argmax(a)
  label_list.append(label)
#
test1['class'] = label_list
test1.to_csv('test4.csv',index=False)
# test1.to_csv('train1.csv',index=False)

# for i in range(1,4):
#   if i == 1:
#     train = pd.read_csv(f'train{i}.csv')
#   else:
#     train_ = pd.read_csv(f'train{i}.csv')
#     train = pd.concat([train,train_],axis=0,ignore_index=True)
# train.to_csv('train_total.csv',index=False)
# for i in range(1,4):
#   if i == 1:
#     test = pd.read_csv(f'test{i}.csv')
#   else:
#     test_ = pd.read_csv(f'test{i}.csv')
#     test = pd.concat([test,test_],axis=0,ignore_index=True)
# test.to_csv('test_total.csv',index=False)