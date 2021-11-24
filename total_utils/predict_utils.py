import torch
from tqdm import tqdm
import pandas as pd
from models.model_crf import Model
from models.model_classify import Model as CModel
from models.model_classify import Model_outputFake as CModel_outputFake
import numpy as np

# 分类预测
def classify_predict(args, test_iter):
    model = CModel(args).to(args.device)
    model.load_state_dict(torch.load(args.best_model))
    print("read model from {}".format(args.best_model))
    model.eval()
    preds = []
    with torch.no_grad():
        for input_ids_list, input_mask_list, segment_ids_list, _ in tqdm(test_iter, position=0,
                                                                         dynamic_ncols=True, desc='测试中'):
            outputs = model.forward(input_ids_list, input_mask_list, segment_ids_list).cpu().numpy()
            pred = np.argmax(outputs, axis=-1)
            preds.extend(pred)

    result = pd.read_csv(args.result_data_path)
    result['class'] = preds
    result.to_csv(args.result_data_path, index=False)

# can后处理的预测
def classify_predict_with_can(args, test_iter):
    model = CModel_outputFake(args).to(args.device)
    model.load_state_dict(torch.load(args.best_model))
    print("read model from {}".format(args.best_model))
    model.eval()
    # preds = []

    count = 0
    with torch.no_grad():
        for input_ids_list, input_mask_list, segment_ids_list, _ in tqdm(test_iter, position=0,
                                                                         dynamic_ncols=True, desc='测试中'):
            model_output = model.forward(input_ids_list, input_mask_list, segment_ids_list).cpu().numpy()

            if count == 0:
                outputs = model_output
                count += 1
            else:
                outputs = np.vstack((outputs, model_output))

            # pred = np.argmax(outputs, axis=-1)
            # preds.extend(pred)

    outputs = prob_postprocess(outputs)
    preds = np.argmax(outputs,axis=-1)

    result = pd.read_csv(args.result_data_path)
    result['class'] = preds
    result.to_csv(args.result_data_path, index=False)
# 生成指定置信度伪标签
def creadte_fake_label(args, test_iter, prob=0.95):
    model = CModel_outputFake(args).to(args.device)
    model.load_state_dict(torch.load(args.best_model))
    model.eval()
    preds = []
    with torch.no_grad():
        for input_ids_list, input_mask_list, segment_ids_list, _ in tqdm(test_iter, position=0,
                                                                         dynamic_ncols=True, desc='输出伪标签'):
            outputs = model.forward(input_ids_list, input_mask_list, segment_ids_list)
            preds.extend(outputs.cpu().numpy())
    # 记录达标的伪标签样本
    fake_index,fake_label = [False]*len(preds) , []
    for index, value in enumerate(preds):
        flag = False
        for i in range(3):
            if value[i] >= prob:
                flag = True
        if flag:
            fake_index[index] = True
            fake_label.append(np.argmax(value))
    # 处理输出
    result = pd.read_csv(args.test_data_path[0])
    result['class'] = preds

    result = result[fake_index]
    result['class'] = fake_label
    result.to_csv(f'./data/fake_{args.pre_model_type}_{prob}.csv', index=False)
# 生成置信度topK条伪标签
def creadte_fake_label_topK(args, test_iter, k=2200):
    model = CModel_outputFake(args).to(args.device)
    model.load_state_dict(torch.load(args.best_model))
    model.eval()
    preds = []
    with torch.no_grad():
        for input_ids_list, input_mask_list, segment_ids_list, _ in tqdm(test_iter, position=0,
                                                                         dynamic_ncols=True, desc='输出伪标签'):
            outputs = model.forward(input_ids_list, input_mask_list, segment_ids_list)
            preds.extend(outputs.cpu().numpy())

    # 记录达标的伪标签样本
    fake_index,fake_label = [False]*len(preds) , []
    # {index:max_prob....}
    count = {}
    for index, item in enumerate(preds):
        count[index] = np.max(item)
    # 倒序排列
    sort_result = sorted(count.items(), key=lambda kv:(kv[1],kv[0]),reverse=True)
    # 记录前K条的索引
    index_list = []
    index_count = 0
    for item in sort_result:
        index_list.append(item[0])
        index_count += 1
        if index_count >= 2200:
            break

    for index, value in enumerate(preds):
        if index in index_list:
            fake_index[index] = True
            fake_label.append(np.argmax(value))

    # 处理输出
    result = pd.read_csv(args.test_data_path[0])
    result = result[fake_index]
    result['class'] = fake_label
    result.to_csv(f'./data/fake_{args.pre_model_type}_top{k}.csv', index=False)
# can后处理
def prob_postprocess(y_pred):
    prior = np.array([0.105605738575983, 0.03825717321997875, 0.8561370882040382])

    # 取前K个概率
    k = 3
    y_pred_topk = np.sort(y_pred, axis=1)[:, -k:]
    y_pred_topk /= y_pred_topk.sum(axis=1, keepdims=True)
    y_pred_uncertainty = -(y_pred * np.log(y_pred)).sum(1) / np.log(k)

    threshold = 0.85
    y_pred_confident = y_pred[y_pred_uncertainty < threshold]
    y_pred_unconfident = y_pred[y_pred_uncertainty >= threshold]

    right, alpha, iters = 0, 1, 1
    post = []
    for i, y in enumerate(y_pred_unconfident):
        Y = np.concatenate([y_pred_confident, y[None]], axis=0)
        for j in range(iters):
            Y = Y ** alpha
            Y /= Y.sum(axis=0, keepdims=True)
            Y *= prior[None]
            Y /= Y.sum(axis=1, keepdims=True)
        y = Y[-1]
        post.append(y.tolist())

    post = np.array(post)
    if len(post) == 0:
        print('空')
        return y_pred
    print(post)
    y_pred[y_pred_uncertainty >= threshold] = post

    return y_pred
# ner预测
def predict(args, test_iter):
    model = Model(args).to(args.device)
    model.load_state_dict(torch.load(args.best_model))
    print("read model from {}".format(args.best_model))
    model.eval()
    y_pred_label_list = []
    with torch.no_grad():
        for input_ids_list, input_mask_list, segment_ids_list, _, _ in tqdm(test_iter, position=0,
                                                                            ncols=200, desc='测试中'):
            predict = model.forward(input_ids_list, input_mask_list, segment_ids_list)
            temp_predict_list = make_lables(predict, args)
            y_pred_label_list.extend(temp_predict_list)

    origin_text_list = pd.read_csv(args.test_data_path[0])['text'].to_list()

    tags = []
    for tag, text in zip(y_pred_label_list, origin_text_list):
        tag = tag[1:-1]
        pad = len(text) - len(tag)
        tag = tag + ['O' for _ in range(pad)]
        tags.append(' '.join(tag))
    ids = [i for i in range(len(tags))]
    result = pd.DataFrame({'id': ids, 'BIO_anno': tags})
    result['class'] = [3 for _ in range(len(ids))]
    result.to_csv(args.result_data_path, index=False)


def make_lables(label_ids_list, args):
    batch_bio_list = []
    label_type = args.special_label + [j + "-" + i for i in args.entity_label for j in
                                       args.bio_label]
    trigger_id2label = {index: value for index, value in enumerate(label_type)}
    for index, item in enumerate(label_ids_list):
        temp_list = [trigger_id2label[i] for i in item]
        batch_bio_list.append(temp_list)
    return batch_bio_list
