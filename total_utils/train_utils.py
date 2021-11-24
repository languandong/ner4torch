from tqdm import tqdm
from transformers import AdamW, get_linear_schedule_with_warmup
from models.model_crf import Model
from models.model_classify import Model as CModel
import torch
import copy
from total_utils.common import *
import torch.nn as nn
from sklearn.metrics import f1_score, classification_report, cohen_kappa_score, accuracy_score
import numpy as np
from torch.nn import functional as F

class MultiFocalLoss(nn.Module):
    def __init__(self, num_class=3, alpha=[0.105,0.038,0.856], gamma=2, balance_index=-1, smooth=None, size_average=True):
        super(MultiFocalLoss, self).__init__()
        self.num_class = num_class
        self.alpha = alpha
        self.gamma = gamma
        self.smooth = smooth
        self.size_average = size_average

        if self.alpha is None:
            self.alpha = torch.ones(self.num_class, 1)
        elif isinstance(self.alpha, (list, np.ndarray)):
            assert len(self.alpha) == self.num_class
            self.alpha = torch.FloatTensor(alpha).view(self.num_class, 1)
            self.alpha = self.alpha / self.alpha.sum()
        elif isinstance(self.alpha, float):
            alpha = torch.ones(self.num_class, 1)
            alpha = alpha * (1 - self.alpha)
            alpha[balance_index] = self.alpha
            self.alpha = alpha
        else:
            raise TypeError('Not support alpha type')

        if self.smooth is not None:
            if self.smooth < 0 or self.smooth > 1.0:
                raise ValueError('smooth value should be in [0,1]')

    def forward(self, input, target):
        logit = F.softmax(input, dim=1)

        if logit.dim() > 2:
            # N,C,d1,d2 -> N,C,m (m=d1*d2*...)
            logit = logit.view(logit.size(0), logit.size(1), -1)
            logit = logit.permute(0, 2, 1).contiguous()
            logit = logit.view(-1, logit.size(-1))
        target = target.view(-1, 1)

        # N = input.size(0)
        # alpha = torch.ones(N, self.num_class)
        # alpha = alpha * (1 - self.alpha)
        # alpha = alpha.scatter_(1, target.long(), self.alpha)
        epsilon = 1e-10
        alpha = self.alpha
        if alpha.device != input.device:
            alpha = alpha.to(input.device)

        idx = target.cpu().long()
        one_hot_key = torch.FloatTensor(target.size(0), self.num_class).zero_()
        one_hot_key = one_hot_key.scatter_(1, idx, 1)
        if one_hot_key.device != logit.device:
            one_hot_key = one_hot_key.to(logit.device)

        if self.smooth:
            one_hot_key = torch.clamp(
                one_hot_key, self.smooth, 1.0 - self.smooth)
        pt = (one_hot_key * logit).sum(1) + epsilon
        logpt = pt.log()

        gamma = self.gamma

        alpha = alpha[idx]
        loss = -1 * alpha * torch.pow((1 - pt), gamma) * logpt

        if self.size_average:
            loss = loss.mean()
        else:
            loss = loss.sum()
        return loss

def classify_train(args, train_iter, dev_iter):
    logger = get_logger(args.model_save_path + "/log.log")
    model = CModel(args).to(args.device)

    bert_param_optimizer = list(model.pre_model.named_parameters())
    linear_param_optimizer = list(model.hidden2label.named_parameters())

    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        # pre-train model
        {'params': [p for n, p in bert_param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01,
         "lr": args.bert_learning_rate},
        {'params': [p for n, p in bert_param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0,
         "lr": args.bert_learning_rate},
        # linear layer
        {'params': [p for n, p in linear_param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01,
         "lr": args.bert_learning_rate},
        {'params': [p for n, p in linear_param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0,
         "lr": args.bert_learning_rate},
    ]

    optimizer = AdamW(optimizer_grouped_parameters,
                      betas=(0.9, 0.98),  # according to RoBERTa paper betas=(0.9, 0.98),
                      lr=args.bert_learning_rate,
                      eps=1e-8
                      )
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=len(train_iter) * args.epoch * args.warmup_prop,
                                                num_training_steps=len(train_iter) * (args.epoch + args.extra_epoch))
    criterion = nn.CrossEntropyLoss()
    # criterion = MultiFocalLoss()
    fgm = None
    if args.use_fgm:
        fgm = FGM(model)
    model.zero_grad()

    count, epoch_step, best_kappa = 0, 0, 0
    for i in range(args.epoch):
        model.train()
        pbar = tqdm(train_iter, position=0, dynamic_ncols=True)
        for input_ids_list, input_mask_list, segment_ids_list, label_list in pbar:
            optimizer.zero_grad()
            logits = model.forward(input_ids_list, input_mask_list, segment_ids_list)
            loss = criterion(logits, label_list)
            loss.backward()

            if args.use_fgm:
                fgm.attack(epsilon=args.fgm_epsilon)
                logits_adv = model.forward(input_ids_list, input_mask_list, segment_ids_list)
                loss_adv = criterion(logits_adv, label_list)
                loss_adv.backward()
                fgm.restore()

            optimizer.step()
            scheduler.step()  # Update learning rate schedule

            pbar.set_description(
                f"epoch:{i+1}/{args.epoch} lr:{optimizer.state_dict()['param_groups'][0]['lr']:.7f} classify_loss:{loss:.5f}")

        # 验证
        f1, kappa, report, accuracy = classify_valid(model, dev_iter)
        logger.info(f'dev set: epoch_{i+1}, acc_{accuracy:.4f} f1_{f1:.4f}, kappa_{kappa:.4f},\nreport:\n{report}')
        count += 1
        torch.save(model.state_dict(),
                   args.model_save_path + "/model_{:.4f}.bin".format(kappa))
        logger.info("save model in {}".format(
            args.model_save_path + "/model_{:.4f}.bin".format(kappa)))
        if kappa > best_kappa:
            best_kappa = kappa
            count = 0
            torch.save(model.state_dict(), args.model_save_path + f"/model_{best_kappa:.4f}.bin")
            logger.info("save model in {}".format(args.model_save_path + f"/model_{best_kappa:.4f}.bin"))
        if count >= 3:
            print('效果不增,终止训练')
            break


def classify_valid(model, dev_iter):
    model.eval()
    true, preds = [], []
    pbar = tqdm(dev_iter, dynamic_ncols=True)
    with torch.no_grad():
        for input_ids_list, input_mask_list, segment_ids_list, label_list in pbar:
            outputs = model(input_ids_list, input_mask_list, segment_ids_list)
            pred = np.argmax(outputs.cpu().numpy(), axis=-1)
            true.extend(label_list.cpu())
            preds.extend(pred)
    f1 = f1_score(true, preds, average='macro')
    kappa = cohen_kappa_score(true, preds)
    report = classification_report(true, preds)
    accuracy = accuracy_score(true, preds)

    return f1, kappa, report, accuracy


def train(args, train_iter, dev_iter):
    logger = get_logger(args.model_save_path + "/log.log")
    model = Model(args).to(args.device)

    bert_param_optimizer = list(model.pre_model.named_parameters())
    linear_param_optimizer = list(model.hidden2label.named_parameters())

    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        # pre-train model
        {'params': [p for n, p in bert_param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01,
         "lr": args.bert_learning_rate},
        {'params': [p for n, p in bert_param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0,
         "lr": args.bert_learning_rate},
        # linear layer
        {'params': [p for n, p in linear_param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01,
         "lr": args.crf_learning_rate},
        {'params': [p for n, p in linear_param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0,
         "lr": args.crf_learning_rate},
    ]

    if args.use_crf:
        crf_param_optimizer = list(model.crf_layer.named_parameters())
        crf_parameters = [
            {'params': [p for n, p in crf_param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01,
             "lr": args.crf_learning_rate},
            {'params': [p for n, p in crf_param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0,
             "lr": args.crf_learning_rate}
        ]
        optimizer_grouped_parameters += crf_parameters

    optimizer = AdamW(optimizer_grouped_parameters,
                      betas=(0.9, 0.98),  # according to RoBERTa paper betas=(0.9, 0.98),
                      lr=args.bert_learning_rate,
                      eps=1e-8
                      )
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=len(train_iter) * args.epoch * args.warmup_prop,
                                                num_training_steps=len(train_iter) * (args.epoch + args.extra_epoch))

    fgm = None
    if args.use_fgm:
        fgm = FGM(model)
    model.zero_grad()

    cum_step, epoch_step, best_f1 = 0, 0, 0
    # 训练
    for i in range(args.epoch):
        model.train()
        pbar = tqdm(train_iter, position=0, dynamic_ncols=True)
        for input_ids_list, input_mask_list, segment_ids_list, label_ids_list, char_lists in pbar:
            optimizer.zero_grad()
            loss = model.forward(input_ids_list, input_mask_list, segment_ids_list, label_ids_list)
            loss.backward()

            if args.use_fgm:
                fgm.attack(epsilon=args.fgm_epsilon)
                loss_adv = model.forward(input_ids_list, input_mask_list, segment_ids_list, label_ids_list)
                loss_adv.backward()
                fgm.restore()

            optimizer.step()
            scheduler.step()  # Update learning rate schedule
            cum_step += 1
            pbar.set_description(
                f"训练中：ner_loss:{loss:.3f}")

        P, R, f1 = valid(args, model, dev_iter, logger)
        logger.info(
            'dev set: epoch_{}, step_{}, f1_{:.4f}'.format(i, cum_step, f1))
        if f1 > best_f1:
            best_f1 = f1
            torch.save(model.state_dict(),
                       args.model_save_path + "/model_{:.4f}_{}.bin".format(best_f1, str(cum_step)))
            logger.info("save model in {}".format(
                args.model_save_path + "/model_{:.4f}_{}.bin".format(best_f1, str(cum_step))))

        epoch_step += 1


def valid(args, model, dev_iter, logger):
    y_pred_label_list, y_true_label_list = [], []
    model.eval()
    with torch.no_grad():
        for input_ids_list, input_mask_list, segment_ids_list, label_ids_list, char_lists in tqdm(
                dev_iter, position=0,
                dynamic_ncols=True,
                desc='测试中'):
            predict = model.forward(input_ids_list, input_mask_list, segment_ids_list)

            temp_predict_list = make_lables(args, predict, char_lists)
            temp_true_list = make_lables(args, label_ids_list.cpu().numpy(), char_lists)

            y_true_label_list.extend(temp_true_list)
            y_pred_label_list.extend(temp_predict_list)

    f1, P, R = get_entity_evaluate(args, y_pred_label_list, y_true_label_list, logger)

    return P, R, f1


def get_entity_evaluate(args, y_pred_label_list, y_ture_label_list, logger):
    TP_list, true_num_list, pre_num_list = [0] * len(args.entity_label), [0] * len(args.entity_label), [0] * len(
        args.entity_label)
    for index, items in enumerate(y_pred_label_list):
        predict_dict = y_pred_label_list[index]
        ture_dict = y_ture_label_list[index]
        for label_index, label_type in enumerate(args.entity_label):
            predict_list = [str(i["start"]) + "_" + str(i["end"]) + "_" + str(i["content"]) for i in
                            predict_dict[label_type]]
            ture_list = [str(i["start"]) + "_" + str(i["end"]) + "_" + str(i["content"]) for i in ture_dict[label_type]]

            ture_list_change = copy.deepcopy(ture_list)
            for y_pred in predict_list:
                if y_pred in ture_list_change:
                    ture_list_change.remove(y_pred)
                    TP_list[label_index] += 1
            true_num_list[label_index] += len(ture_list)
            pre_num_list[label_index] += len(predict_list)
    P_list, R_list, f1_list = get_single_entity(TP_list, true_num_list, pre_num_list)
    for index, _ in enumerate(args.entity_label):
        logger.info(
            "{}-->true_num:{},pre_num:{},TP_num:{},P:{:.4f},R:{:.4f},f1:{:.4f}".format(args.entity_label[index],
                                                                                       true_num_list[index],
                                                                                       pre_num_list[index],
                                                                                       TP_list[index], P_list[index],
                                                                                       R_list[index], f1_list[index]))
    P, R, f1 = get_total_entity(TP_list, true_num_list, pre_num_list)
    return f1, P, R


def get_single_entity(TP_list, true_num_list, pre_num_list):
    P_list, R_list, f1_list = [0] * len(TP_list), [0] * len(TP_list), [0] * len(TP_list)
    for index, _ in enumerate(TP_list):
        P_list[index] = TP_list[index] / pre_num_list[index] if pre_num_list[index] != 0 else 0
        R_list[index] = TP_list[index] / true_num_list[index] if true_num_list[index] != 0 else 0
        f1_list[index] = 2 * P_list[index] * R_list[index] / (P_list[index] + R_list[index]) if (P_list[index] + R_list[
            index]) != 0 else 0
    return P_list, R_list, f1_list


def get_total_entity(TP_list, true_num_list, pre_num_list):
    TP = sum(TP_list)
    true_num = sum(true_num_list)
    pre_num = sum(pre_num_list)
    P = TP / pre_num if pre_num != 0 else 0
    R = TP / true_num if pre_num != 0 else 0

    f1 = 2 * P * R / (P + R) if (P + R) != 0 else 0
    return P, R, f1


def make_lables(args, label_ids_list, tokens_list):
    batch_bio_list = []
    label_type = args.special_label + [j + "-" + i for i in args.entity_label for j in
                                       args.bio_label]
    id2label = {index: value for index, value in enumerate(label_type)}
    for index, item in enumerate(label_ids_list):
        label_values_list = [id2label[i] for i in label_ids_list[index]]
        temp_dict = make_label(args, label_ids_list[index], tokens_list[index], label_values_list)
        batch_bio_list.append(temp_dict)
    return batch_bio_list


def make_label(args, label_list, token_list, label_values_list):
    label_dict = {label_type: [] for label_type in args.entity_label}
    for char_index, (type_index, type_name) in enumerate(zip(label_list, label_values_list)):
        if type_index >= 1:
            if type_name.split('-')[0] == 'B':
                start_index = char_index
                end_index = start_index + 1
                if len(label_list) <= end_index:
                    break
                while label_values_list[end_index].split('-')[0] == 'I':
                    end_index += 1
                    if len(label_list) <= end_index:
                        break
                temp_dict = {"start": start_index, "end": end_index,
                             "content": "".join(token_list[start_index: end_index + 1])}
                label_dict[type_name.split('-')[1]].append(temp_dict)

    return label_dict
