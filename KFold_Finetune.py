from args import args
from total_utils.DataIterator import DataIterator, DataIterator4Classify
from total_utils.train_utils import train, classify_train
from total_utils.predict_utils import predict, classify_predict
from tqdm import tqdm
from transformers import AdamW, get_linear_schedule_with_warmup
from models.model_classify import Model as CModel
from total_utils.common import *
import torch.nn as nn
from sklearn.metrics import f1_score, classification_report, cohen_kappa_score



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


def valid(model, dev_iter):
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

    return f1, kappa, report

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    train_init()
    data_path = './data/KFold_dataset'

    for fold in range(1,6):
        print('第{}折开始'.format(fold))
        get_save_path()
        train_iter = DataIterator4Classify(args, data_file=[data_path + f'/fold{fold}/train.csv'], is_test=False)
        valid_iter = DataIterator4Classify(args, data_file=[data_path + f'/fold{fold}/val.csv'], is_test=True)
        print(f'第{fold}折数据加载完成')
        logger = get_logger(args.model_save_path + "/log.log")
        model = CModel(args).to(args.device)

        bert_param_optimizer = list(model.pre_model.named_parameters())
        linear_param_optimizer = list(model.hidden2label.named_parameters())

        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            # pre-train model
            {'params': [p for n, p in bert_param_optimizer if not any(nd in n for nd in no_decay)],
             'weight_decay': 0.01,
             "lr": args.bert_learning_rate},
            {'params': [p for n, p in bert_param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0,
             "lr": args.bert_learning_rate},
            # linear layer
            {'params': [p for n, p in linear_param_optimizer if not any(nd in n for nd in no_decay)],
             'weight_decay': 0.01,
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
                                                    num_training_steps=len(train_iter) * (
                                                            args.epoch + args.extra_epoch))
        criterion = nn.CrossEntropyLoss()

        fgm = None
        if args.use_fgm:
            fgm = FGM(model)
        model.zero_grad()
        # ==================训练
        print(f'第{fold}折训练开始')
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
                    f"fold:{fold} epoch:{i + 1}/{args.epoch} lr:{optimizer.state_dict()['param_groups'][0]['lr']:.7f} classify_loss:{loss:.5f}")

            # 验证
            f1, kappa, report = valid(model, valid_iter)
            logger.info(f'dev set: epoch_{i + 1}, f1_{f1:.4f}, kappa_{kappa:.4f},\nreport:\n{report}')
            count += 1
            if kappa > best_kappa:
                best_kappa = kappa
                count = 0
                model_best = model.state_dict()
            if count >= 3:
                print('效果不增,终止训练')
                break
        # ====================预测
        print(f'第{fold}折测试开始')
        test_iter = DataIterator4Classify(args, data_file=args.test_data_path, is_test=True, is_predict=True)
        model = CModel(args).to(args.device)
        model.load_state_dict(model_best)
        model.eval()
        preds = []
        with torch.no_grad():
            for input_ids_list, input_mask_list, segment_ids_list, _ in tqdm(test_iter, position=0,
                                                                             dynamic_ncols=True, desc='测试中'):
                outputs = model.forward(input_ids_list, input_mask_list, segment_ids_list)
                pred = np.argmax(outputs.cpu().numpy(), axis=-1)
                preds.extend(pred)

        pd.DataFrame({'label': [i for i in preds]}).to_csv(data_path + f'/fold{fold}/result{fold}.csv', index=False,
                                                           header=None)





