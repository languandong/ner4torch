import torch
import torch.nn as nn
from total_utils.pretrain_model_utils.NEZHA.modeling_nezha import NeZhaModel
from total_utils.pretrain_model_utils.NEZHA.configuration_nezha import NeZhaConfig
from transformers import BertModel


class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        if args.pre_model_type == 'NEZHA':
            nezha_config = NeZhaConfig.from_json_file(args.pretrain_model_path + "/bert_config.json")
            nezha_config.output_hidden_states = True
            # nezha_config.max_position_embeddings = args.sequence_length
            # nezha_config.max_relative_position = int(args.sequence_length / 4)
            self.pre_model = NeZhaModel(nezha_config)
        elif args.pre_model_type == 'bert':
            self.pre_model = BertModel.from_pretrained(args.pretrain_model_path)
        else:
            raise ValueError('Pre-train Model type must be NEZHA or bert!')

        # args参数传递
        self.pre_model_type = args.pre_model_type
        self.use_dynamic_fusion = args.use_dynamic_fusion

        if self.use_dynamic_fusion:
            self.classifier = nn.Linear(args.bert_hidden_size, 1)

        # 模型层
        self.dropout = torch.nn.Dropout(args.dropout_rate)
        self.hidden2label = nn.Linear(args.bert_hidden_size, args.relation_num)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, input_ids_list, input_mask_list, segment_ids_list, label_ids_list=None):
        if self.pre_model_type == 'NEZHA':
            nezha_output = self.pre_model(
                input_ids=input_ids_list,
                attention_mask=input_mask_list,
                token_type_ids=segment_ids_list
            )
            sequence_out = nezha_output[0]
            pooled_output = nezha_output[1]
            encoded_layers = nezha_output[2]

        else:
            sequence_out, ori_pooled_output, encoded_layers = self.pre_model(input_ids=input_ids_list,
                                                                             attention_mask=input_mask_list,
                                                                             token_type_ids=segment_ids_list,
                                                                             return_dict=False,
                                                                             output_hidden_states=True)

        sequence_out = self.dropout(sequence_out)
        if self.use_dynamic_fusion:
            sequence_out = self.get_dym_layer(encoded_layers)
            sequence_out = self.dropout(sequence_out)

        logits = self.hidden2label(sequence_out)

        if label_ids_list is not None:
            logits = logits.view(-1, logits.shape[-1])
            label_ids_list = label_ids_list.view(-1)
            loss = self.criterion(logits, label_ids_list)
            return loss
        else:
            predict = logits.argmax(-1)
            return predict

    def get_dym_layer(self, outputs):
        #bert中文base版总共有12层，也就是每一层都可以输出相应的特征，
        # 我们可以使用model.all_encoder_layers来获取，然后我们将每一层的768维度的特征映射成1维，
        # 对每一个特征进行最后一个维度的拼接后经过softmax层，得到每一层特征相对应的权重，
        # 最后经过[batchsize,max_len,1,12] × [batchsize,max_len,12,768]，得到[batchszie,max_len,1,768]，
        # 去除掉一维得到[batchsize,max_len,768]，这样我们就得到了可以动态选择的特征，接下来就可以利用该特征进行相关的微调任务了。
        layer_logits = []
        all_encoder_layers = outputs[1:]
        for i, layer in enumerate(all_encoder_layers):
            layer_logits.append(self.classifier(layer))
        layer_logits = torch.cat(layer_logits, 2)
        # layer_dist = torch.nn.functional.softmax(layer_logits)
        layer_dist = torch.softmax(layer_logits, dim=-1)
        # with open(args.processed_data + 'dynamic_weight_record.txt', 'w') as fw:
        #     t = layer_dist
        #     fw.write(str(t.argmax(-1)) + '\n')
        seq_out = torch.cat([torch.unsqueeze(x, 2) for x in all_encoder_layers], dim=2)
        pooled_output = torch.matmul(torch.unsqueeze(layer_dist, 2), seq_out)
        pooled_output = torch.squeeze(pooled_output, 2)

        return pooled_output
