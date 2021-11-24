import torch
import torch.nn as nn
from total_utils.pretrain_model_utils.NEZHA.modeling_nezha import NeZhaModel
from total_utils.pretrain_model_utils.NEZHA.configuration_nezha import NeZhaConfig
from transformers import BertModel
from torchcrf import CRF
import torch.nn.functional as F
from transformers.models.bert import BertConfig
from args import pretrain_model_path


class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        if args.pre_model_type.split('_')[0] == 'nezha':
            nezha_config = NeZhaConfig.from_json_file(pretrain_model_path[args.pre_model_type] + "config.json")
            nezha_config.output_hidden_states = True
            nezha_config.max_position_embeddings = 1024
            self.pre_model = NeZhaModel.from_pretrained(pretrain_model_path[args.pre_model_type], config=nezha_config)

        elif args.pre_model_type.split('_')[0] == 'ernie':
            bert_config = BertConfig.from_json_file(pretrain_model_path[args.pre_model_type] + "config.json")
            bert_config.output_hidden_states = True
            self.pre_model = BertModel.from_pretrained(pretrain_model_path[args.pre_model_type], config=bert_config)
        else:
            bert_config = BertConfig.from_json_file(pretrain_model_path[args.pre_model_type] + "config.json")
            bert_config.output_hidden_states = True
            self.pre_model = BertModel.from_pretrained(pretrain_model_path[args.pre_model_type], config=bert_config)

        self.label_type = args.special_label + [j + "-" + i for i in args.entity_label for j in
                                                args.bio_label]
        self.label2id = {value: index for index, value in enumerate(self.label_type)}

        # args参数传递
        self.pre_model_type = args.pre_model_type
        self.use_dynamic_fusion = args.use_dynamic_fusion
        self.use_avgpool = args.use_avgpool
        self.batch_size = args.batch_size
        self.device = args.device

        if self.use_dynamic_fusion:
            self.classifier = nn.Linear(args.bert_hidden_size, 1)

        # 模型层
        self.dropout = torch.nn.Dropout(args.dropout_rate)
        if self.use_avgpool:
            self.hidden2label = nn.Linear(args.bert_hidden_size + 1 - 16, args.relation_num)
        else:
            self.hidden2label = nn.Linear(args.bert_hidden_size, args.relation_num)
        self.crf_layer = CRF(num_tags=args.relation_num)

    def forward(self, input_ids_list, input_mask_list, segment_ids_list, label_ids_list=None):
        if self.pre_model_type.split('_')[0] == 'nezha':
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

        if self.use_dynamic_fusion:
            sequence_out = self.get_dym_layer(encoded_layers)
            sequence_out = self.dropout(sequence_out)
        if self.use_avgpool:
            sequence_out = F.avg_pool1d(sequence_out, kernel_size=16, stride=1)
            sequence_out = self.dropout(sequence_out)
        logits = self.hidden2label(sequence_out) # batch_size,max_len,label

        if self.training:
            loss = self.crf_layer.forward(logits.transpose(1, 0), label_ids_list.transpose(1, 0),
                                          mask=input_mask_list.transpose(1, 0))
            return -loss
        else:
            predict = self.crf_layer.decode(logits.transpose(1, 0), input_mask_list.transpose(1, 0))
            return predict

    def get_dym_layer(self, all_encoder_layers):
        layer_logits = []
        all_encoder_layers = all_encoder_layers[1:]
        for i, layer in enumerate(all_encoder_layers):
            layer_logits.append(self.classifier(layer))
        layer_logits = torch.cat(layer_logits, 2)
        layer_dist = torch.softmax(layer_logits, dim=-1)
        # with open(args().processed_data + 'dynamic_weight_record.txt', 'w') as fw:
        #     t = layer_dist
        #     fw.write(str(t.argmax(-1)) + '\n')
        seq_out = torch.cat([torch.unsqueeze(x, 2) for x in all_encoder_layers], dim=2)
        pooled_output = torch.matmul(torch.unsqueeze(layer_dist, 2), seq_out)
        pooled_output = torch.squeeze(pooled_output, 2)

        return pooled_output
