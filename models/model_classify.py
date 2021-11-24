import torch
import torch.nn as nn
from total_utils.pretrain_model_utils.NEZHA.modeling_nezha import NeZhaModel
from total_utils.pretrain_model_utils.NEZHA.configuration_nezha import NeZhaConfig
from transformers import BertModel
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
            bert_config = BertConfig.from_pretrained(pretrain_model_path[args.pre_model_type] + "config.json")
            bert_config.output_hidden_states = True
            self.pre_model = BertModel.from_pretrained(pretrain_model_path[args.pre_model_type], config=bert_config)

        else:
            bert_config = BertConfig.from_json_file(pretrain_model_path[args.pre_model_type] + "config.json")
            bert_config.output_hidden_states = True
            self.pre_model = BertModel.from_pretrained(pretrain_model_path[args.pre_model_type], config=bert_config)
        print(f'预训练模型{args.pre_model_type}')
        print(f'下接结构{args.struc}')
        # args参数传递
        self.args = args
        self.pre_model_type = args.pre_model_type
        self.use_dynamic_fusion = args.use_dynamic_fusion
        self.use_avgpool = args.use_avgpool
        self.batch_size = args.batch_size

        if self.use_dynamic_fusion:
            self.classifier = nn.Linear(args.bert_hidden_size, 1)

        # 模型层
        self.dropouts = nn.ModuleList([nn.Dropout(0.2) for _ in range(args.dropout_num)])
        self.dropout = nn.Dropout(args.dropout_rate)
        if self.use_avgpool:
            self.hidden2label = nn.Linear(args.bert_hidden_size + 1 - 16, args.class_num)
        else:
            if args.struc == 'cls':
                self.hidden2label = nn.Linear(args.bert_hidden_size, args.class_num)
            elif args.struc == 'bilstm':
                self.bilstm = nn.LSTM(args.bert_hidden_size, args.lstm_dim, bidirectional=True, num_layers=1,
                                      batch_first=True)
                self.hidden2label = nn.Linear(args.lstm_dim * 2, args.class_num)
            elif args.struc == 'bigru':
                self.bigru = nn.GRU(args.bert_hidden_size, args.gru_dim, bidirectional=True, num_layers=1,
                                    batch_first=True)
                self.hidden2label = nn.Linear(args.gru_dim * 2, args.class_num)


    def forward(self, input_ids_list, input_mask_list, segment_ids_list):
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

        if self.args.struc == 'cls':
            sequence_out = sequence_out[:, 0, :]  # cls
        else:
            if self.args.struc == 'bilstm':
                _, hidden = self.bilstm(sequence_out)
                last_hidden = hidden[0].permute(1, 0, 2)
                sequence_out = last_hidden.contiguous().view(-1, self.args.lstm_dim * 2)
            elif self.args.struc == 'bigru':
                _, hidden = self.bigru(sequence_out)
                last_hidden = hidden.permute(1, 0, 2)
                sequence_out = last_hidden.contiguous().view(-1, self.args.gru_dim * 2)

        if self.args.dropout_num == 0:
            logits = self.hidden2label(sequence_out)
        elif self.args.dropout_num == 1:
            sequence_out = self.dropouts[0](sequence_out)
            logits = self.hidden2label(sequence_out)
        else:
            out = None
            for i, dropout in enumerate(self.dropouts):
                if i == 0:
                    out = dropout(sequence_out)
                    out = self.hidden2label(out)
                else:
                    temp_out = dropout(sequence_out)
                    out = out + self.hidden2label(temp_out)
            logits = out / len(self.dropouts)



        return logits

    def get_dym_layer(self, all_encoder_layers):
        layer_logits = []
        all_encoder_layers = all_encoder_layers[1:]
        for i, layer in enumerate(all_encoder_layers):
            layer_logits.append(self.classifier(layer))
        layer_logits = torch.cat(layer_logits, 2)
        layer_dist = torch.softmax(layer_logits, dim=-1)
        seq_out = torch.cat([torch.unsqueeze(x, 2) for x in all_encoder_layers], dim=2)
        pooled_output = torch.matmul(torch.unsqueeze(layer_dist, 2), seq_out)
        pooled_output = torch.squeeze(pooled_output, 2)

        return pooled_output

class Model_outputFake(nn.Module):
    def __init__(self, args):
        super(Model_outputFake, self).__init__()
        if args.pre_model_type.split('_')[0] == 'nezha':
            nezha_config = NeZhaConfig.from_json_file(pretrain_model_path[args.pre_model_type] + "config.json")
            nezha_config.output_hidden_states = True
            nezha_config.max_position_embeddings = 1024
            self.pre_model = NeZhaModel.from_pretrained(pretrain_model_path[args.pre_model_type], config=nezha_config)
        elif args.pre_model_type.split('_')[0] == 'ernie':
            bert_config = BertConfig.from_pretrained(pretrain_model_path[args.pre_model_type] + "config.json")
            bert_config.output_hidden_states = True
            self.pre_model = BertModel.from_pretrained(pretrain_model_path[args.pre_model_type], config=bert_config)

        else:
            bert_config = BertConfig.from_json_file(pretrain_model_path[args.pre_model_type] + "config.json")
            bert_config.output_hidden_states = True
            self.pre_model = BertModel.from_pretrained(pretrain_model_path[args.pre_model_type], config=bert_config)
        print(f'预训练模型{args.pre_model_type}')
        print(f'下接结构{args.struc}')
        # args参数传递
        self.args = args
        self.pre_model_type = args.pre_model_type
        self.use_dynamic_fusion = args.use_dynamic_fusion
        self.use_avgpool = args.use_avgpool
        self.batch_size = args.batch_size

        if self.use_dynamic_fusion:
            self.classifier = nn.Linear(args.bert_hidden_size, 1)

        # 模型层
        self.dropout = torch.nn.Dropout(args.dropout_rate)
        if self.use_avgpool:
            self.hidden2label = nn.Linear(args.bert_hidden_size + 1 - 16, args.class_num)
        else:
            if args.struc == 'cls':
                self.hidden2label = nn.Linear(args.bert_hidden_size, args.class_num)
            elif args.struc == 'bilstm':
                self.bilstm = nn.LSTM(args.bert_hidden_size, args.lstm_dim, bidirectional=True, num_layers=1,
                                      batch_first=True)
                self.hidden2label = nn.Linear(args.lstm_dim * 2, args.class_num)
            elif args.struc == 'bigru':
                self.bigru = nn.GRU(args.bert_hidden_size, args.gru_dim, bidirectional=True, num_layers=1,
                                    batch_first=True)
                self.hidden2label = nn.Linear(args.gru_dim * 2, args.class_num)

        self.softmax = nn.Softmax(dim=1)

    def forward(self, input_ids_list, input_mask_list, segment_ids_list):
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

        if self.args.struc == 'cls':
            sequence_out = sequence_out[:, 0, :]  # cls
        else:
            if self.args.struc == 'bilstm':
                _, hidden = self.bilstm(sequence_out)
                last_hidden = hidden[0].permute(1, 0, 2)
                sequence_out = last_hidden.contiguous().view(-1, self.args.lstm_dim * 2)
            elif self.args.struc == 'bigru':
                _, hidden = self.bigru(sequence_out)
                last_hidden = hidden.permute(1, 0, 2)
                sequence_out = last_hidden.contiguous().view(-1, self.args.gru_dim * 2)

        logits = self.hidden2label(sequence_out)
        logits = self.softmax(logits)

        return logits