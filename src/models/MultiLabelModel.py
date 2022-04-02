# -*- coding: utf-8 -*-
# @Time    : 2022/3/26 14:03
# @Author  : zxf
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel
from transformers import AutoModel


class MultiLabelModel(nn.Module):
    def __init__(self, num_labels, pretrain_model_type, pretrain_model_path,
                 model_layer_type):
        super(MultiLabelModel, self).__init__()
        self.num_labels = num_labels
        self.pretrain_model_type = pretrain_model_type
        self.pretrain_model_path = pretrain_model_path
        self.model_layer_type = model_layer_type
        if self.pretrain_model_type == "bert":
            self.encoder = BertModel.from_pretrained(self.pretrain_model_path)
            self.hidden_size = self.encoder.config.hidden_size
        else:
            self.encoder = AutoModel.from_pretrained(self.pretrain_model_path)
            self.hidden_size = self.encoder.config.hidden_size
        self.linear = nn.Linear(self.hidden_size, self.num_labels)
        
    def forward(self, input_ids, attention_mask, token_type_ids, label):
        outputs = self.encoder(input_ids, token_type_ids, attention_mask,
                               output_hidden_states=True)
        if self.model_layer_type == "pooler":
            output = outputs["pooler_output"]
        elif self.model_layer_type == "cls":
            output = outputs["last_hidden_state"][:, 0]
        elif self.model_layer_type == "mean":
            output = outputs["last_hidden_state"]
            attention_mask_extend = attention_mask.unsqueeze(-1).expand(output.size()).float()
            sum_output = torch.sum(output * attention_mask_extend, 1)
            sum_mask = attention_mask_extend.sum(1)
            sum_mask = torch.clamp(sum_mask, min=1e-9)
            output = sum_output / sum_mask
            # # 方式2
            # attention_mask = attention_mask.unsqueeze(-1)
            # output = torch.sum(output["last_hidden_state"] * attention_mask,
            # dim=1) / torch.sum(attention_mask, dim=1)
        elif self.model_layer_type == "first_last_mean":
            output = (outputs["hidden_states"][1] + outputs["hidden_states"][-1]).mean(dim=1)

        logits = self.linear(output)
        if label is not None:
            # loss = F.binary_cross_entropy_with_logits(logits, label)
            loss = multilabel_categorical_crossentropy(label, logits)
        else:
            loss = None
        return logits, loss


def multilabel_categorical_crossentropy(y_true, y_pred):
    """多标签分类的交叉熵
    说明：y_true和y_pred的shape一致，y_true的元素非0即1，
         1表示对应的类为目标类，0表示对应的类为非目标类。
    警告：请保证y_pred的值域是全体实数，换言之一般情况下y_pred
         不用加激活函数，尤其是不能加sigmoid或者softmax,预测
         阶段则输出y_pred大于0的类。
    """
    # y_pred = (1 - 2 * y_true) * y_pred
    # y_pred_neg = y_pred - y_true * 1e12
    # y_pred_pos = y_pred - (1 - y_true) * 1e12
    # zeros = K.zeros_like(y_pred[..., :1])
    # y_pred_neg = K.concatenate([y_pred_neg, zeros], axis=-1)
    # y_pred_pos = K.concatenate([y_pred_pos, zeros], axis=-1)
    # neg_loss = K.logsumexp(y_pred_neg, axis=-1)
    # pos_loss = K.logsumexp(y_pred_pos, axis=-1)
    y_true = y_true.view(1, -1)
    y_pred = y_pred.view(1, -1)
    y_pred = (1 - 2 * y_true) * y_pred
    y_pred_neg = y_pred - y_true * 1e12
    y_pred_pos = y_pred - (1 - y_true) * 1e12
    zeros = torch.zeros_like(y_pred[..., :1])
    y_pred_neg = torch.cat((y_pred_neg, zeros), dim=1)
    y_pred_pos = torch.cat((y_pred_pos, zeros), dim=1)
    neg_loss = torch.logsumexp(y_pred_neg, dim=1)
    pos_loss = torch.logsumexp(y_pred_pos, dim=1)
    # return neg_loss + pos_loss
    total_loss = (neg_loss + pos_loss)[0]
    return total_loss