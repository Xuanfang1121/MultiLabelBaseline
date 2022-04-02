# -*- coding: utf-8 -*-
# @Time    : 2022/3/26 15:33
# @Author  : zxf
import os
import json

import torch
import numpy as np
import torch.nn.functional as F
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_score


class GeneratorModleData(object):
    def __init__(self, tokenizer, max_length):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.pad_token_ids = self.tokenizer.pad_token_id

    def read_data(self, data_file):
        data = []
        labels = []
        label2id = {}
        with open(data_file, "r", encoding="utf-8") as f:
            for line in f.readlines():
                sent, label = line.strip().split('\t')
                data.append(sent)
                label = label.split('|')
                labels.append(label)
                for item in label:
                    if item not in label2id:
                        label2id[item] = len(label2id)
        return data, labels, label2id

    def label_encoder(self, labels, label2id):
        """
           多标签label编码，编码为[[0,0,0,1,1,1]]
        """
        result = []
        for label in labels:
            temp = [0] * len(label2id)
            for item in label:
                index = label2id[item]
                temp[index] = 1
            result.append(temp)
        return result

    def sentence_encoder(self, data, label):
        """
        Args:
            data: [sent1, sent2]

        Returns:

        """
        features = []
        for i in range(len(data)):
            tokens = self.tokenizer(data[i], max_length=self.max_length,
                                    truncation=True, add_special_tokens=True)
            if "token_type_ids" not in tokens:
                token_type_ids = None
            input_ids = tokens["input_ids"] + [self.pad_token_ids] * (self.max_length -
                                                                      len(tokens["input_ids"]))
            attention_mask = tokens["attention_mask"] + [0] * (self.max_length -
                                                               len(tokens["attention_mask"]))
            if "token_type_ids" in tokens:
                token_type_ids = tokens["token_type_ids"] + [0] * (self.max_length -
                                                                   len(tokens["token_type_ids"]))
            assert len(input_ids) == self.max_length, f'input ids size:{len(input_ids)}' \
                                                      f'max length:{len(self.max_length)}'
            features.append({"input_ids": input_ids,
                             "attention_mask": attention_mask,
                             "token_type_ids": token_type_ids,
                             "label": label[i]})
        return features


def collate_fn(batch_data):
    input_ids = []
    attention_mask = []
    token_type_ids = []
    label = []
    for item in batch_data:
        input_ids.append(item["input_ids"])
        attention_mask.append(item["attention_mask"])
        if "token_type_ids" in item:
            token_type_ids.append(item["token_type_ids"])
        label.append(item["label"])
    if token_type_ids:
        token_type_ids_tensor = torch.tensor(token_type_ids, dtype=torch.long)
    else:
        token_type_ids_tensor = None
    input_ids_tensor = torch.tensor(input_ids, dtype=torch.long)
    attention_mask_tensor = torch.tensor(attention_mask, dtype=torch.long)
    labels_tensor = torch.tensor(label, dtype=torch.float)
    return {"input_ids": input_ids_tensor,
            "token_type_ids": token_type_ids_tensor,
            "attention_mask": attention_mask_tensor,
            "label": labels_tensor}


def model_evaluate(model, dataloader, device, threshold, evaluate_type, logger):
    model.eval()
    pred_label = []
    true_label = []
    with torch.no_grad():
        for step, batch_data in enumerate(dataloader):
            input_ids = batch_data["input_ids"].to(device)
            attention_mask = batch_data["attention_mask"].to(device)
            if batch_data["token_type_ids"] is not None:
                token_type_ids = batch_data["token_type_ids"].to(device)
            else:
                token_type_ids = None
            label = batch_data["label"].tolist()
            true_label.extend(label)
            logits, _ = model(input_ids, attention_mask, token_type_ids, None)
            pred = F.sigmoid(logits).data.cpu().numpy()
            pred[pred >= threshold] = 1.0
            pred[pred < threshold] = 0.0
            pred_label.extend(pred.tolist())

    if evaluate_type == "roc_auc":
        score = model_roc_auc_score(np.array(true_label), np.array(pred_label))
    elif evaluate_type == "abs_acc":
        score = absolute_accuracy(true_label, pred_label)
    elif evaluate_type == "score":
        score = 0
        for true_label, pred_label in zip(true_label, pred_label):
            score += evaluate_score(true_label, pred_label)
    elif evaluate_type == "min_error_num":
         score = predict_label_acc_number(true_label, pred_label)
    elif evaluate_type == "f1":
        precision, recall, score = precison_recall_f1_score_mean(true_label, pred_label)
        logger.info("precision:{} recall:{} f1:{}".format(precision, recall, score))
    return score


def evaluate_score(reference, prediction):
    score = 0
    for i in range(len(reference)):
        if reference[i] == float(1) and prediction[i] == float(1):
            score += 1
        elif reference[i] == float(0) and prediction[i] == float(1):
            score -= 1
        elif reference[i] == float(1) and prediction[i] == float(0):
            score -= 1
    return score


def evaluate(model, data_loader, device, threshold):
    """
        模型评价
    """
    model.eval()
    true_label = []
    pred_label = []
    with torch.no_grad():
        for step, batch_data in enumerate(data_loader):
            input_ids = batch_data["input_ids"].to(device)
            attention_mask = batch_data["attention_mask"].to(device)
            if batch_data["token_type_ids"] is not None:
                token_type_ids = batch_data["token_type_ids"].to(device)
            else:
                token_type_ids = None
            label = batch_data["label"].tolist()
            true_label.extend(label)
            logits, _ = model(input_ids, attention_mask, token_type_ids, None)
            pred = F.sigmoid(logits).data.cpu().numpy()
            pred[pred >= threshold] = 1.0
            pred[pred < threshold] = 0.0
            pred_label.extend(pred.tolist())
    scores = 0
    for true_label, pred_label in zip(true_label, pred_label):
        scores += evaluate_score(true_label, pred_label)
    return scores


def predict_label_acc_number(y_true, y_pred):
    """
    Args:
        y_true: list(list(1, 1, 1, 0, 0, 0, 0), (0, 0, 1, 1, 1, 0, 0))
        y_pred: list(list(1, 1, 0, 0, 0, 0, 0), (0, 0, 0, 1, 1, 0, 0))

    Returns: int
    统计 预测错误的数量， 值越小，模型越好
    """
    num = 0
    assert len(y_true) == len(y_pred), f'true label size:{len(y_true)} pred label size:{y_pred}'
    for i in range(len(y_true)):
        for j in range(len(y_true[0])):
            if y_true[i][j] == 1 and y_true[i][j] != y_pred[i][j]:
                num += 1
    return num


def absolute_accuracy(y_true, y_pred):
    """
    Args:
        y_true: list(list(1, 1, 1, 0, 0, 0, 0), (0, 0, 1, 1, 1, 0, 0))
        y_pred: list(list(1, 1, 0, 0, 0, 0, 0), (0, 0, 0, 1, 1, 0, 0))

    Returns:
    绝对准确率: 真实标签和预测标签相等，表示预测正确,
    """
    acc_num = 0
    assert len(y_true) == len(y_pred), f'true label size:{len(y_true)} pred label size:{y_pred}'
    for i in range(len(y_true)):
        if y_true[i] == y_pred[i]:
            acc_num += 1
    acc = acc_num / len(y_true)
    return acc


def relative_accuracy(y_true, y_pred):
    """
    Args:
        y_true: list(list(1, 1, 1, 0, 0, 0, 0), (0, 0, 1, 1, 1, 0, 0))
        y_pred: list(list(1, 1, 0, 0, 0, 0, 0), (0, 0, 0, 1, 1, 0, 0))

    Returns: acc:float
    相对准确率：预测正确一个标签记为预测正确
    """
    acc_num = 0
    assert len(y_true) == len(y_pred), f'true label size:{len(y_true)} pred label size:{y_pred}'
    for i in range(len(y_true)):
        for j in range(len(y_true[i])):
            if y_true[i][j] == y_pred[i][j] and y_true[i][j] == 1:
                acc_num += 1
                break
    acc = acc_num / len(y_true)
    return acc


def model_roc_auc_score(y_true, y_pred):
    """
        计算模型的roc_auc score
    """
    try:
        roc_auc = roc_auc_score(np.array(y_true), np.array(y_pred), average=None)
        score = np.mean(roc_auc)

    except ValueError:
        score = 0.0
        pass
    return score


def precison_recall_f1_score_mean(y_true, y_pred):
    """
    Args:
        y_true: list(list(1, 1, 1, 0, 0, 0, 0), (0, 0, 1, 1, 1, 0, 0))
        y_pred: list(list(1, 1, 0, 0, 0, 0, 0), (0, 0, 0, 1, 1, 0, 0))

    Returns:
    计算每一个标签的recall， precision, f1 score,最后取mean, 这里计算多标签模型的平均
    precision，recall，f1 score， 如果有一个多标签样本不均衡，模型的评价指标不会很高
    """
    recall_result = []
    f1_result = []
    precision_result = []
    assert len(y_true) == len(y_pred), f'true label size:{len(y_true)} ' \
                                       f'pred label size:{y_pred}'
    for j in range(len(y_true[0])):
        temp_true = []
        temp_pred = []
        for i in range(len(y_true)):
            temp_true.append(y_true[i][j])
            temp_pred.append(y_pred[i][j])
        assert len(temp_true) == len(temp_pred), f'temp true size:{len(temp_true)},' \
                                                 f'temp pred size:{len(temp_pred)}'
        temp_precision = precision_score(temp_true, temp_pred, average='macro', zero_division=0)
        temp_recall = recall_score(temp_true, temp_pred, average='macro', zero_division=0)
        temp_f1 = f1_score(temp_true, temp_pred, average='macro', zero_division=0)
        precision_result.append(temp_precision)
        recall_result.append(temp_recall)
        f1_result.append(temp_f1)

    precision = np.mean(precision_result)
    recall = np.mean(recall_result)
    f1 = np.mean(f1_result)
    return precision, recall, f1


def get_best_score_init(evaluate_type):
    """
    Args:
        evaluate_type: string

    Returns:best score
    根据evaluate_type初始话best score
    """
    if evaluate_type in ["abs_acc", "f1", "roc_auc"]:
        best_score = 0.0
    elif evaluate_type in ["min_error_num"]:
        best_score = np.inf
    elif evaluate_type in ["score"]:
        best_score = -np.inf
    return best_score


def infer_data_feature(sentence, tokenizer, max_length):
    """
       模型推理的数据处理
    """
    tokens = tokenizer(sentence, max_length=max_length,
                       truncation=True, add_special_tokens=True)
    if "token_type_ids" not in tokens:
        token_type_ids = None
    input_ids = tokens["input_ids"] + [tokenizer.pad_token_id] * (max_length -
                                                                  len(tokens["input_ids"]))
    attention_mask = tokens["attention_mask"] + [0] * (max_length -
                                                       len(tokens["attention_mask"]))
    if "token_type_ids" in tokens:
        token_type_ids = tokens["token_type_ids"] + [0] * (max_length -
                                                           len(tokens["token_type_ids"]))
    assert len(input_ids) == max_length, f'input ids size:{len(input_ids)}' \
                                         f'max length:{len(max_length)}'
    return input_ids, attention_mask, token_type_ids


def load_json(data_file):
    """
    Args:
        data_file: string

    Returns: dict
    读取json文件
    """
    with open(data_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data