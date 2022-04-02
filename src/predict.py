# -*- coding: utf-8 -*-
# @Time    : 2022/3/27 16:04
# @Author  : zxf
import os
import json

import torch
from transformers import AutoTokenizer
from transformers import BertTokenizer

from config import config
from common.common import logger
from utils.util import load_json
from utils.util import infer_data_feature
from models.MultiLabelModel import MultiLabelModel


def predict(sentence, label2id_file, model_file, threshold):
    device = "cpu"
    # load label2id
    label2id = load_json(label2id_file)
    id2label = {value: key for key, value in label2id.items()}
    logger.info("label size:{}".format(len(label2id)))
    # get tokenizer
    if config.model_type == "bert":
        tokenizer = BertTokenizer.from_pretrained(config.pretrain_model_path)
    else:
        tokenizer = AutoTokenizer.from_pretrained(config.pretrain_model_path)

    # get sentence feature
    input_ids, attention_mask, token_type_ids = infer_data_feature(sentence, tokenizer,
                                                                   config.max_length)
    # get tensor
    input_ids = torch.tensor([input_ids], dtype=torch.long).to(device)
    attention_mask = torch.tensor([attention_mask], dtype=torch.long).to(device)
    if token_type_ids is not None:
        token_type_ids = torch.tensor([token_type_ids], dtype=torch.long).to(device)

    # model
    model = MultiLabelModel(len(label2id), config.model_type, config.pretrain_model_path,
                            config.model_layer_type)
    model.load_state_dict(torch.load(model_file, map_location=torch.device(device)),
                          strict=True)
    model.to(device)

    logits, _ = model(input_ids, attention_mask, token_type_ids, None)
    pred = torch.sigmoid(logits).data.cpu().numpy()
    pred[pred >= threshold] = 1
    pred[pred < threshold] = 0
    pred = pred.tolist()[0]
    pred_label = []
    for index in range(len(pred)):
        if pred[index] == 1:
            pred_label.append(id2label[index])
    result = {"sent": sentence,
              "label": pred_label}
    return result


def predictv2(data_file, label2id_file, pretrain_model_path, model_file, threshold, output_file):
    device = "cpu"
    # load label2id
    label2id = load_json(label2id_file)
    id2label = {value: key for key, value in label2id.items()}
    logger.info("label size:{}".format(len(label2id)))
    # get tokenizer
    if config.model_type == "bert":
        tokenizer = BertTokenizer.from_pretrained(pretrain_model_path)
    else:
        tokenizer = AutoTokenizer.from_pretrained(pretrain_model_path)

    sentences = []
    labels = []
    with open(data_file, "r", encoding="utf-8") as f:
        for line in f.readlines():
            sent, label = line.strip().split('\t')
            sentences.append(sent)
            labels.append(label.split('|'))

    # model
    model = MultiLabelModel(len(label2id), config.model_type, pretrain_model_path,
                            config.model_layer_type)
    model.load_state_dict(torch.load(model_file, map_location=torch.device(device)),
                          strict=True)
    model.to(device)

    result = []
    for i in range(len(sentences)):
        sent = sentences[i]
        # get sentence feature
        input_ids, attention_mask, token_type_ids = infer_data_feature(sent, tokenizer,
                                                                       config.max_length)
        # get tensor
        input_ids = torch.tensor([input_ids], dtype=torch.long).to(device)
        attention_mask = torch.tensor([attention_mask], dtype=torch.long).to(device)
        if token_type_ids is not None:
            token_type_ids = torch.tensor([token_type_ids], dtype=torch.long).to(device)
        # logits
        logits, _ = model(input_ids, attention_mask, token_type_ids, None)
        pred = torch.sigmoid(logits).data.cpu().numpy()
        pred[pred >= threshold] = 1
        pred[pred < threshold] = 0
        pred = pred.tolist()[0]
        pred_label = []
        for index in range(len(pred)):
            if pred[index] == 1:
                pred_label.append(id2label[index])
        result.append({"text": sent,
                       "pred_label": pred_label,
                       "true_label": labels[i]})

        if i % 100 == 0:
            print("predict processing: {}/{}".format(i, len(sentences)))

    with open(output_file, "w", encoding="utf-8") as f:
        f.write(json.dumps(result, ensure_ascii=False, indent=2))


def check_pred_result(data_file, output_file):
    result = []
    with open(data_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    for item in data:
        true_label = item["true_label"]
        for pred in item["pred_label"]:
            if pred not in true_label:
                result.append({"text": item["text"],
                               "pred_label": item["pred_label"],
                               "true_label": item["true_label"]})
                break
    print("预测错误的数据的数量为:{}".format(len(result)))

    with open(output_file, "w", encoding="utf-8") as f:
        f.write(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    # sentence = "苹果2018下半年透明度报告出炉：共下架634款应用"
    # label2id_file = "./output/lable2id.json"
    # model_file = "./output/model.ckpt"
    # threshold = 0.5
    # result = predict(sentence, label2id_file, model_file, threshold)
    # print(result)

    data_file = "./data/event/dev_data.txt"
    label2id_file = "./output_event_loss/lable2id.json"
    pretrain_model_path = "D:/Spyder/pretrain_model/transformers_torch_tf/bert-base-chinese/"
    model_file = "./output_event_loss/model.ckpt"
    threshold = 0.5
    output_file = "./result/event_dev_predict_loss.json"
    predictv2(data_file, label2id_file, pretrain_model_path, model_file, threshold, output_file)

    data_file = "./result/event_dev_predict_loss.json"
    output_file = "./result/event_dev_check_result_loss.json"
    check_pred_result(data_file, output_file)
