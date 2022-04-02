# -*- coding: utf-8 -*-
# @Time    : 2022/4/2 14:52
# @Author  : zxf
import os
import json

import torch
import numpy as np
from torch.optim import Adam
from torch.optim import AdamW
from transformers import AutoTokenizer
from transformers import BertTokenizer
from torch.utils.data import DataLoader

from config import getConfig
from common.common import logger
from utils.util import collate_fn
from utils.util import model_evaluate
from utils.util import GeneratorModleData
from utils.util import get_best_score_init
from models.MultiLabelModel import MultiLabelModel


def main(config_file):
    Config = getConfig.get_config(config_file)
    os.environ["CUDA_VISIBLE_DEVICES"] = Config['gpu_ids']
    device = "cpu" if Config['gpu_ids'] == "-1" else "cuda"
    # check path
    if not os.path.exists(Config['output_path']):
        os.mkdir(Config['output_path'])
    # tokenizer
    if Config['model_type'] == "bert":
        tokenizer = BertTokenizer.from_pretrained(Config['pretrain_model_path'])
    else:
        tokenizer = AutoTokenizer.from_pretrained(Config['pretrain_model_path'])

    # read data
    data_gene = GeneratorModleData(tokenizer, Config['max_length'])
    # train data
    train_data, train_label, label2id = data_gene.read_data(Config['train_data_file'])
    logger.info("train data size: {}".format(len(train_data)))
    # save label2id
    with open(os.path.join(Config['output_path'],
                           Config['label2id_file']), "w", encoding="utf-8") as f:
        f.write(json.dumps(label2id, ensure_ascii=False, indent=2))
    dev_data, dev_label, _ = data_gene.read_data(Config['test_data_file'])
    logger.info("test data size: {}".format(len(dev_data)))

    # labele encoder
    train_label = data_gene.label_encoder(train_label, label2id)
    dev_label = data_gene.label_encoder(dev_label, label2id)
    # get data features
    train_feature = data_gene.sentence_encoder(train_data, train_label)
    dev_feature = data_gene.sentence_encoder(dev_data, dev_label)
    # data loader
    train_dataloader = DataLoader(train_feature, shuffle=False,
                                  batch_size=Config['batch_size'],
                                  collate_fn=collate_fn)
    dev_dataloader = DataLoader(dev_feature, shuffle=False,
                                batch_size=Config['dev_batch_size'],
                                collate_fn=collate_fn)
    logger.info("pre epoch batch number:{}".format(len(train_dataloader)))

    # model
    model = MultiLabelModel(len(label2id), Config['model_type'], Config['pretrain_model_path'],
                            Config['model_layer_type'])
    model.to(device)
    # optimizer
    optimizer = AdamW(model.parameters(), lr=Config['lr'])
    # get best score init value
    best_score = get_best_score_init(Config['evaluate_type'])
    logger.info("best_score: {}".format(best_score))
    logger.info("model training")

    for epoch in range(Config['epochs']):
        model.train()
        for step, batch_data in enumerate(train_dataloader):
            input_ids = batch_data["input_ids"].to(device)
            if "token_type_ids" is not None:
                token_type_ids = batch_data["token_type_ids"].to(device)
            attention_mask = batch_data["attention_mask"].to(device)
            label = batch_data["label"].to(device)
            _, loss = model(input_ids, attention_mask, token_type_ids, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (step + 1) % Config['pre_epoch_print_step'] == 0:
                logger.info("epoch:{}/{} step:{}/{} loss:{}".format(epoch + 1, Config['epochs'],
                                                                    step + 1, len(train_dataloader),
                                                                    loss))
        score = model_evaluate(model, dev_dataloader, device, Config['threshold'],
                               Config['evaluate_type'], logger)
        # score = evaluate(model, dev_dataloader, device, Config['threshold)
        if Config['evaluate_type'] in ["min_error_num"]:
            if score <= best_score:
                best_score = score
                torch.save(model.state_dict(), os.path.join(Config['output_path'],
                                                            Config['save_model_name']))
        else:
            if score >= best_score:
                best_score = score
                torch.save(model.state_dict(), os.path.join(Config['output_path'],
                                                            Config['save_model_name']))
        logger.info("best score:{}".format(best_score))


if __name__ == "__main__":
    config_file = "./config/config.ini"
    main(config_file)