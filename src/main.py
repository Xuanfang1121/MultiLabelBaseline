# -*- coding: utf-8 -*-
# @Time    : 2022/3/26 18:17
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

from config import config
from utils.util import evaluate
from common.common import logger
from utils.util import collate_fn
from utils.util import model_evaluate
from utils.util import GeneratorModleData
from utils.util import get_best_score_init
from models.MultiLabelModel import MultiLabelModel


def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu_ids
    device = "cpu" if config.gpu_ids == "-1" else "cuda"
    # check path
    if not os.path.exists(config.output_path):
        os.mkdir(config.output_path)
    # tokenizer
    if config.model_type == "bert":
        tokenizer = BertTokenizer.from_pretrained(config.pretrain_model_path)
    else:
        tokenizer = AutoTokenizer.from_pretrained(config.pretrain_model_path)

    # read data
    data_gene = GeneratorModleData(tokenizer, config.max_length)
    # train data
    train_data, train_label, label2id = data_gene.read_data(config.train_data_file)
    logger.info("train data size: {}".format(len(train_data)))
    # save label2id
    with open(os.path.join(config.output_path,
                           config.label2id_file), "w", encoding="utf-8") as f:
        f.write(json.dumps(label2id, ensure_ascii=False, indent=2))
    dev_data, dev_label, _ = data_gene.read_data(config.test_data_file)
    logger.info("test data size: {}".format(len(dev_data)))

    # labele encoder
    train_label = data_gene.label_encoder(train_label, label2id)
    dev_label = data_gene.label_encoder(dev_label, label2id)
    # get data features
    train_feature = data_gene.sentence_encoder(train_data, train_label)
    dev_feature = data_gene.sentence_encoder(dev_data, dev_label)
    # data loader
    train_dataloader = DataLoader(train_feature, shuffle=False,
                                  batch_size=config.batch_size,
                                  collate_fn=collate_fn)
    dev_dataloader = DataLoader(dev_feature, shuffle=False,
                                batch_size=config.dev_batch_size,
                                collate_fn=collate_fn)
    logger.info("pre epoch batch number:{}".format(len(train_dataloader)))

    # model
    model = MultiLabelModel(len(label2id), config.model_type, config.pretrain_model_path,
                            config.model_layer_type)
    model.to(device)
    # optimizer
    optimizer = AdamW(model.parameters(), lr=config.lr)
    # get best score init value
    best_score = get_best_score_init(config.evaluate_type)
    logger.info("best_score: {}".format(best_score))
    logger.info("model training")

    for epoch in range(config.epochs):
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

            if (step + 1) % config.pre_epoch_print_step == 0:
                logger.info("epoch:{}/{} step:{}/{} loss:{}".format(epoch + 1, config.epochs,
                                                                    step + 1, len(train_dataloader),
                                                                    loss))
        score = model_evaluate(model, dev_dataloader, device, config.threshold,
                               config.evaluate_type, logger)
        # score = evaluate(model, dev_dataloader, device, config.threshold)
        if config.evaluate_type in ["min_error_num"]:
            if score <= best_score:
                best_score = score
                torch.save(model.state_dict(), os.path.join(config.output_path,
                                                            config.save_model_name))
        else:
            if score >= best_score:
                best_score = score
                torch.save(model.state_dict(), os.path.join(config.output_path,
                                                            config.save_model_name))
        logger.info("best score:{}".format(best_score))


if __name__ == "__main__":
    main()