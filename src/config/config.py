# -*- coding: utf-8 -*-
# @Time    : 2022/3/9 11:08
# @Author  : zxf


# data_path
train_data_file = "./data/event/train_data.txt"
test_data_file = "./data/event/dev_data.txt"
# train_data_file = "./data/ITT/train_data.txt"
# test_data_file = "./data/ITT/test_data.txt"

#  pretrain model path
# pretrain_model_path = "D:/Spyder/pretrain_model/transformers_torch_tf/bert-base-chinese/"
pretrain_model_path = "/opt/nlp/pretrain_model/bert-base-chinese/"
# pretrain_model_path = "/opt/nlp/pretrain_model/simbert-base-chinese"

# model paras
threshold = 0.5
gpu_ids = "2"
seed = 1
batch_size = 32
dev_batch_size = 16
lr = 1e-5
epochs = 100
max_length = 256
model_type = "bert"
# select pooler, cls, mean, first_last_mean
model_layer_type = "mean"  # "cls"
# model evaluate methods, "abs_acc", "roc_auc", "score", "min_error_num", "f1"
evaluate_type = "f1"
pre_epoch_print_step = 100
improvement_epoch = 10
# save para
output_path = "./output/"
label2id_file = "lable2id.json"
save_model_name = "model.ckpt"
