### 多标签分类
baseline版本，这里模型计算loss的方式有两种：<br>
(1) 二值交叉熵
(2) 苏神公开的loss

#### 环境依赖
```
torch==1.8.1
numpy==1.19.2
transformers==4.5.1
scikit_learn==1.0.2
```

#### 代码结构
```
  src
   |__common
   |__config 配置文件
   |__data 数据
   |__log 日志文件
   |__models 模型文件
   |__output 输出目录
   |__utils 工具类
   |__main.py 模型训练 采用原始的参数配置文件
   |__predict.py 模型推理
   |__train.py 模型训练， 采用参数配置文件
```
#### 运行说明
1.修改配置文件中的参数<br>
模型评价方式有：f1，score， abs_acc，roc_auc
2.模型训练<br>
  python main.py  or python train.py <br>
3.模型推理<br>
  python predict.py 

#### 实验
1. 项目数据
```
采用绝对acc 评价：acc=0.34， 错误样本27
采用score， -36， 错误样本24
采用了min_error_num, 71, 38
```
2. event 事件抽取
```
验证集的数量为1498
1. 采用苏神的loss
评价指标f1： 0.97， 测试集预测错误的样本为89
```