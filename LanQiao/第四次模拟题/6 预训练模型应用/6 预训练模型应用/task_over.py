import json
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pandas as pd

# 从json文件中提取data
with open('resnet_train.json', 'r') as f:
    data_resnet = json.load(f)
# print(data_resnet)
with open('inception_train.json', 'r') as f:
    data_inception = json.load(f)
# print(data_inception)
with open('xception_train.json', 'r') as f:
    data_xception = json.load(f)

with open('test.json', 'r') as f:
    data_test = json.load(f)
# print(data_test)


# 按顺序提取特征向量（长度6144）并转换为numpy数组
features_resnet = np.array([data_resnet[id]['feature'] for id in data_resnet])
print("features_resnet.shape:")
print(features_resnet.shape)
features_inception = np.array([data_inception[id]['feature'] for id in data_inception])
print("features_inception.shape:")
print(features_inception.shape)
features_xception = np.array([data_xception[id]['feature'] for id in data_xception])
print("features_xception.shape:")
print(features_xception.shape)
# 按照resnet、inception、xception的顺序合并特征向量
train_features = np.concatenate([features_resnet, features_inception, features_xception], axis=1)
train_labels = np.array([data_resnet[id]['label'] for id in data_resnet])
print("train_features.shape:", train_features.shape)
print("train_labels.shape:", train_labels.shape)

test_features = np.array([data_test[id]['feature'] for id in data_test])
print(test_features.shape)
# 训练模型,这里以随机树森林为例
model = RandomForestClassifier()
model.fit(train_features, train_labels)

# 测试集数据预测
test_predictions = model.predict(test_features)
# 保存预测结果
# 提取所有测试样本的ID
test_ids = list(data_test.keys())
result = pd.DataFrame({'id': test_ids, 'label': test_predictions})
result.to_csv('/home/project/result.csv', index=False)

# 计算在训练集上的预测准确率(没有测试集的标签，无奈之举)
train_predictions = model.predict(train_features)
accuracy = accuracy_score(train_labels, train_predictions)
print("当前使用的模型在训练集上的测试效果准确率为:", accuracy)
