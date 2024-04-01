import json
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

# 载入训练数据
with open('resnet_train.json', 'r') as f1:
    train_data1 = json.load(f1)
# print(train_data1)
with open('inception_train.json', 'r') as f2:
    train_data2 = json.load(f2)
# print(train_data2)
with open('xception_train.json', 'r') as f3:
    train_data3 = json.load(f3)
# print(train_data3)

# 读取向量特征转化为vector
X_resnet = np.array([sample['feature'] for sample in train_data1.values()])
# print(X_resnet)
X_inception = np.array([sample['feature'] for sample in train_data2.values()])
# print(X_inception)
X_xception = np.array([sample['feature'] for sample in train_data3.values()])
# print(X_xception)

# 拼接向量特征
X_train = np.column_stack((X_resnet, X_inception, X_xception))
print(X_train)
# 读取样本标签
Labels_train = np.array([id['label']for id in train_data3.values()])
print(Labels_train)
# 载入测试数据
with open('test.json', 'r') as f4:
    test_data = json.load(f4)
X_test = np.array([sample['feature'] for sample in test_data.values()])
# 构建随机森林模型
model = RandomForestClassifier(n_estimators=100,random_state=1)

# 模型训练
model.fit(X_train,Labels_train)

# 测试集标签预测
Labels_test = model.predict(X_test)

print(Labels_test)

result = pd.DataFrame({'id': [sample_id for sample_id in test_data.keys()], 'label': Labels_test})
result.to_csv('/home/project/result.csv', index=False)