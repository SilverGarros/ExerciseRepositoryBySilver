from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import jieba

# 读取训练集文本和标签
with open('news_train.txt', 'r', encoding='utf-8') as f:
    train_data = [' '.join(jieba.cut(text)) for text in f.readlines()]
# print(train_data)
with open('label_newstrain.txt', 'r', encoding='utf-8') as f:
    train_labels = [int(label) for label in f.readlines()]

# 创建TF-IDF转换器
vectorizer = TfidfVectorizer()

# 使用TF-IDF转换训练集文本
X_train = vectorizer.fit_transform(train_data)

# 创建并训练随机森林模型
model = RandomForestClassifier()
model.fit(X_train, train_labels)

# 读取测试集文本
with open(r'news_test.txt', 'r', encoding='utf-8') as f:
    test_data = [' '.join(jieba.cut(text)) for text in f.readlines()]

# 使用TF-IDF转换测试集文本
X_test = vectorizer.transform(test_data)
print(X_test)

# 预测测试集的类别标签
test_predictions = model.predict(X_test)

# 计算训练集的准确率
print("训练集上的准确率为：")
print(accuracy_score(train_labels, model.predict(X_train)))

with open('pred_test.txt', 'w', encoding='utf-8') as f:
    f.write('\n'.join(map(str, test_predictions)))

# 保存预测结果
with open('pred_test.txt', 'w', encoding='utf-8') as f:
    for i, label in enumerate(test_predictions):
        f.write(str(int(label)))
        if i != len(test_predictions) - 1:
            f.write('\n')