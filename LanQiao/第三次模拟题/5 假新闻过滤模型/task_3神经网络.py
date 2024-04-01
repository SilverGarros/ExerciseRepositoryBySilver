from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import jieba

# 读取训练集文本和标签
with open('news_train.txt', 'r', encoding='utf-8') as f:
    train_data = [' '.join(jieba.cut(text)) for text in f.readlines()]
with open('label_newstrain.txt', 'r', encoding='utf-8') as f:
    train_labels = [int(label) for label in f.readlines()]

# 创建TF-IDF转换器
vectorizer = TfidfVectorizer()

# 使用TF-IDF转换训练集文本
X_train = vectorizer.fit_transform(train_data)

# 创建并训练神经网络模型
model = MLPClassifier()
model.fit(X_train, train_labels)

# 读取测试集文本
with open('news_test.txt', 'r', encoding='utf-8') as f:
    test_data = [' '.join(jieba.cut(text)) for text in f.readlines()]

# 使用TF-IDF转换测试集文本
X_test = vectorizer.transform(test_data)

# 预测测试集的类别标签
test_predictions = model.predict(X_test)

# 计算训练集的准确率
print(accuracy_score(train_labels, model.predict(X_train)))