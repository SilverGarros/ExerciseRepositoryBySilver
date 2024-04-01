from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import jieba

# 读取训练集文本和标签
with open(r'G:\Project Silver\LanQiao\第三次模拟题\5 假新闻过滤模型\news_train.txt', encoding='utf-8') as f:
    train_data = [' '.join(jieba.cut(text)) for text in f.readlines()]
with open(r'G:\Project Silver\LanQiao\第三次模拟题\5 假新闻过滤模型\label_news_train.txt', encoding='utf-8') as f:
    train_labels = [int(label) for label in f.readlines()]

# 创建一个pipeline，包含TF-IDF转换和SVM模型
pipeline = make_pipeline(TfidfVectorizer(), SVC())

# 训练模型
pipeline.fit(train_data, train_labels)

# 读取测试集文本
with open(r'G:\Project Silver\LanQiao\第三次模拟题\5 假新闻过滤模型\news_test.txt', 'r', encoding='utf-8') as f:
    test_data = [' '.join(jieba.cut(text)) for text in f.readlines()]

# 预测测试集的类别标签
test_predictions = pipeline.predict(test_data)

# 计算训练集的准确率
print(accuracy_score(train_labels, pipeline.predict(train_data)))