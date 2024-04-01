import jieba
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import pandas as pd

# 加载训练数据
with open('news_train.txt', 'r', encoding='utf-8') as f:
    train_data = f.readlines()
with open('label_newstrain.txt', 'r', encoding='utf-8') as f:
    train_labels = [int(line.strip()) for line in f.readlines()]

# 转换训练数据
train_data = [' '.join(jieba.cut(text)) for text in train_data]


pipeline = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('clf', LogisticRegression()),
])


pipeline.fit(train_data, train_labels)


with open('news_test.txt', 'r', encoding='utf-8') as f:
    test_data = f.readlines()


test_data = [' '.join(jieba.cut(text)) for text in test_data]


test_predictions = pipeline.predict(test_data)

with open('pred_test.txt', 'w', encoding='utf-8') as f:
    for prediction in test_predictions:
        f.write(str(prediction) + '\n')