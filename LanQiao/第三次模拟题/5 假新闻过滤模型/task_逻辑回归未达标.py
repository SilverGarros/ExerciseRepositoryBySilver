import jieba
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
import pandas as pd

# 载入训练数据和标签
with open('news_train.txt', 'r', encoding='utf-8') as f:
    train_data = f.readlines()
    print(len(train_data))
with open('label_newstrain.txt', 'r', encoding='utf-8') as f:
    train_labels = [int(line.strip()) for line in f.readlines()]
    print(len(train_labels))
    # 不用int转为整形会报错


train_data = [' '.join(jieba.cut(text)) for text in train_data]

pipeline = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('clf', LogisticRegression()),
])


pipeline.fit(train_data, train_labels)

with open('news_test.txt', 'r', encoding='utf-8') as f:
    test_data = f.readlines()


test_data = [' '.join(jieba.cut(text)) for text in test_data]
print(len(test_data))

test_predictions = pipeline.predict(test_data)

print(accuracy_score(train_labels,pipeline.predict(train_data)))

with open('pred_test.txt', 'w', encoding='utf-8') as f:
    f.write('\n'.join(map(str, test_predictions)))