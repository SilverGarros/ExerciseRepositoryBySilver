import jieba
import numpy as np
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity

w2v_file_path = "word2vec_model.bin"
W2V_MODEL= Word2Vec.load(w2v_file_path)
W2V_SIZE = 100

def get_w2v(word):
    #TODO
    # print(word)
    try:
        # 在 gensim 的 Word2Vec 模型中，wv 属性用于访问和操作词向量。
        # 例如，`model.wv['word']` 来获取 'word' 的向量，
        # 或者使用 `model.wv.most_similar('word')` 来获取与 'word' 最相似的词。
        #    >>> vector = model.wv['computer']  # get numpy vector of a word
        #    >>> sims = model.wv.most_similar('computer', topn=10)  # get other similar words
        word_vector=W2V_MODEL.wv[word]
        return word_vector
    except KeyError:
        return None



def get_sentence_vector(sentence):

    #TODO
    Word_Vectors = []
    for word in sentence:
        Word_Vector=get_w2v(word)
        # print(type(Word_Vector))
        # print(Word_Vector)
        if Word_Vector is not None:
            Word_Vectors.append(Word_Vector)
       # 在 Word_Vectors []中至少得有一个词向量时计算所有词向量的平均值，/
       # 得到句子向量，如果 Word_Vectors 为空，则返回一个全零向量 
    if Word_Vectors:
        sentence_vector = np.mean(Word_Vectors,axis=0)
    else:
        sentence_vector = np.zeros(100)
    return sentence_vector


def get_similarity(array1, array2):
    array1_2d = np.reshape(array1, (1, -1))
    array2_2d = np.reshape(array2, (1, -1))
    similarity = cosine_similarity(array1_2d, array2_2d)[0][0]
    return similarity

def main():
    
    # 测试两个句子
    sentence1 = '我不喜欢看新闻。'
    sentence2 = '我觉得新闻不好看。'
    sentence_split1 = jieba.lcut(sentence1)
    sentence_split2 = jieba.lcut(sentence2)
    # 获取句子的句向量
    sentence1_vector = get_sentence_vector(sentence_split1)
    sentence2_vector = get_sentence_vector(sentence_split2)
    # 计算句子的相似度
    similarity = get_similarity(sentence1_vector, sentence2_vector)
    print(similarity) 

if __name__ == '__main__':
    main()