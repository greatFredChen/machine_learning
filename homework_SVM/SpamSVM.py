import numpy as np
import heapq
from scipy.io import loadmat
from sklearn import svm
import pandas as pd
import re
import nltk
import nltk.stem.porter

# 简单NLP实验，该实验将使用映射词表来将词转化为特征从而构筑X
file = open('emailSample1.txt', mode='r')
email = file.read()
file.close()
# print(email)


# 处理邮件
def process_email(email):
    email = email.lower()  # lower-casing
    email = re.sub(r'<[a-zA-Z0-9]*>', ' ', email)  # remove all the HTML tags
    email = re.sub(r'(http|https)://\S*', 'httpaddr', email)  # URL可能有问号等特殊符号，只要屏蔽不可见符号 假设url带https/http头
    email = re.sub(r'\S+@\S+(.\S+)+', 'emailaddr', email)  # replace email address
    email = re.sub(r'\d+', 'number', email)  # replace number
    email = re.sub(r'\$+', 'dollar', email)  # replace $ signs
    return email


# word stemming
def word_stemming(email):
    stemmer = nltk.stem.porter.PorterStemmer()
    email = process_email(email)
    # split
    words = re.split(r'[ \@\$\/\#\.\-\:\&\*\+\=\[\]\?\!\(\)\{\}\,\'\"\>\_\<\;\%]', email)
    wordlist = []
    for word in words:
        word = re.sub('[^a-zA-Z0-9]+', '', word)  # 删除所有不为数字和英文字符的字符
        if len(word) == 0:
            # 全部为特殊字符
            continue
        stemmed = stemmer.stem(word)
        wordlist.append(stemmed)
    return wordlist


def read_vocab():
    path = 'vocab.txt'
    data = pd.read_csv(path, names=['index', 'words'], delimiter='\t')
    return data.iloc[:, 1].values  # return vocab word list


def map_word(email):
    wordlist = word_stemming(email)
    print(wordlist)
    vocab = read_vocab()
    return [i+1 for word in wordlist for i in range(len(vocab)) if vocab[i] == word]  # 当word不在vocab中时则跳过该词


# 实际上这里是one-hot编码处理  将index改为one-hot
def extract_features(email):
    vocab = read_vocab()
    maplist = map_word(email)
    print(maplist)
    _maplist = []
    for index in maplist:
        if index - 1 not in _maplist:
            _maplist.append(index - 1)  # 修正index顺便去重
    email_features = np.zeros((len(_maplist), len(vocab)))
    for i in range(len(_maplist)):
        email_features[i][_maplist[i]] = 1
    return email_features


def top_predictors(model):
    # vocab  使用extract_feature的方法构造矩阵代入model运算
    # model.predict_proba  根据关键词预测y标签的可能性，推出最可能的几个词
    vocab = read_vocab()
    probalist = []
    vocab_x = np.zeros((len(vocab), len(vocab)))
    for i in range(len(vocab)):
        vocab_x[i][i] = 1
    proba = model.predict_proba(vocab_x)
    for i in range(len(proba)):
        probalist.append(proba[i][1])
    '''pp = sorted(probalist, reverse=True)
    list2 = [pp[i] for i in range(15)]'''  # 第一种方法
    max_proba_index = list(map(probalist.index, heapq.nlargest(15, probalist)))
    # print(max_proba_index)
    return [vocab[index] for index in max_proba_index]


email_features = extract_features(email)
print('The shape of email_features is ({}, {})'.format(email_features.shape[0],
                                                       email_features.shape[1]))
# 使用自带的数据集训练
data_train = loadmat('spamTrain.mat')  # X  y
data_test = loadmat('spamTest.mat')  # Xtest  ytest
X, y = data_train['X'], data_train['y']
Xtest, ytest = data_test['Xtest'], data_test['ytest']

model = svm.SVC(C=0.1, kernel='linear', probability=True)
model.fit(X, y.flatten())  # 训练模型
train_score = model.score(X, y.flatten())
test_score = model.score(Xtest, ytest.flatten())
print('training accuracy: {}, test accuracy: {}'.format(train_score, test_score))
# model.predict输出预测的y(类别),model.predict_proba输出预测出y值的可能性
toplist = top_predictors(model)
print(toplist)
