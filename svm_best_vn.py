# -*- encoding: utf8 -*-
import re

import unicodedata
from sklearn.ensemble import RandomForestClassifier

from sklearn.externals import joblib
from sklearn.metrics import accuracy_score
import datetime
import pandas as pd
import time
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import os

from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix


def time_diff_str(t1, t2):
    """
    Calculates time durations.
    """
    diff = t2 - t1
    mins = int(diff / 60)
    secs = round(diff % 60, 2)
    return str(mins) + " mins and " + str(secs) + " seconds"

def load_model(model):
    print('loading model ...' + model)
    if os.path.isfile(model):
        return joblib.load(model)
    else:
        return None

def clean_str_vn(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    """
    string = re.sub(r"[~`@#$%^&*-+]", " ", string)
    def sharp(str):
        b = re.sub('\s[A-Za-z]\s\.', ' .', ' '+str)
        while (b.find('. . ')>=0): b = re.sub(r'\.\s\.\s', '. ', b)
        b = re.sub(r'\s\.\s', ' # ', b)
        return b
    string = sharp(string)
    string = re.sub(r" : ", ":", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", "", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()

def review_to_words(review, filename):
    # 1. Convert to lower case, split into individual words
    # words = review.lower().split()
    words = review.split()
    with open(filename, "r") as f3:
        dict_data = f3.read()
        array = dict_data.splitlines()
    meaningful_words = [w for w in words if not w in array]
    return " ".join(meaningful_words)


def review_to_words2(review, filename,n):
    with open(filename, "r") as f3:
        dict_data = f3.read()
        array = dict_data.splitlines()
    words = [' '.join(x) for x in ngrams(review, n)]
    meaningful_words = [w for w in words if not w in array]
    return build_sentence(meaningful_words)

def word_clean(array, review):
    words = review.lower().split()
    meaningful_words = [w for w in words if w in array]
    return " ".join(meaningful_words)
#
# def print_words_frequency(train_data_features):
#     # Take a look at the words in the vocabulary
#     vocab = vectorizer.get_feature_names()
#     print "Words in vocabulary:", vocab
#
#     # Sum up the counts of each vocabulary word
#     dist = np.sum(train_data_features, axis=0)
#
#     # For each, print the vocabulary word and the number of times it
#     # appears in the training set
#     print "Words frequency..."
#     for tag, count in zip(vocab, dist):
#         print count, tag

def ngrams(input, n):
  input = input.split(' ')
  output = []
  for i in range(len(input)-n+1):
    output.append(input[i:i+n])
  return output # output dang ['a b','b c','c d']

def ngrams2(input, n):
  input = input.split(' ')
  output = {}
  for i in range(len(input)-n+1):
    g = ' '.join(input[i:i+n])
    output.setdefault(g, 0)
    output[g] += 1
  return output # output la tu dien cac n-gram va tan suat cua no {'a b': 1, 'b a': 1, 'a a': 3}

def ngrams_array(arr,n):
    output = {}
    for x in arr:
        d = ngrams2(x, n)  # moi d la 1 tu dien
        for x in d:
            count = d.get(x)
            output.setdefault(x, 0)
            output[x] += count
    return output

# def build_dict(arr,n,m):
#     d={}
#     ngram = ngrams_array(arr,n)
#     for x in ngram:
#         p = ngram.get(x)
#         if p < m:
#             d.setdefault(x,p)
#     return d
def buid_dict(filename,arr,n,m):
    with open(filename, 'r') as f:
        ngram = ngrams_array(arr, n)
        for x in ngram:
            p = ngram.get(x)
            if p < m:
                f.write(x)

def build_sentence(input_arr):
    d = {}
    for x in range(len(input_arr)):
        d.setdefault(input_arr[x], x)
    chuoi = []
    for i in input_arr:
        x = d.get(i)
        if x == 0:
            chuoi.append(i)
        for j in input_arr:
            y = d.get(j)
            if y == x + 1:
                z = j.split(' ')
                chuoi.append(z[1])
    return " ".join(chuoi)

def clean_doc(question):
    rm_junk_mark = re.compile(ur'[?,\.\n]')
    normalize_special_mark = re.compile(ur'(?P<special_mark>[\.,\(\)\[\]\{\};!?:“”\"\`\'/])')
    question = normalize_special_mark.sub(u' \g<special_mark> ', question)
    question = rm_junk_mark.sub(u'', question)
    question = re.sub(' +', ' ', question)  # remove multiple spaces in a string
    return question

def load_data(filename, dict):
    col1 = []; col2 = []; col3 = []; col4 = []

    with open(filename, 'r') as f,open(dict, "w") as f2:
        for line in f:
            label, question = line.split(" ", 1)
            label1, label2 = label.split(":")
            question = clean_doc(question)
            # question = review_to_words(question,'datavn/question_stopwords.txt')
            # question = clean_str_vn(question)
            # if filename == 'data/train_5500.label':
            #     if label1 == "ABBR":
            #         for i in xrange(6):
            #             col1.append(label1)
            #             col2.append(label2)
            #             col3.append(question)
            # if label1
            col1.append(label1)
            col2.append(label2)
            col3.append(question)

        ngram = ngrams_array(col3,2)    # tu dien cac tu va so lan xuat hien cua no
        dict_arr = []       # list cac tu co tan suat < 1
        for x in ngram:
            p = ngram.get(x)
            if p<1:
                dict_arr.append(x)
                f2.write(x+"\n")
        col4 = []
        for q in col3:
            q = review_to_words2(q, dict, 2)  # q la 1 cau
            q1 = [' '.join(x) for x in ngrams(q, 1)]  # q1:mang cac 1-grams
            q2 = [' '.join(x) for x in ngrams(q, 2)]  # q2: mang cac phan tu 2-grams
            q3 = [' '.join(x.replace(' ', '_') for x in q2)]
            y = q1 + q3
            z = " ".join(y)
            col4.append(z)
        d = {"label1":col1, "label2":col2, "question": col4}

        train = pd.DataFrame(d)
        if filename == 'data/train_5500.label':
            joblib.dump(train, 'model2/train5.pkl')
        else:
            joblib.dump(train, 'model2/test5.pkl')
    return train


def svm():
    train = load_model('model2/train5.pkl')
    if train is None:
        train = load_data('data/train_5500.label', 'datavn/dict1')

    vectorizer = load_model('model2/vectorizer5.pkl')
    if vectorizer == None:
        # vectorizer = TfidfVectorizer(ngram_range=(1, 1), max_df=0.7, min_df=2, max_features=1000)
        vectorizer = CountVectorizer(ngram_range=(1,1), max_df=0.7, min_df=2, max_features=1000)
    test = load_model('model2/test5.pkl')
    if test is None:
        test = load_data('data/TREC_10.label', 'datavn/dict2')

    print "Data dimensions:", train.shape
    print "List features:", train.columns.values
    print "First review:", train["label1"][0], "|", train["question"][0]

    print "Data dimensions:", test.shape
    print "List features:", test.columns.values
    print "First review:", test["label1"][0], "|", test["question"][0]
    # train, test = train_test_split(train, test_size=0.2)

    train_text = train["question"].values
    test_text = test["question"].values

    vectorizer.fit(train_text)
    X_train = vectorizer.transform(train_text)
    joblib.dump(vectorizer, 'model2/vectorizer5.pkl')
    X_train = X_train.toarray()
    y_train = train["label1"]
    y_train2 = train["label2"]

    X_test = vectorizer.transform(test_text)
    X_test = X_test.toarray()
    y_test = test["label1"]
    y_test2 = test["label2"]
    # joblib.dump(vectorizer, 'model/vectorizer2.pkl')
    print "---------------------------"
    print "Training"
    print "---------------------------"
    names = ["RBF SVC"]
    t0 = time.time()
    # iterate over classifiers
    clf = load_model('model2/uni_big5.pkl')
    if clf is None:
        clf = SVC(kernel='rbf', C=1000, class_weight='balanced')
        clf.fit(X_train, y_train)
        joblib.dump(clf, 'model2/uni_big5.pkl')
    y_pred = clf.predict(X_test)
    print y_pred

    print " accuracy: %0.3f" % accuracy_score(y_test, y_pred)
    print " %s - Converting completed %s" % (datetime.datetime.now(), time_diff_str(t0, time.time()))
    print "confuse matrix: \n", confusion_matrix(y_test, y_pred, labels=["ABBR", "DESC", "ENTY", "HUM", "LOC", "NUM"])

    print "-----------------------"
    print "fine grained category"
    print "-----------------------"
    clf2 = SVC(kernel='rbf', C=1000, class_weight='balanced')
    clf2.fit(X_train, y_train2)
    y_pred2 = clf2.predict(X_test)
    print y_pred2

    print " accuracy: %0.3f" % accuracy_score(y_test2,y_pred2)

def rf():
    train = load_model('model2/train5a.pkl')
    if train is None:
        train = load_data('data/train_5500.label', 'datavn/dict1')

    vectorizer = load_model('model2/vectorizer5a.pkl')
    if vectorizer == None:
        vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_df=0.7, min_df=2, max_features=1000)
    test = load_model('model2/test5a.pkl')
    if test is None:
        test = load_data('data/TREC_10.label', 'datavn/dict2')

    print "Data dimensions:", train.shape
    print "List features:", train.columns.values
    print "First review:", train["label1"][0], "|", train["question"][0]

    print "Data dimensions:", test.shape
    print "List features:", test.columns.values
    print "First review:", test["label1"][0], "|", test["question"][0]
    # train, test = train_test_split(train, test_size=0.2)

    train_text = train["question"].values
    test_text = test["question"].values

    vectorizer.fit(train_text)
    X_train = vectorizer.transform(train_text)
    joblib.dump(vectorizer, 'model2/vectorizer5a.pkl')
    X_train = X_train.toarray()
    y_train = train["label1"]
    y_train2 = train["label2"]

    X_test = vectorizer.transform(test_text)
    X_test = X_test.toarray()
    y_test = test["label1"]
    y_test2 = test["label2"]
    # joblib.dump(vectorizer, 'model/vectorizer2.pkl')
    print "---------------------------"
    print "Training"
    print "---------------------------"
    names = ["RBF SVC"]
    t0 = time.time()
    # iterate over classifiers
    clf = load_model('model3/forest5a.pkl')
    if clf is None:
        clf = RandomForestClassifier(n_estimators=100)
        clf.fit(X_train, y_train)
        joblib.dump(clf, 'model3/forest5a.pkl')
    y_pred = clf.predict(X_test)
    print y_pred

    print " accuracy: %0.3f" % accuracy_score(y_test, y_pred)
    print " %s - Converting completed %s" % (datetime.datetime.now(), time_diff_str(t0, time.time()))
    print "confuse matrix: \n", confusion_matrix(y_test, y_pred, labels=["ABBR", "DESC", "ENTY", "HUM", "LOC", "NUM"])

    print "-----------------------"
    print "fine grained category"
    print "-----------------------"
    clf2 = RandomForestClassifier(n_estimators=100)
    clf2.fit(X_train, y_train2)
    y_pred2 = clf2.predict(X_test)
    print y_pred2

    print " accuracy: %0.3f" % accuracy_score(y_test2, y_pred2)

if __name__ == '__main__':
    svm()
    # rf()
