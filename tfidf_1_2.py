# -*- encoding: utf8 -*-
import re

import unicodedata
from sklearn.ensemble import RandomForestClassifier

from sklearn.externals import joblib
from sklearn.metrics import accuracy_score
import datetime
import pandas as pd
import time
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import os

from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix


def time_diff_str(t1, t2):
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

def remove_stopword(question, filename):
    # 1. Convert to lower case, split into individual words
    # words = review.lower().split()
    words = question.split()
    with open(filename, "r") as f3:
        dict_data = f3.read()
        array = dict_data.splitlines()
    meaningful_words = [w for w in words if not w in array]
    return " ".join(meaningful_words)

def word_clean(array, review):
    words = review.lower().split()
    meaningful_words = [w for w in words if w in array]
    return " ".join(meaningful_words)


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

def load_data(filename):
    col1 = []; col2 = []; col3 = []; col4 = []
    with open(filename, 'r') as f:
        for line in f:
            label, question = line.split(" ", 1)
            label1, label2 = label.split(":")
            question = clean_doc(question)
            # question = review_to_words(question,'datavn/question_stopwords.txt')
            # question = clean_str_vn(question)
            col1.append(label1)
            col2.append(label2)
            col3.append(question)

        d = {"label1":col1, "label2":col2, "question": col3}

        train = pd.DataFrame(d)
        if filename == 'data/train_5500.label':
            joblib.dump(train, 'model2/train_tfidf12.pkl')
        else:
            joblib.dump(train, 'model2/test_tfidf12.pkl')
    return train


def svm():
    train = load_model('model2/train_tfidf12.pkl')
    if train is None:
        train = load_data('data/train_5500.label')

    vectorizer = load_model('model2/vectorizer_tfidf12.pkl')
    if vectorizer == None:
        vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_df=0.7, min_df=2, max_features=1000)
    test = load_model('model/test_tfidf1.pkl')
    if test is None:
        test = load_data('data/TREC_10.label')

    print "Data dimensions:", train.shape
    print "List features:", train.columns.values
    print "First review:", train["label1"][0], "|", train["question"][0]

    print "Data dimensions:", test.shape
    print "List features:", test.columns.values
    print "First review:", test["label1"][0], "|", test["question"][0]

    train_text = train["question"].values
    test_text = test["question"].values

    vectorizer.fit(train_text)
    X_train = vectorizer.transform(train_text)
    joblib.dump(vectorizer, 'model2/vectorizer_tfidf12.pkl')
    X_train = X_train.toarray()
    y_train = train["label1"]
    y_train2 = train["label2"]

    X_test = vectorizer.transform(test_text)
    X_test = X_test.toarray()
    y_test = test["label1"]
    y_test2 = test["label2"]
    print "---------------------------"
    print "Training"
    print "---------------------------"
    names = ["RBF SVC"]
    # iterate over classifiers
    clf = load_model('model2/tfidf12.pkl')
    if clf is None:
        t0 = time.time()
        clf = SVC(kernel='rbf', C=1000, class_weight='balanced')
        clf.fit(X_train, y_train)
        joblib.dump(clf, 'model2/tfidf12.pkl')
        print " %s - Training completed %s" % (datetime.datetime.now(), time_diff_str(t0, time.time()))
    t1 = time.time()
    y_pred = clf.predict(X_test)
    print " %s - Converting completed %s" % (datetime.datetime.now(), time_diff_str(t1, time.time()))
    print " accuracy: %0.3f" % accuracy_score(y_test, y_pred)

    print "confuse matrix: \n", confusion_matrix(y_test, y_pred, labels=["ABBR", "DESC", "ENTY", "HUM", "LOC", "NUM"])

    print "-----------------------"
    print "fine grained category"
    print "-----------------------"
    clf2 = load_model('model2/tfidf_fine12.pkl')
    if clf2 is None:
        t2 = time.time()
        clf2 = SVC(kernel='rbf', C=1000, class_weight='balanced')
        clf2.fit(X_train, y_train2)
        joblib.dump(clf, 'model2/tfidf_fine12.pkl')
        print " %s - Training for fine grained category completed %s" % (datetime.datetime.now(), time_diff_str(t2, time.time()))
    t3 = time.time()
    y_pred2 = clf2.predict(X_test)
    print " %s - Converting completed %s" % (datetime.datetime.now(), time_diff_str(t3, time.time()))
    print " accuracy for fine grained category: %0.3f\n" % accuracy_score(y_test2,y_pred2)

if __name__ == '__main__':
    svm()
    # rf()
