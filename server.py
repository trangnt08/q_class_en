# -*- encoding: utf-8 -*-
__author__ = 'trangnt'

from flask import Flask, request, flash, render_template
from sklearn.externals import joblib
import uni_bigrams
from io import open


app = Flask('crf')
d = {"NUM":u"Hỏi về số lượng", "HUM":u"Hỏi về con người", "LOC":u"Hỏi về địa điểm", "DESC":u"Hỏi về thông tin mô tả", "ABBR":u"Hỏi về tên viết tắt", "ENTY":u"Hỏi về thực thể"}
with open('home.html', 'r', encoding='utf-8') as f:
	data1 = f.read()

@app.route('/',methods = ['GET','POST'])
def homepage():
    try:
        error = None
        if request.method == "GET":
            return data1
        if request.method == "POST":
            data2 = request.get_data()
            print "b", data2
            print "cc"
            kq = uni_bigrams.predict_ex(data2)
            print 'kq ',kq
            return d[kq]
    except:
        return 'err'
	return data


@app.route('/svm/', methods=['POST'])
def process_request():
    data = request.form['input']
    flash(data)
    svm = uni_bigrams.fit()
    pass
    return uni_bigrams.predict(svm, data)



if __name__ == '__main__':
    app.run(port=9000)