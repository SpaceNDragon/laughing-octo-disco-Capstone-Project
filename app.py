from flask import Flask,render_template,request
import numpy as np 
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import re

import pickle

tf = pickle.load(open('E:\\10-02-23,Developing Spam and ham Project end to end\\NLP Spam project\\Spam_ham_tfidf.pkl','rb'))
model = pickle.load(open('E:\\10-02-23,Developing Spam and ham Project end to end\\NLP Spam project\\Spam_ham.pkl','rb'))


s = PorterStemmer()

app = Flask(__name__)

def predictions(text):
    c = []
    sol = re.sub('[^a-zA-Z0-9]',' ',text)
    sol = sol.lower()
    sol = sol.split()
    sol = [i for i in sol if i not in stopwords.words('english')]
    sol = [s.stem(j) for j in sol]
    sol = ' '.join(sol)
    c.append(sol)
    sol = tf.transform(c)

    # Making predictions:
    final_vector = sol.toarray()
 
    model.predict(final_vector)

    if model.predict(final_vector)[0] == 0:
        return 'Ham Mail'
    else:
        return 'Spam'

@app.route('/')
def task_1():
    return render_template('index.html')


@app.route('/predict' , methods = ['GET','POST'])
def fun():
    if request.method == 'POST':
        text = request.form["message"]
        res = predictions(text)
        return render_template('index.html' , var = res)





if __name__ == '__main__':
    app.run(debug=True)  # saving changes while server is in live :
