from ast import pattern
from cProfile import label
from turtle import done
import spacy
from spacy import displacy

from flask import Flask, render_template, request
from flaskext.markdown import Markdown

from flask import Flask, render_template, request
import os
from werkzeug.utils import secure_filename

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from collections import Counter
from nltk.stem import WordNetLemmatizer
from gensim.corpora.dictionary import Dictionary
from gensim.models.tfidfmodel import TfidfModel
from collections import defaultdict

from unittest.util import _MAX_LENGTH
from transformers import AutoTokenizer,AutoModelForSequenceClassification

from textblob import TextBlob


app = Flask(__name__)
Markdown(app)

s = 0


@app.route('/', methods = ["GET", "POST"])
def tagging():
    global s, article
    nlp = spacy.load('en_core_web_sm')

    html = ''
    if request.method == "POST" :
        if 'subject' in request.form :
            s = 1
            article = request.form.get('subject')
            doc = nlp(article)
            html = displacy.render(doc, style ='ent')
            return render_template('display.html',msg = (request.form.get('subject')), shdis = html , bow = bowfunction(article) , tfidshow = tfid(article), predict = Predictionfakenews(article) , sentiment = sentimentfunction(article))
        if 'searchword' in request.form :
            print("Pass")
            doc = nlp(article)
            html = displacy.render(doc, style ='ent')
            word = request.form.get('searchword')
            return render_template('display.html',msg = article, shdis = html , bow = bowfunction(article) , tfidshow = tfid(article), shword = searchword(word,article),  predict = Predictionfakenews(article), sentiment = sentimentfunction(article))
        if 'checklist' in request.form :
            result = (request.form.getlist('checklist'))
            doc = nlp(article)
            options = {"ents": result}
            html = displacy.render(doc, style ='ent',options=options)
            return render_template('display.html',msg = article, shdis = html , bow = bowfunction(article) , tfidshow = tfid(article), predict = Predictionfakenews(article), sentiment = sentimentfunction(article))
    else :
        return render_template('display.html',msg = "" , shdis = html)


model_path = "fake-news-bert-base-uncased"
model = AutoModelForSequenceClassification.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

def get_prediction(text, convert_to_label=False):
    # prepare our text into tokenized sequence
    inputs = tokenizer(text, padding=True, truncation=True,max_length=512,return_tensors="pt")
    # perform inference to our model
    outputs = model(**inputs)
    # get output probabilities by doing softmax
    probs = outputs[0].softmax(1)
    # executing argmax function to get the candidate label
    d = {
        0: "Prediction FakeNews : Reliable",
        1: "Prediction FakeNews : Fake"
    }
    if convert_to_label:
        return d[int(probs.argmax())]
    else:
        return int(probs.argmax())

def Predictionfakenews(article):
    return get_prediction(article, convert_to_label=True)

def sentimentfunction(article):
    text = article
    blob_text = TextBlob(text)
    result = blob_text.polarity
    if result <= -0.5 :
        return ("The article is : Negative sentiment")
    elif result <= 0.5 :
        return ("The article is : Neutral sentiment")
    elif result <= 1 :
        return ("The article is : Positive sentiment")
    else :
        pass

def searchword(word,article):
    articles =[]
    tokens = word_tokenize(article)
    lower_tokens = [t.lower()for t in tokens]
    alpha_only = [t for t in lower_tokens if t.isalpha()]
    no_stops = [t for t in alpha_only if t not in stopwords.words('english')]
    wordnet_lemmatizer = WordNetLemmatizer()
    lemmatized = [wordnet_lemmatizer.lemmatize(t) for t in no_stops]

    articles.append(lemmatized)

    search = articles[0]
    if word.lower() in search :
        return f"Found {word.lower()} have {search.count(word.lower())}"
    return f"Not Have {word.lower()}"

def bowfunction(article):

    tokens = word_tokenize(article)
    lower_tokens = [t.lower()for t in tokens]
    alpha_only = [t for t in lower_tokens if t.isalpha()]
    no_stops = [t for t in alpha_only if t not in stopwords.words('english')]
    wordnet_lemmatizer = WordNetLemmatizer()
    lemmatized = [wordnet_lemmatizer.lemmatize(t) for t in no_stops]
    bow = Counter(lemmatized)
    l = bow.most_common(5)
    return l

def tfid(article):
    articles =[[]]
    mylist = []

    tokens = word_tokenize(article)
    lower_tokens = [t.lower()for t in tokens]
    alpha_only = [t for t in lower_tokens if t.isalpha()]
    no_stops = [t for t in alpha_only if t not in stopwords.words('english')]
    wordnet_lemmatizer = WordNetLemmatizer()
    lemmatized = [wordnet_lemmatizer.lemmatize(t) for t in no_stops]

    articles.append(lemmatized)

    dictionary = Dictionary(articles)

    corpus =[dictionary.doc2bow(a)for a in articles]
    doc = corpus[1]

    tfidf = TfidfModel(corpus)
    tfidf_weights=tfidf[doc]
    sorted_tfidf_weights = sorted(tfidf_weights, key=lambda w:w[1],
                        reverse=True)
    for term_id, weight in sorted_tfidf_weights[:5]:
        result = (dictionary.get(term_id),weight)
        mylist.append(result)
    return(mylist)

 



if __name__ == "__main__":
   import os
   HOST = os.environ.get('SERVER_HOST', '0.0.0.0')
   try:
      PORT = int(os.environ.get('SERVER_PORT','8000'))
   except ValueError:
         PORT = 8000
   app.debug = True
   app.run(HOST,PORT)
