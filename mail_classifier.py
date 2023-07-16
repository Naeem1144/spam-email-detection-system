import streamlit as st
import pickle
import nltk
import string
from nltk.stem.porter import PorterStemmer

stem = PorterStemmer()

def MakeClean(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    x = []

    for i in text:
        if i.isalnum():
            x.append(i)

    text = x[:]
    x.clear()

    for i in text:
        if i not in nltk.corpus.stopwords.words('english') and i not in string.punctuation:
            x.append(i)

    text = x[:]
    x.clear()

    for i in text:
        x.append(stem.stem(i))

    return " ".join(x)


pipe = pickle.load(open("Bernoulli_model_for_email.pkl", "rb"))


st.title('Spam Mail Classifier')

input_mail = st.text_area('enter your text/mail')

if st.button('Predict'):

    transformed_text = MakeClean(text=input_mail)

    result = pipe.predict([transformed_text])[0]

    if result == 1:
        st.header("phishing mail")
    else:
        st.header("safe mail")

