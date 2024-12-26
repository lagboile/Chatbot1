import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords.words')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('averaged_perceptron_tagger')
import streamlit as st
import string
from nltk.tokenize import word_tokenize,sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

with open('C:/Users/LENOVO/chatbot/venv/Dictionnary.txt','r',encoding='utf-8') as f :

    data = f.read().replace('\n','')
    corpus = sent_tokenize(data)

def preprocess(text):
    words = word_tokenize(text)
    words = [word.lower()
             for word in words
             if word.lower() not in stopwords.words('english') and word not in string.punctuation]
    legitimatise = WordNetLemmatizer()
    words = [legitimatise.lemmatize(word) for word in words]
    return words


def get_most_relevant_sentence(query, corpus_local):
    query_words = set(preprocess(query))
    max_similarity = 0
    most_relevant_sentence = ""

    for sentence in corpus_local:
        sentence_words = set(preprocess(sentence))
        union_size = len(query_words.union(sentence_words))
        if union_size == 0:
            continue

        similarity = len(query_words.intersection(sentence_words)) / float(
            len(query_words.union(sentence_words)))

        if similarity > max_similarity:
            max_similarity = similarity
            most_relevant_sentence = sentence
    return most_relevant_sentence


def chatbot(query, corpus_local):
    most_relevant_sentence = get_most_relevant_sentence(query, corpus_local)

    if most_relevant_sentence:
        return most_relevant_sentence
    return "stp"


def main():
    st.title("Chatbot de l'amour")

    st.write("Bonjour !")

    query = st.text_input(" ")

    if st.button("submit"):
        response = chatbot(query, corpus)

        st.write("Dr love: ", response)


if __name__ == "__main__":
    main()