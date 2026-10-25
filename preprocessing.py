import re
import nltk

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


def download_nltk_data():

    resources = {
        "corpora/stopwords": "stopwords",
        "corpora/wordnet": "wordnet",
        "corpora/omw-1.4": "omw-1.4"
    }

    for path, resource in resources.items():

        try:
            nltk.data.find(path)

        except LookupError:
            nltk.download(resource)


download_nltk_data()

stop_words = set(stopwords.words("english"))

lemmatizer = WordNetLemmatizer()


def clean_text(text):

    text = re.sub(r"<.*?>", " ", text)

    text = re.sub(r"[^a-zA-Z\s]", " ", text)

    text = text.lower()

    words = text.split()

    words = [
        lemmatizer.lemmatize(word)
        for word in words
        if word not in stop_words
    ]

    return " ".join(words)