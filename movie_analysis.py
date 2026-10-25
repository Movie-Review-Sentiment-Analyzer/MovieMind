from pathlib import Path
import pickle

from preprocessing import clean_text


BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_FILE = BASE_DIR / "models" / "sentiment_model.pkl"
VECTORIZER_FILE = BASE_DIR / "models" / "tfidf_vectorizer.pkl"


with open(MODEL_FILE, "rb") as file:
    model = pickle.load(file)

with open(VECTORIZER_FILE, "rb") as file:
    vectorizer = pickle.load(file)


def analyze_review(review):

    cleaned_review = clean_text(review)

    review_vector = vectorizer.transform(
        [cleaned_review]
    )

    sentiment = model.predict(
        review_vector
    )[0]

    probabilities = model.predict_proba(
        review_vector
    )[0]

    confidence = max(probabilities) * 100

    return sentiment, confidence