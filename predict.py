from pathlib import Path
import pickle

from preprocessing import clean_text


BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_FILE = BASE_DIR / "models" / "sentiment_model.pkl"

VECTORIZER_FILE = BASE_DIR / "models" / "tfidf_vectorizer.pkl"


with open(
    MODEL_FILE,
    "rb"
) as file:

    model = pickle.load(file)


with open(
    VECTORIZER_FILE,
    "rb"
) as file:

    vectorizer = pickle.load(file)


def predict_sentiment(review):

    cleaned_review = clean_text(
        review
    )

    review_vector = vectorizer.transform(
        [cleaned_review]
    )

    prediction = model.predict(
        review_vector
    )[0]

    probabilities = model.predict_proba(
        review_vector
    )[0]

    confidence = max(
        probabilities
    ) * 100

    return prediction, confidence


def main():

    print("==============================")
    print("MovieMind Sentiment Analyzer")
    print("==============================")

    while True:

        review = input(
            "\nEnter movie review (or type 'exit'): "
        )

        if review.lower() == "exit":
            break


        sentiment, confidence = predict_sentiment(
            review
        )


        print("\nSentiment :", sentiment)

        print(
            "Confidence:",
            round(confidence, 2),
            "%"
        )


if __name__ == "__main__":
    main()