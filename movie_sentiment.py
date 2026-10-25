from pathlib import Path
import pickle
import pandas as pd

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


def analyze_movie_reviews(reviews):

    results = []

    for review in reviews:

        sentiment, confidence = analyze_review(
            review
        )

        results.append({
            "review": review,
            "sentiment": sentiment,
            "confidence": round(confidence, 2)
        })

    return results


def calculate_movie_sentiment(results):

    df = pd.DataFrame(results)

    total = len(df)

    positive = len(
        df[df["sentiment"] == "positive"]
    )

    negative = len(
        df[df["sentiment"] == "negative"]
    )

    positive_percent = (
        positive / total
    ) * 100

    negative_percent = (
        negative / total
    ) * 100

    return {
        "total_reviews": total,
        "positive": positive,
        "negative": negative,
        "positive_percent": round(
            positive_percent, 2
        ),
        "negative_percent": round(
            negative_percent, 2
        )
    }


def main():

    movie_reviews = [
        "The story was amazing and the acting was excellent.",
        "I loved this movie. The visuals were fantastic.",
        "The movie was boring and too slow.",
        "The acting was good but the story was weak.",
        "Absolutely wonderful movie. I enjoyed every scene."
    ]

    results = analyze_movie_reviews(
        movie_reviews
    )

    print("\nIndividual Review Analysis")
    print("==========================")

    for result in results:

        print("\nReview:")
        print(result["review"])

        print(
            "Sentiment:",
            result["sentiment"]
        )

        print(
            "Confidence:",
            result["confidence"],
            "%"
        )


    summary = calculate_movie_sentiment(
        results
    )

    print("\n\nMovie Sentiment Summary")
    print("======================")

    print(
        "Total Reviews:",
        summary["total_reviews"]
    )

    print(
        "Positive:",
        summary["positive_percent"],
        "%"
    )

    print(
        "Negative:",
        summary["negative_percent"],
        "%"
    )


if __name__ == "__main__":
    main()