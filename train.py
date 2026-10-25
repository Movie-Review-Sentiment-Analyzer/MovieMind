from pathlib import Path
import pickle

import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

from preprocessing import clean_text


BASE_DIR = Path(__file__).resolve().parent.parent

DATA_FILE = BASE_DIR / "data" / "imdb_reviews.csv"

MODEL_DIR = BASE_DIR / "models"

MODEL_FILE = MODEL_DIR / "sentiment_model.pkl"

VECTORIZER_FILE = MODEL_DIR / "tfidf_vectorizer.pkl"


def main():

    print("Loading dataset...")

    df = pd.read_csv(DATA_FILE)

    print("Total reviews:", len(df))

    train_df = df[df["split"] == "train"]

    test_df = df[df["split"] == "test"]

    print("\nTraining reviews:", len(train_df))
    print("Testing reviews:", len(test_df))


    print("\nCleaning training reviews...")

    X_train = train_df["review"].apply(clean_text)

    print("Cleaning testing reviews...")

    X_test = test_df["review"].apply(clean_text)


    y_train = train_df["sentiment"]

    y_test = test_df["sentiment"]


    print("\nCreating TF-IDF features...")

    vectorizer = TfidfVectorizer(
        max_features=20000,
        ngram_range=(1, 2),
        min_df=2,
        sublinear_tf=True
    )


    X_train_tfidf = vectorizer.fit_transform(X_train)

    X_test_tfidf = vectorizer.transform(X_test)


    print("Training TF-IDF shape:", X_train_tfidf.shape)

    print("\nTraining Logistic Regression model...")

    model = LogisticRegression(
        max_iter=1000
    )

    model.fit(
        X_train_tfidf,
        y_train
    )


    print("\nEvaluating model...")

    predictions = model.predict(X_test_tfidf)

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    print("\nAccuracy:", round(accuracy * 100, 2), "%")

    print("\nClassification Report:")

    print(
        classification_report(
            y_test,
            predictions
        )
    )


    MODEL_DIR.mkdir(
        exist_ok=True
    )


    with open(
        MODEL_FILE,
        "wb"
    ) as file:

        pickle.dump(
            model,
            file
        )


    with open(
        VECTORIZER_FILE,
        "wb"
    ) as file:

        pickle.dump(
            vectorizer,
            file
        )


    print("\nModel saved:")
    print(MODEL_FILE)

    print("\nVectorizer saved:")
    print(VECTORIZER_FILE)


if __name__ == "__main__":
    main()