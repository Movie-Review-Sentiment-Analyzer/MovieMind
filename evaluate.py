from pathlib import Path
import pickle

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

from preprocessing import clean_text


BASE_DIR = Path(__file__).resolve().parent.parent

DATA_FILE = BASE_DIR / "data" / "imdb_reviews.csv"

MODEL_FILE = BASE_DIR / "models" / "sentiment_model.pkl"

VECTORIZER_FILE = BASE_DIR / "models" / "tfidf_vectorizer.pkl"

RESULTS_DIR = BASE_DIR / "results"


def main():

    RESULTS_DIR.mkdir(
        exist_ok=True
    )


    print("Loading dataset...")

    df = pd.read_csv(DATA_FILE)

    test_df = df[
        df["split"] == "test"
    ]


    print("Testing reviews:", len(test_df))


    print("\nLoading model...")

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


    print("Cleaning test reviews...")

    X_test = test_df["review"].apply(
        clean_text
    )

    y_test = test_df["sentiment"]


    print("Creating TF-IDF features...")

    X_test_tfidf = vectorizer.transform(
        X_test
    )


    print("Making predictions...")

    predictions = model.predict(
        X_test_tfidf
    )


    accuracy = accuracy_score(
        y_test,
        predictions
    )

    precision = precision_score(
        y_test,
        predictions,
        pos_label="positive"
    )

    recall = recall_score(
        y_test,
        predictions,
        pos_label="positive"
    )

    f1 = f1_score(
        y_test,
        predictions,
        pos_label="positive"
    )


    print("\n==============================")
    print("MODEL EVALUATION")
    print("==============================")

    print(
        "Accuracy :",
        round(accuracy * 100, 2),
        "%"
    )

    print(
        "Precision:",
        round(precision * 100, 2),
        "%"
    )

    print(
        "Recall   :",
        round(recall * 100, 2),
        "%"
    )

    print(
        "F1 Score :",
        round(f1 * 100, 2),
        "%"
    )


    print("\nClassification Report:")

    report = classification_report(
        y_test,
        predictions
    )

    print(report)


    metrics_file = RESULTS_DIR / "metrics.txt"

    with open(
        metrics_file,
        "w"
    ) as file:

        file.write(
            "MovieMind NLP Model Evaluation\n"
        )

        file.write(
            "==============================\n\n"
        )

        file.write(
            f"Accuracy : {accuracy * 100:.2f}%\n"
        )

        file.write(
            f"Precision: {precision * 100:.2f}%\n"
        )

        file.write(
            f"Recall   : {recall * 100:.2f}%\n"
        )

        file.write(
            f"F1 Score : {f1 * 100:.2f}%\n\n"
        )

        file.write(
            report
        )


    cm = confusion_matrix(
        y_test,
        predictions,
        labels=["negative", "positive"]
    )


    plt.figure(
        figsize=(7, 5)
    )

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        xticklabels=["Negative", "Positive"],
        yticklabels=["Negative", "Positive"]
    )

    plt.xlabel("Predicted")

    plt.ylabel("Actual")

    plt.title(
        "Movie Review Sentiment Confusion Matrix"
    )

    plt.tight_layout()


    confusion_file = RESULTS_DIR / "confusion_matrix.png"

    plt.savefig(
        confusion_file
    )

    plt.close()


    print("\nResults saved in:")
    print(RESULTS_DIR)


if __name__ == "__main__":
    main()