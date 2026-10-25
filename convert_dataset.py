from pathlib import Path
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "aclImdb"
OUTPUT_FILE = BASE_DIR / "data" / "imdb_reviews.csv"


def read_reviews(folder, sentiment, split):
    rows = []

    folder_path = DATA_DIR / split / folder

    for file_path in folder_path.glob("*.txt"):
        try:
            review = file_path.read_text(
                encoding="utf-8",
                errors="ignore"
            )

            rows.append({
                "review": review,
                "sentiment": sentiment,
                "split": split
            })

        except Exception as e:
            print("Error reading:", file_path)
            print(e)

    return rows


def main():

    data = []

    print("Reading training positive reviews...")
    data.extend(read_reviews("pos", "positive", "train"))

    print("Reading training negative reviews...")
    data.extend(read_reviews("neg", "negative", "train"))

    print("Reading testing positive reviews...")
    data.extend(read_reviews("pos", "positive", "test"))

    print("Reading testing negative reviews...")
    data.extend(read_reviews("neg", "negative", "test"))

    df = pd.DataFrame(data)

    df.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8"
    )

    print("\nDataset converted successfully.")
    print("Total reviews:", len(df))
    print("\nClass distribution:")
    print(df["sentiment"].value_counts())

    print("\nSplit distribution:")
    print(df["split"].value_counts())

    print("\nSaved to:")
    print(OUTPUT_FILE)


if __name__ == "__main__":
    main()