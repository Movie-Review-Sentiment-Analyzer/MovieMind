from pathlib import Path
import re
import hashlib
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS
from movie_analysis import analyze_review, model, vectorizer
from preprocessing import clean_text

app = Flask(__name__)
CORS(app)

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_FILE = BASE_DIR / "data" / "imdb_reviews.csv"
METRICS_FILE = BASE_DIR / "results" / "metrics.txt"

# Cache sample reviews for quick retrieval
cached_samples = None
pos_reviews = []
neg_reviews = []

def clean_html(text):
    text = re.sub(r"<.*?>", " ", str(text))
    text = re.sub(r"\s+", " ", text).strip()
    return text

def load_dataset():
    global cached_samples, pos_reviews, neg_reviews
    if cached_samples is None and DATA_FILE.exists():
        try:
            df = pd.read_csv(DATA_FILE)
            cached_samples = df
            pos_reviews = df[df["sentiment"] == "positive"]["review"].tolist()
            neg_reviews = df[df["sentiment"] == "negative"]["review"].tolist()
            print(f"Loaded {len(df)} reviews ({len(pos_reviews)} pos, {len(neg_reviews)} neg)")
        except Exception as e:
            print("Error loading dataset:", e)
    return cached_samples

load_dataset()

CRITIC_NAMES = [
    "CinemaVerdict", "FilmAffinity_Critic", "ScreenRant_Reviewer", "IMDb_TopReviewer",
    "MovieLover_Priya", "FilmBuff_Arun", "TheCinematicView", "GlobalFilmWatcher",
    "StoryCrafter", "ReelTakeReview", "Cinephile_Sam", "EpicNarrativeReviewer"
]

@app.route("/", methods=["GET"])
def index():
    return jsonify({
        "status": "online",
        "service": "MovieMind NLP Sentiment Analysis Engine",
        "model": "TF-IDF + Logistic Regression (89.08% accuracy)",
        "endpoints": [
            "/predict (POST)",
            "/predict_batch (POST)",
            "/movie_dataset_reviews (POST)",
            "/imdb_samples (GET)",
            "/model_info (GET)"
        ]
    })


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True) or {}
    review = data.get("review", "")

    if not review or not review.strip():
        return jsonify({"error": "Review text is required"}), 400

    sentiment, confidence = analyze_review(review)

    cleaned = clean_text(review)
    review_vec = vectorizer.transform([cleaned])
    probs = model.predict_proba(review_vec)[0]
    classes = list(model.classes_)
    
    pos_idx = classes.index("positive") if "positive" in classes else 1
    neg_idx = classes.index("negative") if "negative" in classes else 0

    return jsonify({
        "sentiment": sentiment,
        "confidence": round(float(confidence), 2),
        "probabilities": {
            "positive": round(float(probs[pos_idx]) * 100, 2),
            "negative": round(float(probs[neg_idx]) * 100, 2)
        },
        "cleaned_text": cleaned
    })


@app.route("/predict_batch", methods=["POST"])
def predict_batch():
    data = request.get_json(silent=True) or {}
    items = data.get("reviews", [])

    if not items or not isinstance(items, list):
        return jsonify({"error": "reviews array is required"}), 400

    results = []
    positive_count = 0
    negative_count = 0

    for item in items:
        if isinstance(item, dict):
            review_text = item.get("content") or item.get("review") or item.get("text") or ""
            item_id = item.get("id")
            author = item.get("author") or item.get("username")
            rating = item.get("rating")
            source = item.get("source") or "Audience"
        else:
            review_text = str(item)
            item_id = None
            author = None
            rating = None
            source = "Audience"

        if not review_text.strip():
            continue

        cleaned = clean_text(review_text)
        review_vec = vectorizer.transform([cleaned])
        sentiment = model.predict(review_vec)[0]
        probs = model.predict_proba(review_vec)[0]
        confidence = max(probs) * 100

        if sentiment == "positive":
            positive_count += 1
        else:
            negative_count += 1

        res_obj = {
            "review": clean_html(review_text),
            "sentiment": sentiment,
            "confidence": round(float(confidence), 2),
            "source": source
        }
        if item_id:
            res_obj["_id"] = str(item_id)
        if author:
            res_obj["username"] = author
        if rating is not None:
            res_obj["rating"] = rating

        results.append(res_obj)

    total = len(results)
    pos_percent = round((positive_count / total) * 100, 2) if total > 0 else 0
    neg_percent = round((negative_count / total) * 100, 2) if total > 0 else 0

    return jsonify({
        "results": results,
        "summary": {
            "total": total,
            "positive_count": positive_count,
            "negative_count": negative_count,
            "positive_percent": pos_percent,
            "negative_percent": neg_percent
        }
    })


@app.route("/movie_dataset_reviews", methods=["POST"])
def movie_dataset_reviews():
    """Generates authentic IMDb dataset reviews tailored to any movie ID & rating."""
    data = request.get_json(silent=True) or {}
    movie_id = str(data.get("movie_id", "0"))
    movie_title = data.get("movie_title", "This Movie")
    tmdb_rating = float(data.get("rating", 7.5))
    count = int(data.get("count", 6))

    load_dataset()
    if not pos_reviews or not neg_reviews:
        return jsonify({"reviews": []})

    # Use hash of movie_id to deterministically select consistent reviews for the movie
    hash_val = int(hashlib.md5(movie_id.encode("utf-8")).hexdigest(), 16)

    # Determine proportion of positive vs negative reviews based on rating
    # e.g., rating 8.0 -> 5 pos, 1 neg; rating 6.0 -> 4 pos, 2 neg; rating 4.0 -> 2 pos, 4 neg
    if tmdb_rating >= 8.0:
        pos_needed = max(1, count - 1)
        neg_needed = count - pos_needed
    elif tmdb_rating >= 6.5:
        pos_needed = max(1, int(count * 0.75))
        neg_needed = count - pos_needed
    elif tmdb_rating >= 5.0:
        pos_needed = max(1, int(count * 0.5))
        neg_needed = count - pos_needed
    else:
        pos_needed = max(1, int(count * 0.25))
        neg_needed = count - pos_needed

    selected_raw = []
    for i in range(pos_needed):
        idx = (hash_val + i * 37) % len(pos_reviews)
        raw_text = clean_html(pos_reviews[idx])
        selected_raw.append({"text": raw_text, "intended": "positive", "rating": min(5, max(4, round(tmdb_rating / 2)))})

    for j in range(neg_needed):
        idx = (hash_val + (j + 10) * 43) % len(neg_reviews)
        raw_text = clean_html(neg_reviews[idx])
        selected_raw.append({"text": raw_text, "intended": "negative", "rating": max(1, min(3, round(tmdb_rating / 3)))})

    # Analyze each selected review through NLP model
    analyzed_reviews = []
    for idx, item in enumerate(selected_raw):
        text = item["text"]
        sentiment, confidence = analyze_review(text)
        author_name = CRITIC_NAMES[(hash_val + idx) % len(CRITIC_NAMES)]

        analyzed_reviews.append({
            "_id": f"imdb-ds-{movie_id}-{idx}",
            "movieId": movie_id,
            "movieTitle": movie_title,
            "username": author_name,
            "review": text,
            "rating": item["rating"],
            "sentiment": sentiment,
            "confidence": round(float(confidence), 2),
            "source": "IMDb Dataset",
            "createdAt": "2024-05-10T12:00:00Z"
        })

    return jsonify({
        "movie_id": movie_id,
        "movie_title": movie_title,
        "reviews": analyzed_reviews
    })


@app.route("/imdb_samples", methods=["GET"])
def imdb_samples():
    load_dataset()
    if cached_samples is None or len(cached_samples) == 0:
        return jsonify({"samples": []})

    count = int(request.args.get("count", 6))
    sentiment_filter = request.args.get("sentiment", None)

    sub_df = cached_samples
    if sentiment_filter in ["positive", "negative"]:
        sub_df = cached_samples[cached_samples["sentiment"] == sentiment_filter]

    sample_size = min(count, len(sub_df))
    sampled = sub_df.sample(n=sample_size).to_dict(orient="records")

    enriched = []
    for row in sampled:
        clean_row_review = clean_html(row["review"])
        pred_sentiment, pred_confidence = analyze_review(clean_row_review)
        enriched.append({
            "review": clean_row_review,
            "actual_sentiment": row["sentiment"],
            "predicted_sentiment": pred_sentiment,
            "confidence": round(float(pred_confidence), 2),
            "split": row.get("split", "test")
        })

    return jsonify({
        "samples": enriched,
        "total_dataset_size": len(cached_samples)
    })


@app.route("/model_info", methods=["GET"])
def model_info():
    metrics_text = ""
    if METRICS_FILE.exists():
        metrics_text = METRICS_FILE.read_text(encoding="utf-8")

    return jsonify({
        "model_type": "Logistic Regression with TF-IDF Vectorizer",
        "vocabulary_size": len(vectorizer.vocabulary_),
        "dataset": "IMDb Large Movie Review Dataset (50,000 reviews)",
        "accuracy": 89.08,
        "precision": 88.83,
        "recall": 89.39,
        "f1_score": 89.11,
        "metrics_report": metrics_text
    })


if __name__ == "__main__":
    print("Starting MovieMind NLP service...")
    print("NLP API running at http://127.0.0.1:8000")
    app.run(
        host="127.0.0.1",
        port=8000,
        debug=False
    )