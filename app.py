"""
app.py - Flask web application for TruthLens (AI Fake News Detection).
Run:  python app.py     then open  http://127.0.0.1:5000
"""

import os

import joblib
from flask import Flask, jsonify, redirect, render_template, request, url_for

import database
from core import predict, explain, writing_signals
from news_feed import FEEDS, fetch_feed
from train import MODEL_FILE, run_training

app = Flask(__name__)
database.init_db()


def load_bundle():
    """Load the trained model; train it first if this is the first run."""
    if not os.path.exists(MODEL_FILE):
        print("No saved model found - training now (this happens only once)...")
        run_training()
    return joblib.load(MODEL_FILE)


BUNDLE = load_bundle()


def fetch_article(url):
    """Download a web page and return its headline + paragraph text."""
    try:
        import requests
        from bs4 import BeautifulSoup
    except ImportError:
        raise ValueError("URL analysis needs:  pip install requests beautifulsoup4")
    if not url.lower().startswith(("http://", "https://")):
        raise ValueError("The link must start with http:// or https://")
    try:
        resp = requests.get(url, timeout=8, headers={"User-Agent": "Mozilla/5.0 (TruthLens college project)"})
        resp.raise_for_status()
    except Exception:
        raise ValueError("Could not download that page. Check the link and your internet connection.")
    soup = BeautifulSoup(resp.text, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
        tag.decompose()
    title = soup.title.get_text(strip=True) if soup.title else ""
    paragraphs = [p.get_text(" ", strip=True) for p in soup.find_all("p")]
    body = " ".join(p for p in paragraphs if len(p.split()) >= 6)
    text = f"{title}. {body}".strip()
    if len(text.split()) < 20:
        raise ValueError("Could not extract enough article text from that page. Paste the text instead.")
    return text


@app.route("/")
def home():
    best = next(r for r in BUNDLE["results"] if r["name"] == BUNDLE["best_name"])
    return render_template("index.html", b=BUNDLE, best=best, page="home")


@app.route("/insights")
def insights():
    return render_template("insights.html", b=BUNDLE, page="insights")


@app.route("/history")
def history():
    return render_template("history.html", rows=database.recent(), s=database.stats(), page="history")


@app.route("/history/clear", methods=["POST"])
def clear_history():
    database.clear()
    return redirect(url_for("history"))


@app.route("/analyze", methods=["POST"])
def analyze():
    data = request.get_json(silent=True) or {}
    url = (data.get("url") or "").strip()
    text = (data.get("text") or "").strip()
    source = "Text"
    try:
        if url:
            text, source = fetch_article(url), "URL"
        if len(text.split()) < 3:
            raise ValueError("Please enter at least a few words of news text.")
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    best = BUNDLE["models"][BUNDLE["best_name"]]
    label, confidence, _ = predict(text, BUNDLE["vectorizer"], best)
    fake_words, real_words = explain(text, BUNDLE["vectorizer"], BUNDLE["explainer"])
    signals, flags = writing_signals(text)

    votes = []
    for name, model in BUNDLE["models"].items():
        l, c, _ = predict(text, BUNDLE["vectorizer"], model)
        votes.append({"name": name, "label": l, "confidence": round(c * 100, 1)})

    database.save_prediction(source, url or text, label, round(confidence * 100, 1))
    return jsonify({
        "label": label,
        "confidence": round(confidence * 100, 1),
        "model": BUNDLE["best_name"],
        "fake_words": fake_words,
        "real_words": real_words,
        "signals": signals,
        "flags": flags,
        "votes": votes,
        "source": source,
        "preview": text[:220] + ("..." if len(text) > 220 else "") if source == "URL" else "",
    })


@app.route("/trending")
def trending():
    return render_template("trending.html", feeds=list(FEEDS), b=BUNDLE, page="trending")


@app.route("/api/trending")
def api_trending():
    """Download today's headlines from RSS feeds and classify each one."""
    source = request.args.get("source", "all")
    force = request.args.get("refresh") == "1"
    if source == "all":
        names = list(FEEDS)
    elif source in FEEDS:
        names = [source]
    else:
        return jsonify({"error": "Unknown news source."}), 400

    items, errors = [], {}
    for name in names:
        try:
            items += fetch_feed(name, force=force)
        except Exception:
            errors[name] = "Feed unavailable (no internet, or the source changed its feed)"

    best = BUNDLE["models"][BUNDLE["best_name"]]
    results = []
    for it in items:
        label, confidence, _ = predict(it["text"], BUNDLE["vectorizer"], best)
        _, flags = writing_signals(it["text"])
        results.append({
            "title": it["title"],
            "summary": it["summary"][:160],
            "link": it["link"],
            "source": it["source"],
            "published": it["published"],
            "label": label,
            "confidence": round(confidence * 100, 1),
            "flags": [f for f in flags if not f.startswith("No sensational")],
            "short": len(it["text"].split()) < 8,
        })
    results.sort(key=lambda r: (r["label"] != "FAKE", -r["confidence"]))  # suspicious first
    fake = sum(1 for r in results if r["label"] == "FAKE")
    return jsonify({
        "items": results,
        "errors": errors,
        "counts": {"total": len(results), "fake": fake, "real": len(results) - fake},
    })


@app.route("/api/stats")
def api_stats():
    """Small JSON endpoint: model results + usage statistics."""
    return jsonify({"model": BUNDLE["best_name"], "results": BUNDLE["results"], "usage": database.stats()})


if __name__ == "__main__":
    app.run(debug=False)