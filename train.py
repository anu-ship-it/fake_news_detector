"""
train.py - trains and compares four models, picks the best one by cross-validation,
saves it (model.pkl) and generates evaluation charts (static/charts/).

Run:  python train.py        (app.py also calls this automatically on first start)
"""

import time
from datetime import datetime

import joblib
import matplotlib

matplotlib.use("Agg")  # no display needed
import matplotlib.pyplot as plt
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

from core import load_data

MODEL_FILE = "model.pkl"
CHART_DIR = "static/charts"

BG, PANEL, TEXT, MUTED = "#0b1020", "#141b34", "#e8ecf8", "#8b97ba"
COLORS = ["#6c8cff", "#2ecc8f", "#ffb84d", "#ff5c7a"]


def _style(fig, ax):
    fig.patch.set_facecolor(PANEL)
    ax.set_facecolor(BG)
    ax.tick_params(colors=MUTED)
    ax.title.set_color(TEXT)
    ax.xaxis.label.set_color(MUTED)
    ax.yaxis.label.set_color(MUTED)
    for s in ax.spines.values():
        s.set_color("#243056")


def _save(fig, name):
    fig.tight_layout()
    fig.savefig(f"{CHART_DIR}/{name}", dpi=110, facecolor=fig.get_facecolor())
    plt.close(fig)


def make_charts(results, y_test, probas, best_name, vectorizer, lr):
    import os
    os.makedirs(CHART_DIR, exist_ok=True)

    # 1) model comparison
    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(results))
    ax.bar(x - 0.2, [r["cv_accuracy"] for r in results], 0.4, label="Cross-val accuracy", color=COLORS[0])
    ax.bar(x + 0.2, [r["accuracy"] for r in results], 0.4, label="Test accuracy", color=COLORS[1])
    ax.set_xticks(x)
    ax.set_xticklabels([r["name"] for r in results], fontsize=8)
    ax.set_ylim(0, 105)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Model comparison")
    ax.legend(facecolor=PANEL, labelcolor=TEXT, edgecolor="#243056", fontsize=8)
    _style(fig, ax)
    _save(fig, "model_comparison.png")

    # 2) confusion matrix of best model
    pred = (probas[best_name] >= 0.5).astype(int)
    cm = confusion_matrix(y_test, pred)
    fig, ax = plt.subplots(figsize=(4.6, 4))
    ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=16, fontweight="bold",
                    color="white" if cm[i, j] > cm.max() / 2 else "#0b1020")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Fake", "Real"])
    ax.set_yticklabels(["Fake", "Real"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion matrix - {best_name}")
    _style(fig, ax)
    _save(fig, "confusion_matrix.png")

    # 3) ROC curves
    fig, ax = plt.subplots(figsize=(5.4, 4))
    for (name, p), c in zip(probas.items(), COLORS):
        fpr, tpr, _ = roc_curve(y_test, p)
        ax.plot(fpr, tpr, color=c, label=name, linewidth=2)
    ax.plot([0, 1], [0, 1], "--", color=MUTED, linewidth=1)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC curves")
    ax.legend(facecolor=PANEL, labelcolor=TEXT, edgecolor="#243056", fontsize=7, loc="lower right")
    _style(fig, ax)
    _save(fig, "roc_curve.png")

    # 4) most influential words (Logistic Regression coefficients)
    names = vectorizer.get_feature_names_out()
    order = lr.coef_[0].argsort()
    fake_idx, real_idx = order[:12], order[::-1][:12]
    idx = list(fake_idx[::-1]) + list(real_idx[::-1])
    vals = [lr.coef_[0][i] for i in idx]
    fig, ax = plt.subplots(figsize=(7, 5.4))
    ax.barh([names[i] for i in idx], vals, color=["#ff5c7a" if v < 0 else "#2ecc8f" for v in vals])
    ax.set_title("Top words: red = fake, green = real")
    ax.set_xlabel("Logistic Regression coefficient")
    ax.tick_params(axis="y", labelsize=8)
    _style(fig, ax)
    _save(fig, "top_words.png")


def run_training():
    df, demo = load_data()
    print(f"Dataset: {len(df)} articles ({'DEMO sample data' if demo else 'Kaggle Fake/Real News'})")

    X_train, X_test, y_train, y_test = train_test_split(
        df["text"], df["label"], test_size=0.2, random_state=42, stratify=df["label"]
    )
    vectorizer = TfidfVectorizer(stop_words="english", max_features=20000, ngram_range=(1, 2))
    Xtr = vectorizer.fit_transform(X_train)
    Xte = vectorizer.transform(X_test)

    candidates = {
        "Naive Bayes": MultinomialNB(),
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Linear SVM": CalibratedClassifierCV(LinearSVC(), cv=3),
        "SGD Classifier": SGDClassifier(loss="modified_huber", max_iter=1000, random_state=42),
    }

    results, probas, fitted = [], {}, {}
    for name, model in candidates.items():
        t0 = time.time()
        cv = cross_val_score(model, Xtr, y_train, cv=3, scoring="accuracy")
        model.fit(Xtr, y_train)
        p = model.predict_proba(Xte)[:, 1]
        pred = (p >= 0.5).astype(int)
        results.append({
            "name": name,
            "cv_accuracy": round(cv.mean() * 100, 2),
            "accuracy": round(accuracy_score(y_test, pred) * 100, 2),
            "precision": round(precision_score(y_test, pred, zero_division=0) * 100, 2),
            "recall": round(recall_score(y_test, pred, zero_division=0) * 100, 2),
            "f1": round(f1_score(y_test, pred, zero_division=0) * 100, 2),
            "auc": round(roc_auc_score(y_test, p), 4),
            "seconds": round(time.time() - t0, 1),
        })
        probas[name], fitted[name] = p, model
        print(f"  {name:<20} CV {results[-1]['cv_accuracy']:>6}%  Test {results[-1]['accuracy']:>6}%  ({results[-1]['seconds']}s)")

    # choose on cross-validation (training data only), NOT on the test set
    best = max(results, key=lambda r: r["cv_accuracy"])
    best_name = best["name"]
    print(f"Best model (by cross-validation): {best_name}")

    make_charts(results, y_test.values, probas, best_name, vectorizer, fitted["Logistic Regression"])

    bundle = {
        "vectorizer": vectorizer,
        "models": fitted,
        "best_name": best_name,
        "explainer": fitted["Logistic Regression"],
        "results": results,
        "samples": len(df),
        "train_size": len(X_train),
        "test_size": len(X_test),
        "demo": demo,
        "trained_at": datetime.now().strftime("%d %b %Y, %H:%M"),
    }
    joblib.dump(bundle, MODEL_FILE)
    print(f"Saved {MODEL_FILE} and charts in {CHART_DIR}/")
    return bundle


if __name__ == "__main__":
    run_training()