# TruthLens - AI Fake News Detection System

A machine-learning web application that classifies news articles as **REAL** or **FAKE**,
compares four models, and explains each verdict.

## Features
- **Trending News page:** pulls today's headlines from live RSS feeds and screens each one
- Paste text **or a URL** (article is scraped automatically)
- 4 models compared (Naive Bayes, Logistic Regression, Linear SVM, SGD) - best chosen by cross-validation
- Model-agreement table, confidence bar, influential-word explanation
- Rule-based writing-style signals (ALL-CAPS, exclamation marks, clickbait phrases)
- Model Insights page: metrics table, confusion matrix, ROC curves, top-word chart
- Analysis history stored in SQLite; JSON endpoint `/api/stats`

## Tech stack
Python | scikit-learn (TF-IDF, models, cross-validation) | pandas | NumPy | Matplotlib |
Flask + Jinja2 | SQLite | joblib | requests + BeautifulSoup | RSS parsing (xml.etree)

## Architecture
```
data/True.csv, Fake.csv --> core.py (clean + load) --> train.py (TF-IDF, 4 models, CV, charts)
                                                           |
                                                       model.pkl
                                                           |
browser <--> app.py (Flask routes) <--> core.py (predict, explain, style signals)
                          |
                     database.py (SQLite history.db)
```

## Run
```
pip install -r requirements.txt
# put True.csv and Fake.csv (Kaggle: "Fake and real news dataset") inside data/
python app.py
```
Open http://127.0.0.1:5000. The first start trains the models (a few minutes on the full
dataset) and saves `model.pkl`. To retrain on new data, delete `model.pkl` and restart.
`python train.py` trains without starting the web app.

## Files
| File | Purpose |
|---|---|
| core.py | data loading, cleaning, prediction, explanation, style signals |
| train.py | model training, comparison, evaluation charts |
| news_feed.py | downloads trending headlines from RSS feeds (edit `FEEDS` to change sources) |
| database.py | SQLite history |
| app.py | Flask routes and JSON API |
| templates/, static/ | UI pages and styling |

## Limitations
The model learns writing style, not factual truth. It cannot verify claims, and a well-written
false article may be classified as real. The dataset is mostly US political news from 2016-2018,
so accuracy on other topics or time periods will be lower.