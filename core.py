"""
core.py - shared logic: data loading, text cleaning, prediction, explanation
and writing-style analysis. Used by train.py and app.py.
"""

import os
import re

import pandas as pd

DATA_DIR = "data"
REAL_FILE = os.path.join(DATA_DIR, "True.csv")
FAKE_FILE = os.path.join(DATA_DIR, "Fake.csv")

# ------------------------------------------------------------------ demo data
# Tiny fallback so the project always runs. Results on it are NOT meaningful.
SAMPLE_REAL = [
    "The central bank held interest rates steady on Wednesday, citing stable inflation figures.",
    "Parliament passed the annual budget after two days of debate, officials said.",
    "Scientists at the national institute published a study on rising sea levels along the coast.",
    "The health ministry reported a decline in seasonal flu cases compared with last month.",
    "The city council approved funding for a new bridge, with construction to begin next spring.",
    "Company reported quarterly revenue of 2.1 billion, in line with analyst expectations.",
    "The education department announced revised exam dates for secondary school students.",
    "Officials confirmed that the railway line will reopen after maintenance work is completed.",
    "The court adjourned the hearing until next week and asked both sides to submit documents.",
    "Researchers said the new battery design could improve electric vehicle range by ten percent.",
    "The foreign minister met with regional counterparts to discuss trade cooperation.",
    "Police said the road accident occurred early Tuesday and traffic was diverted for two hours.",
    "The stock index closed slightly higher as technology shares recovered from earlier losses.",
    "The election commission released the schedule for the upcoming state elections.",
    "A local hospital opened a new cancer treatment wing, according to a statement from administrators.",
    "Meteorologists forecast heavy rainfall over the weekend and advised residents to stay alert.",
    "The university announced a scholarship program for students from rural areas.",
    "Government data showed unemployment fell to its lowest level in three years.",
    "The airline said flights would resume after the airport cleared the runway of debris.",
    "Engineers completed the inspection of the dam and reported no structural damage.",
]
SAMPLE_FAKE = [
    "SHOCKING: Doctors HATE this one weird trick that cures every disease overnight!!!",
    "You won't BELIEVE what the government is hiding from you about the moon, share before it's deleted!",
    "Celebrity secretly replaced by a clone, insiders reveal the TRUTH the media won't tell you!",
    "Drinking hot water with lemon makes you immune to all viruses, scientists are silent!",
    "BREAKING: Aliens landed in a small village and the army is covering it up, leaked photos inside!",
    "Miracle pill melts fat in 3 days with no diet, banned by big pharma!!!",
    "Forward this message to 10 people or your phone will explode tonight, experts warn!",
    "Secret document proves the election was decided by a hidden group controlling everything!",
    "New law will make all bank savings disappear tomorrow, withdraw your money NOW!",
    "Famous actor dead in tragic accident, hoax or truth? Click to see the horrifying video!",
    "5G towers are secretly mind control devices, whistleblower exposes the shocking plan!",
    "Eating this common fruit at night will kill you in your sleep, doctors afraid to say!",
    "Free iPhone giveaway! Just share this post and enter your bank details to win!",
    "Scientists admit the earth is flat and space agencies have been lying for decades!",
    "Politician caught on secret tape planning to sell the country, media refuses to show it!",
    "Ancient remedy cures cancer in a week but the pharmaceutical industry is hiding it from you!",
    "URGENT: WhatsApp will start charging money from tomorrow, send this to everyone you know!",
    "Man lives 200 years thanks to secret potion, government tried to silence him!",
    "The moon landing was filmed in a studio and a new leaked video finally proves it!!!",
    "Vaccines contain tracking chips, hospital insider reveals shocking truth in viral post!",
]


# ------------------------------------------------------------- preprocessing
def strip_source_tags(text):
    """Remove the 'CITY (Reuters) -' dateline that appears at the start of real
    articles in the Kaggle dataset. Left in, the model just learns this tag
    instead of learning anything about the language of fake news."""
    return re.sub(r"^.{0,60}?\(reuters\)\s*-?\s*", " ", str(text), flags=re.IGNORECASE)


def clean_text(text):
    """Lowercase, remove links, publisher names, punctuation and extra spaces."""
    text = str(text).lower()
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"\breuters\b", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def load_data():
    """Return (DataFrame[text, label], is_demo). label: 1 = real, 0 = fake."""
    if os.path.exists(REAL_FILE) and os.path.exists(FAKE_FILE):
        real = pd.read_csv(REAL_FILE)
        fake = pd.read_csv(FAKE_FILE)
        real["text"] = real["text"].apply(strip_source_tags)  # before joining the title
        for df in (real, fake):
            df["text"] = df["title"].fillna("").astype(str) + " " + df["text"].fillna("").astype(str)
        real["label"], fake["label"] = 1, 0
        df = pd.concat([real[["text", "label"]], fake[["text", "label"]]])
        df = df.drop_duplicates(subset="text")
        demo = False
    else:
        df = pd.DataFrame(
            {
                "text": SAMPLE_REAL + SAMPLE_FAKE,
                "label": [1] * len(SAMPLE_REAL) + [0] * len(SAMPLE_FAKE),
            }
        )
        demo = True
    df["text"] = df["text"].apply(clean_text)
    df = df[df["text"].str.len() > 0]
    return df.sample(frac=1, random_state=42).reset_index(drop=True), demo


# ---------------------------------------------------------------- prediction
def predict(text, vectorizer, model):
    """Return (label, confidence 0-1, probability of real)."""
    vec = vectorizer.transform([clean_text(text)])
    p_real = float(model.predict_proba(vec)[0][1])
    label = "REAL" if p_real >= 0.5 else "FAKE"
    return label, (p_real if label == "REAL" else 1 - p_real), p_real


def explain(text, vectorizer, linear_model, n=6):
    """Words (and word pairs) in the text that pushed toward FAKE and toward REAL,
    using the Logistic Regression coefficients."""
    vec = vectorizer.transform([clean_text(text)])
    names = vectorizer.get_feature_names_out()
    scores = {names[i]: float(v * linear_model.coef_[0][i]) for i, v in zip(vec.indices, vec.data)}
    ranked = sorted(scores.items(), key=lambda kv: kv[1])
    fake_words = [w for w, s in ranked if s < 0][:n]
    real_words = [w for w, s in reversed(ranked) if s > 0][:n]
    return fake_words, real_words


# ------------------------------------------------------------- style signals
CLICKBAIT_TERMS = [
    "shocking", "you won't believe", "secret", "exposed", "hoax", "miracle", "banned",
    "urgent", "leaked", "doctors hate", "share this", "forward this", "viral",
    "they don't want you", "mind control", "cover up", "cover-up", "wake up",
]


def writing_signals(text):
    """Simple rule-based writing-style checks, independent of the ML model."""
    words = re.findall(r"[A-Za-z']+", text)
    n = max(len(words), 1)
    caps = [w for w in words if len(w) > 1 and w.isupper()]
    lower = text.lower()
    bait = [t for t in CLICKBAIT_TERMS if t in lower]
    signals = {
        "words": len(words),
        "exclamations": text.count("!"),
        "questions": text.count("?"),
        "caps_percent": round(100 * len(caps) / n, 1),
        "avg_word_length": round(sum(len(w) for w in words) / n, 1),
        "clickbait_terms": len(bait),
    }
    flags = []
    if signals["exclamations"] >= 2:
        flags.append("Heavy use of exclamation marks")
    if signals["caps_percent"] >= 15:
        flags.append("Many words written in ALL CAPS")
    if bait:
        flags.append("Sensational wording: " + ", ".join(bait[:4]))
    if not flags:
        flags.append("No sensational writing patterns found")
    return signals, flags