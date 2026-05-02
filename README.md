# Phishing Email Detector 🎣🚫

> **A machine learning–powered web application that classifies email content as phishing or legitimate**, built with a Multinomial Naive Bayes classifier, TF-IDF feature extraction, and a Flask web interface.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Flask](https://img.shields.io/badge/Flask-2.x-lightgrey)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange)
![License](https://img.shields.io/badge/license-MIT-green)

---

## Table of Contents

- [Overview](#overview)
- [How It Works](#how-it-works)
- [Project Structure](#project-structure)
- [Tech Stack](#tech-stack)
- [Dataset](#dataset)
- [ML Pipeline](#ml-pipeline)
  - [TF-IDF Vectorization](#tf-idf-vectorization)
  - [Multinomial Naive Bayes Classifier](#multinomial-naive-bayes-classifier)
  - [Training & Evaluation](#training--evaluation)
- [Flask Application](#flask-application)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Retrain the Model](#retrain-the-model)
  - [Run the App](#run-the-app)
- [API Reference](#api-reference)
- [File Reference](#file-reference)
- [Extending the Project](#extending-the-project)
- [Limitations & Known Issues](#limitations--known-issues)

---

## Overview

This project demonstrates an end-to-end NLP classification pipeline applied to cybersecurity. It takes raw email text as input, transforms it into numerical features using TF-IDF, and classifies it using a trained Multinomial Naive Bayes model. The result — **"Phishing Email Detected"** or **"Legitimate Email"** — is displayed instantly on a browser-based interface.

The project covers the complete ML lifecycle: data preparation → feature engineering → model training → serialization → Flask deployment.

---

## How It Works

```
User pastes email text
        │
        ▼
 Flask /predict route
        │
        ▼
 TfidfVectorizer.transform()
  (same vocabulary learned during training)
        │
        ▼
 MultinomialNB.predict()
        │
        ▼
  prediction[0] == 1  →  "Phishing Email Detected"
  prediction[0] == 0  →  "Legitimate Email"
        │
        ▼
  Result rendered in index.html
```

The vectorizer and model are loaded **once** at app startup from pre-serialized `.pkl` files, keeping prediction latency minimal (pure in-memory inference with no re-loading per request).

---

## Project Structure

```
phishing-email-detector/
│
├── app.py                        # Flask application: routes, model loading, prediction logic
├── train_model.py                # Standalone training script: data → vectorizer → model → .pkl
├── requirements.txt              # Python dependencies
│
├── dataset/
│   └── phishing_emails.csv       # Labeled dataset: 50 samples (20 phishing, 30 legitimate)
│
├── model/
│   ├── model.pkl                 # Serialized trained MultinomialNB model
│   └── vectorizer.pkl            # Serialized fitted TfidfVectorizer
│
├── templates/
│   └── index.html                # Jinja2 HTML template: textarea input + result display
│
└── static/
    └── style.css                 # Basic CSS: centered layout, textarea and button styling
```

---

## Tech Stack

| Component | Library / Tool | Purpose |
|---|---|---|
| Web Framework | Flask | HTTP routing, template rendering |
| ML Model | scikit-learn `MultinomialNB` | Probabilistic text classification |
| Feature Extraction | scikit-learn `TfidfVectorizer` | Text → numerical TF-IDF feature matrix |
| Data Handling | pandas | CSV loading and DataFrame manipulation |
| Model Persistence | Python `pickle` | Serialize/deserialize model and vectorizer |
| Templating | Jinja2 (via Flask) | Dynamic HTML result injection |
| Numerical | numpy | Underlying array operations (via scikit-learn) |

---

## Dataset

**File:** `dataset/phishing_emails.csv`

| Property | Value |
|---|---|
| Total samples | 50 |
| Phishing (label = 1) | 20 |
| Legitimate (label = 0) | 30 |
| Columns | `email_text`, `label` |
| Format | CSV, UTF-8 |

### Schema

```
email_text                                              label
------------------------------------------------------  -----
Verify your account immediately to avoid suspension     1
Team meeting scheduled tomorrow at 10 AM                0
```

### Label Encoding

| Label | Class | Meaning |
|---|---|---|
| `1` | Phishing | Malicious email attempting credential theft, urgency manipulation, or deception |
| `0` | Legitimate | Normal workplace or personal communication |

### Phishing Sample Patterns
Phishing samples share common social engineering signals:
- Urgency keywords: `immediately`, `urgent`, `now`, `required`
- Account threat language: `locked`, `suspended`, `deleted`, `flagged`
- Action prompts: `verify`, `confirm`, `update`, `click`
- Brand impersonation: PayPal, Netflix, Apple, bank references

### Legitimate Sample Patterns
Legitimate samples reflect typical workplace communication:
- Meeting and scheduling references: `meeting`, `scheduled`, `tomorrow`
- Document and report sharing: `report`, `attached`, `uploaded`, `shared`
- Team coordination: `reminder`, `review`, `deadline`, `project`

> **Note:** The dataset contains 50 synthetic/curated samples for demonstration purposes. For production-grade classification, a significantly larger and more diverse dataset is required (see [Limitations](#limitations--known-issues)).

---

## ML Pipeline

### TF-IDF Vectorization

**File:** `train_model.py` | **Class:** `sklearn.feature_extraction.text.TfidfVectorizer`

TF-IDF (Term Frequency–Inverse Document Frequency) converts raw email text into a numerical feature matrix where each dimension represents a term's weighted importance.

**Configuration used:**

| Parameter | Value | Effect |
|---|---|---|
| `stop_words` | `"english"` | Removes common English words (the, is, are, etc.) that carry no discriminative signal |
| `ngram_range` | `(1, 1)` | Unigrams only — each individual word is a feature |
| `norm` | `"l2"` | L2-normalizes each document vector so document length doesn't skew results |
| `smooth_idf` | `True` | Adds 1 to document frequencies to avoid division by zero on unseen terms |
| `analyzer` | `"word"` | Tokenizes by word boundaries |
| `token_pattern` | `\b\w\w+\b` | Matches tokens of 2+ alphanumeric characters |
| `lowercase` | `True` | Converts all text to lowercase before tokenization |
| `max_features` | `None` | Keeps all terms in vocabulary |

**Resulting vocabulary size: 125 terms** (from 50 training samples).

**Formula:**

```
TF-IDF(t, d) = TF(t, d) × IDF(t)

TF(t, d)  = count of term t in document d / total terms in d
IDF(t)    = log((1 + N) / (1 + df(t))) + 1    [smooth_idf=True variant]

where N = number of documents, df(t) = documents containing term t
```

> **Important:** The vectorizer is **fit on training data only** and saved to `vectorizer.pkl`. During prediction, only `.transform()` is called — never `.fit_transform()` — ensuring no data leakage and consistent feature dimensions at inference time.

---

### Multinomial Naive Bayes Classifier

**Class:** `sklearn.naive_bayes.MultinomialNB`

Multinomial Naive Bayes is a probabilistic classifier well-suited to text classification tasks. It models the likelihood of each term occurring in each class and applies Bayes' theorem to compute posterior class probabilities.

**Configuration used:**

| Parameter | Value | Effect |
|---|---|---|
| `alpha` | `1.0` | Laplace (additive) smoothing — prevents zero-probability for unseen terms |
| `fit_prior` | `True` | Learns class prior probabilities from training data |
| `class_prior` | `None` | Priors are learned from the data (not manually set) |

**Classification rule:**

```
P(class | email) ∝ P(class) × ∏ P(term_i | class)^tf-idf_i

Predicted class = argmax over {0, 1} of P(class | email)
```

**Why Naive Bayes for this task?**
- Works well on small datasets where deep learning would overfit
- Fast training and inference (linear in vocabulary size)
- Naturally handles high-dimensional sparse text features
- Interpretable: you can inspect per-term log-probabilities per class
- Established baseline for spam/phishing detection tasks

---

### Training & Evaluation

**Script:** `train_model.py`

```python
# 80/20 split with fixed random seed for reproducibility
X_train, X_test, y_train, y_test = train_test_split(
    X_vectorized, y, test_size=0.2, random_state=42
)
# → ~40 training samples, ~10 test samples
```

**Training split:**
- Training set: ~40 samples (80%)
- Test set: ~10 samples (20%)
- `random_state=42` ensures reproducible splits across runs

**Evaluation metric:** `accuracy_score` from `sklearn.metrics`

After training, both artifacts are serialized:
```python
pickle.dump(model,      open("model/model.pkl",      "wb"))
pickle.dump(vectorizer, open("model/vectorizer.pkl", "wb"))
```

---

## Flask Application

**File:** `app.py`

### Startup
At module import time (before any request is served), the model and vectorizer are loaded into memory:
```python
model      = pickle.load(open("model/model.pkl",      "rb"))
vectorizer = pickle.load(open("model/vectorizer.pkl", "rb"))
```
This means classification is pure in-memory — no disk I/O per prediction.

### Routes

| Route | Method | Description |
|---|---|---|
| `/` | `GET` | Renders the input form (`index.html` with no prediction) |
| `/predict` | `POST` | Reads `email` form field, vectorizes, predicts, renders result |

### Prediction Logic

```python
email_text   = request.form["email"]               # raw text from textarea
email_vector = vectorizer.transform([email_text])  # shape: (1, 125)
prediction   = model.predict(email_vector)          # array([0]) or array([1])

result = "Phishing Email Detected" if prediction[0] == 1 else "Legitimate Email"
```

The result string is injected into the Jinja2 template via `prediction_text`:
```html
<h2>{{ prediction_text }}</h2>
```

### Debug Mode
The app runs with `debug=True` by default — **disable this before any production or shared deployment**:
```python
app.run(debug=False, host="0.0.0.0", port=5000)
```

---

## Getting Started

### Prerequisites

- Python 3.8 or higher
- `pip` package manager
- (Optional) `virtualenv` or `conda` for environment isolation

### Installation

```bash
# 1. Clone or unzip the project
cd phishing-email-detector

# 2. Create and activate a virtual environment (recommended)
python -m venv venv

# Linux/macOS
source venv/bin/activate

# Windows
venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

### Retrain the Model

The pre-trained `.pkl` files are already included in the `model/` directory. You only need to run this if you've modified the dataset or want to retrain from scratch:

```bash
python train_model.py
```

Expected output:
```
Model Accuracy: 1.0        ← (will vary with dataset changes)
Model saved successfully
```

> Both `model/model.pkl` and `model/vectorizer.pkl` will be overwritten.

### Run the App

```bash
python app.py
```

Open your browser and navigate to:
```
http://127.0.0.1:5000
```

Paste any email text into the textarea and click **Check Email** to get a prediction.

---

## API Reference

### `POST /predict`

Classifies submitted email text.

**Request:**
- Content-Type: `application/x-www-form-urlencoded`
- Body field: `email` — the raw email body text to classify

**Example (curl):**
```bash
curl -X POST http://127.0.0.1:5000/predict \
  -d "email=Verify your account immediately to avoid suspension"
```

**Response:** HTML page (`index.html`) rendered with `prediction_text` set to one of:
- `"Phishing Email Detected"` — model output `1`
- `"Legitimate Email"` — model output `0`

---

## File Reference

### `train_model.py`

| Step | Code | Description |
|---|---|---|
| Load data | `pd.read_csv(...)` | Reads the CSV into a DataFrame |
| Define features/target | `X = data["email_text"]`, `y = data["label"]` | Separates text and labels |
| Vectorize | `TfidfVectorizer(stop_words="english").fit_transform(X)` | Builds vocab and transforms corpus |
| Split | `train_test_split(..., test_size=0.2, random_state=42)` | 80/20 reproducible split |
| Train | `MultinomialNB().fit(X_train, y_train)` | Fits the classifier |
| Evaluate | `accuracy_score(y_test, predictions)` | Prints test accuracy |
| Save | `pickle.dump(...)` | Writes model and vectorizer to `model/` |

### `app.py`

| Component | Description |
|---|---|
| `model` | Global `MultinomialNB` instance loaded from `model/model.pkl` |
| `vectorizer` | Global `TfidfVectorizer` instance loaded from `model/vectorizer.pkl` |
| `home()` | GET `/` — serves the empty input form |
| `predict()` | POST `/predict` — reads form, vectorizes input, returns classification result |

### `dataset/phishing_emails.csv`

| Column | Type | Description |
|---|---|---|
| `email_text` | `str` | Raw email body text (plain text, no HTML) |
| `label` | `int` | `1` = phishing, `0` = legitimate |

### `model/model.pkl`

Serialized `MultinomialNB` object. Contains learned log-probabilities for each term per class and class priors. Loaded with `pickle.load()`.

### `model/vectorizer.pkl`

Serialized `TfidfVectorizer` object. Contains the 125-term vocabulary and fitted IDF weights. **Must be used only for `.transform()` at inference time** — never re-fit on new input.

---

## Extending the Project

### 1. Expand the Dataset
The current 50-sample dataset is intentionally minimal. For better generalization:
- Use real phishing datasets: [CEAS 2008](http://www.ceas.cc/2008/), [SpamAssassin Public Corpus](https://spamassassin.apache.org/old/publiccorpus/), or Nazario Phishing Corpus
- Aim for at least 1,000–10,000 balanced samples
- Include full email headers, HTML bodies, and URL patterns for richer features

### 2. Try More Powerful Models
Swap out `MultinomialNB` in `train_model.py`:

```python
# Logistic Regression — often strong on TF-IDF features
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(max_iter=1000)

# SVM (linear kernel excels at text classification)
from sklearn.svm import LinearSVC
model = LinearSVC()

# Random Forest
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=100)
```

### 3. Add Bigrams
Capture two-word phishing cues like "click here" or "verify now":
```python
vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
```

### 4. Return a Confidence Score
`MultinomialNB` supports probability outputs:

```python
# In app.py
proba = model.predict_proba(email_vector)[0]
confidence = round(max(proba) * 100, 2)
# Render: "Phishing Email Detected (94.3% confidence)"
```

### 5. Add Input Validation
```python
@app.route("/predict", methods=["POST"])
def predict():
    email_text = request.form.get("email", "").strip()
    if not email_text:
        return render_template("index.html", prediction_text="Please enter email text.")
    if len(email_text) > 10000:
        return render_template("index.html", prediction_text="Input too long (max 10,000 chars).")
    ...
```

### 6. Expose a JSON API Endpoint
```python
@app.route("/api/predict", methods=["POST"])
def api_predict():
    data = request.get_json()
    email_vector = vectorizer.transform([data["email"]])
    prediction = model.predict(email_vector)
    proba = model.predict_proba(email_vector)[0]
    return {
        "label": int(prediction[0]),
        "result": "phishing" if prediction[0] == 1 else "legitimate",
        "confidence": round(float(max(proba)) * 100, 2)
    }
```

### 7. Add URL Feature Engineering
A major phishing signal is suspicious URLs. Extract features before vectorizing:
- Presence of IP addresses in URLs instead of domain names
- Use of URL shorteners (bit.ly, tinyurl)
- Mismatch between anchor text and actual `href`
- Excessive subdomains or hyphenated domains

---

## Limitations & Known Issues

| Limitation | Details |
|---|---|
| **Small dataset** | 50 samples is insufficient for real-world generalization. The model is likely memorizing patterns rather than learning robust features. |
| **Synthetic samples** | The dataset consists of curated short phrases, not actual email bodies. Real phishing emails are longer, contain HTML, and use more varied language. |
| **No URL analysis** | Malicious or spoofed URLs are a primary phishing signal entirely absent from the current feature set. |
| **Unigrams only** | Single words miss multi-word cues like "click here" or "verify now" that are strong phishing indicators. |
| **No email header analysis** | SPF, DKIM, DMARC, and From/Reply-To spoofing are completely unanalyzed. |
| **Debug mode enabled** | `app.run(debug=True)` exposes the Werkzeug interactive debugger. Must be disabled before any deployment. |
| **No input validation** | `request.form["email"]` raises `400 Bad Request` if the field is missing. Use `.get()` with a fallback. |
| **Pickle security risk** | `.pkl` files should never be loaded from untrusted sources — pickle deserialization can execute arbitrary code. |
| **`nltk` unused** | `nltk` is listed in `requirements.txt` but is not imported or used anywhere in the codebase. It can be safely removed. |

---

## Requirements

```
pandas
numpy
scikit-learn
flask
nltk        ← listed but not used in current code; safe to remove
```

Install all dependencies:
```bash
pip install -r requirements.txt
```

---

<div align="center">

Built as a cybersecurity ML demonstration · Python · scikit-learn · Flask

</div>
