# 🛒 Predicting Amazon Product Sales via Sentiment Analysis

> **MSc Thesis — Data Science, AI & Digital Business**
> GISMA University of Applied Sciences, Potsdam | 2024–2025

---

## 📌 Project Overview

This project addresses a real-world business problem: **how can sellers forecast product demand without historical sales data?**

By leveraging **customer review sentiment**, **Word2Vec embeddings**, and **product metadata** from Amazon, this thesis builds an end-to-end ML pipeline to predict a custom **Sales Proxy** metric (number of ratings) — a reliable stand-in for actual sales volume.

---

## 🗂️ Repository Contents

| File | Description |
|------|-------------|
| Thesis_GH1044613.ipynb | Full analysis notebook — data loading, EDA, NLP, modelling |
| Thesiss_Final HTML File.html | Exported HTML version of the complete notebook |
| Dataset Link.pdf | Links to the Amazon review & metadata datasets (JSONL) |

---

## 🔄 Pipeline Architecture

1. Load Amazon Reviews JSONL + Product Metadata JSONL
2. Data Cleaning — drop image columns, remove duplicates, handle nulls
3. Sentiment Analysis (NLTK VADER) — compound score per review
4. Word2Vec Embeddings (Gensim) — vector_size=100, window=5, average per review
5. Aggregate to Product Level — avg_sentiment, sentiment_std, num_reviews, avg_review_rating
6. Merge with Metadata — average_rating, main_category, store, Sales Proxy
7. Feature Engineering — LabelEncoder for categories, log1p transform on target
8. EDA — correlation heatmap, scatter/box plots, category & store analysis
9. Regression Models — Linear Regression, Random Forest, XGBoost
10. Classification Models — Logistic Regression, Random Forest, XGBoost (for comparison)

---

## 📊 Exploratory Data Analysis

Key findings from EDA:

- **Sentiment vs Sales:** Negative sentiment products never achieve high sales. High sales require at minimum moderate–high sentiment.
- **Rating vs Sales:** Excellent rated products (4–5 stars) consistently show higher sales proxy values.
- **Review Volume:** Products with more reviews tend to have higher sales, even with average sentiment scores.
- **Category Trends:** Sales proxy varies significantly across Amazon product categories.
- **Correlation:** num_reviews shows the strongest correlation with the sales proxy target.

---

## 🤖 Models & Results

### Regression (Predicting Sales Volume)

| Model | RMSE | MAE | R² |
|-------|------|-----|-----|
| Baseline (Mean Predictor) | 1.156 | 0.913 | -0.0003 |
| Sentiment-Only Linear Regression | 1.018 | 0.797 | 0.224 |
| Linear Regression (All Features) | 0.988 | 0.767 | 0.270 |
| Random Forest (200 trees) | 0.801 | 0.598 | 0.520 |
| **XGBoost (300 trees)** | **0.773** | **0.569** | **0.553** |

XGBoost is the best-performing model with R² = 0.55, explaining 55% of variance in sales.

### Classification (Low / Medium / High Sales Buckets)

| Model | Accuracy |
|-------|----------|
| Baseline (Majority Class) | ~33% |
| Logistic Regression | ~45% |
| Random Forest | ~55% |
| XGBoost Classifier | ~57% |

Regression outperforms classification — continuous prediction captures more nuance than 3-class bucketing.

---

## 🔑 Key Features Used

- main_category — Product category (label encoded)
- store — Seller/store name (label encoded)
- average_rating — Overall product star rating
- avg_review_rating — Mean star rating across reviews
- num_reviews — Total number of reviews
- avg_sentiment — Mean VADER sentiment score
- sentiment_std — Sentiment variability across reviews
- w2v_0_mean to w2v_99_mean — 100-dimensional Word2Vec embeddings

Target: log1p(Rating_number) — Sales proxy, log-transformed for normality

---

## 💡 Business Insights

- **High-sentiment products** (avg_sentiment > 0.5) are strong candidates for increased inventory and marketing spend.
- **Review volume** is the most predictive feature — products gaining traction in reviews are likely to convert to sales.
- **Sentiment as a quality gate:** No product with predominantly negative reviews achieved high sales proxy.
- **Category matters:** Certain Amazon categories show structurally higher sales proxy values — useful for new product launches.
- **Actionable for sellers:** This model enables proactive stock and marketing decisions without needing proprietary sales data.

---

## 🛠️ Tech Stack

- Python, Pandas, NumPy — Data manipulation
- Matplotlib, Seaborn — Visualisation
- NLTK (VADER) — Sentiment analysis
- Gensim (Word2Vec) — Text embeddings
- Scikit-learn — Preprocessing, model selection, metrics
- XGBoost — Gradient boosting regression and classification

---

## ▶️ How to Run

```bash
# 1. Clone the repository
git clone https://github.com/ketan3107/Thesis-.git
cd Thesis-

# 2. Install dependencies
pip install pandas numpy matplotlib seaborn nltk gensim scikit-learn xgboost

# 3. Download NLTK VADER lexicon
python -c "import nltk; nltk.download('vader_lexicon')"

# 4. Download datasets (see Dataset Link.pdf) and place in working directory

# 5. Open and run the notebook
jupyter notebook Thesis_GH1044613.ipynb
```

---

## 👤 Author

**Ketan Sharma**
MSc Data Science, AI & Digital Business — GISMA University of Applied Sciences
[LinkedIn](https://www.linkedin.com/in/ketan-sharma-993a0a1b6) | [GitHub](https://github.com/ketan3107)
