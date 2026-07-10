# 📱 M-Pajak App Review Clustering using K-Means

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-K--Means-orange?logo=scikitlearn)
![Status](https://img.shields.io/badge/Status-Completed-success)
![License](https://img.shields.io/badge/License-MIT-green)
![NLP](https://img.shields.io/badge/NLP-TF--IDF-red)
![Unsupervised Learning](https://img.shields.io/badge/Machine%20Learning-Unsupervised-purple)

An end-to-end **Natural Language Processing (NLP)** project that analyzes **Google Play Store reviews** of the **M-Pajak** mobile application (Indonesia's Directorate General of Taxes) to automatically identify major technical complaint categories using **K-Means Clustering**.

---

# 📌 Project Overview

The **M-Pajak** application has been downloaded over **1 million times**, yet it maintains a relatively low user rating (**~2.2/5**) with more than **14,000 reviews**.

Since manually reading thousands of user reviews is inefficient, this project applies **unsupervised machine learning** to automatically group similar complaints into meaningful categories. The insights can help product teams prioritize bug fixes and improve user experience.

---

# 🎯 Objectives

- 📊 Analyze user feedback from Google Play Store
- 🧹 Clean and preprocess Indonesian text data
- 🤖 Cluster technical complaints using K-Means
- 📈 Identify the most frequent user issues
- 🔍 Compare clustering results with a rule-based categorization
- 💡 Generate actionable insights for product improvement

---

# 📂 Dataset

| Information | Value |
|------------|-------|
| Source | Google Play Store |
| Collection Method | `google-play-scraper` |
| Raw Reviews | **8,459** |
| Reviews After Cleaning | **7,552** |
| Language | Indonesian |
| Fields | reviewId, userName, content, score, thumbsUpCount, at, appVersion |

---

# ⚙️ Project Workflow

```text
Google Play Reviews
        │
        ▼
Data Understanding
        │
        ▼
Adjusted Rating Labeling
        │
        ▼
Text Preprocessing
(Lowercase • Cleaning • Stopword Removal • Stemming)
        │
        ▼
Exploratory Data Analysis
        │
        ▼
Negative Review Filtering
(adjusted_score ≤ 3)
        │
        ▼
TF-IDF Vectorization
(Unigram + Bigram)
        │
        ▼
Elbow Method + Silhouette Score
        │
        ▼
K-Means Clustering
        │
        ▼
Cluster Validation
(Confusion Matrix)
        │
        ▼
Cluster Interpretation
```

---

# 🧹 Text Preprocessing

The preprocessing pipeline includes:

- ✅ Lowercase conversion
- ✅ URL & punctuation removal
- ✅ Number removal
- ✅ Indonesian stopword removal (Sastrawi)
- ✅ Custom stopword filtering
- ✅ Indonesian stemming
- ✅ Duplicate removal

---

# 🤖 Machine Learning Pipeline

### Feature Extraction

- TF-IDF Vectorization
- Unigram + Bigram
- `max_df = 0.5`
- Sparse matrix representation

### Model Selection

- 📉 Elbow Method
- 📈 Silhouette Score

### Clustering Algorithm

- **K-Means Clustering**

### Validation

- Confusion Matrix against manually defined keyword categories

---

# 📊 Identified Complaint Categories

| Cluster | Description |
|---------|-------------|
| 🔑 Login Issues | Login failures, password, NIK problems |
| 📩 OTP Verification | Verification code not received or delayed |
| 📝 NPWP Registration | Online NPWP registration issues |
| 💳 Tax Payment & Reporting | Billing code and tax reporting problems |
| 📄 Annual Tax Return (SPT) | Difficulties submitting annual tax reports |
| ⚠️ Application Errors | Crash, bugs, and unexpected errors |
| 💬 General Complaints | Non-specific complaints and poor app quality |

---

# 📈 Key Findings

📌 **86.9%** of reviews have an **adjusted_score ≤ 2**, indicating extremely high user dissatisfaction.

📌 Around **1.3%** of **5-star reviews** actually contain technical complaints, showing that star ratings alone can be misleading.

📌 **Login** and **OTP Verification** clusters achieved the highest consistency when compared with rule-based categorization.

📌 Most complaints are related to authentication and account access rather than tax regulations themselves.

---

# 📁 Repository Structure

```text
.
├── scrapping.py
├── clustering_mpajak.py
├── processed_mpajak_data.csv
├── kesimpulan.txt
├── top_terms_per_cluster.txt
├── elbow_silhouette.png
├── confusion_matrix_cluster_vs_rule.png
├── boxplot_per_cluster.png
├── top20_words.png
├── wordcloud_all.png
├── requirements.txt
└── README.md
```

---

# 🚀 Getting Started

## Clone Repository

```bash
git clone https://github.com/AlAkbar44/M-Pajak-App-Review-Clustering.git
cd M-Pajak-App-Review-Clustering
```

## Install Dependencies

```bash
pip install -r requirements.txt
```

## Run Scraping (Optional)

```bash
python3 scrapping.py
```

## Run Analysis

```bash
python3 clustering_mpajak.py
```

---

# 🛠️ Tech Stack

### Programming

- Python

### Libraries

- pandas
- numpy
- scikit-learn
- matplotlib
- wordcloud
- Sastrawi
- google-play-scraper

### Machine Learning

- TF-IDF Vectorization
- K-Means Clustering
- Elbow Method
- Silhouette Score

---

# 📷 Output Files

The project automatically generates several analysis outputs:

- 📊 Word Frequency Chart
- ☁️ Word Cloud
- 📉 Elbow Method Plot
- 📈 Silhouette Score Plot
- 📦 Boxplot per Cluster
- 🔥 Confusion Matrix
- 📝 Top TF-IDF Terms
- 📄 Summary Report

---

# ⚠️ Limitations

- Silhouette scores are relatively low (~0.02–0.03) because app reviews are generally short, informal, and contain mixed Indonesian-English expressions.

- Cluster naming is manually interpreted using the highest TF-IDF terms and validated against a keyword-based rule system.

- TF-IDF captures lexical similarity but cannot fully understand semantic meaning.

---

# 🚀 Future Improvements

Possible enhancements include:

- ✅ IndoBERT Sentence Embeddings
- ✅ FastText / Word2Vec
- ✅ BERTopic
- ✅ HDBSCAN
- ✅ Topic Modeling (LDA)
- ✅ Interactive Dashboard using Streamlit

---

# 💼 Skills Demonstrated

- Natural Language Processing (NLP)
- Text Mining
- Data Cleaning
- Exploratory Data Analysis
- Feature Engineering
- Unsupervised Machine Learning
- K-Means Clustering
- Model Evaluation
- Data Visualization
- Python Programming

---

# 👨‍💻 Author

**Al Akbar Himawan**

🎓 Digital Business Student

📊 Aspiring Data Analyst | Data Scientist

🔗 GitHub: https://github.com/AlAkbar44

If you find this project useful, don't forget to ⭐ the repository!
