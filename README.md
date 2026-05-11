# 🔒 AI-Powered Phishing URL Detection System

A machine learning web application that detects phishing, malware, and defacement URLs using **Random Forest** and **Logistic Regression** with a combination of handcrafted URL features and NLP (CountVectorizer).

## 🚀 Features

- **Two models** – Random Forest (best accuracy) and Logistic Regression
- **Feature extraction** – URL length, HTTPS usage, dot/hyphen/digit/slash counts, special characters, suspicious keywords
- **NLP approach** – Top 100 most frequent words from URLs
- **Interactive UI** – Built with Streamlit
- **Real-time predictions** – Enter any URL and get instant classification
- **Confidence scores** – Shows probability of prediction
- **Model performance** – Displays accuracy and feature importance charts
- **Lightweight** – Uses sparse matrices and subsampling to run on modest hardware

## 📊 Dataset

The model is trained on the **malicious_phish.csv** dataset from [Faizan Ahmad's Phishing URL Detection repository](https://github.com/faizann24/Phishing-URL-Detection). It contains over 500,000 labeled URLs across four categories:

- `benign` – safe websites
- `phishing` – fake login/sensitive data stealers
- `malware` – sites hosting malicious software
- `defacement` – hacked sites with altered content

## 🧠 How It Works

1. **Handcrafted features** (8 numeric features) are extracted from each URL.
2. **NLP features** (top 100 words) are extracted using `CountVectorizer` (sparse matrix).
3. Features are combined and fed into **Random Forest** and **Logistic Regression**.
4. The model with better accuracy (Random Forest) is used by default.
5. Streamlit provides a clean interface for real-time predictions.

## 🛠️ Installation & Setup

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Clone the repository

```bash
git clone https://github.com/ifrazaib/PhishingUrlDetection.git
cd PhishingUrlDetection/IS_Project
