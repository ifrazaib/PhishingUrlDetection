# =====================================================
# AI-BASED PHISHING URL DETECTION SYSTEM
# Streamlit Web App - Optimized for local machine
# =====================================================

import streamlit as st
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
import requests
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from scipy.sparse import hstack, csr_matrix

# =========================
# CONFIGURATION
# =========================
DATASET_URL = "https://raw.githubusercontent.com/faizann24/Phishing-URL-Detection/master/dataset/malicious_phish.csv"
DATASET_PATH = "malicious_phish.csv"
MODEL_DIR = "models"
RF_MODEL_PATH = os.path.join(MODEL_DIR, "random_forest.pkl")
LR_MODEL_PATH = os.path.join(MODEL_DIR, "logistic_regression.pkl")
VECTORIZER_PATH = os.path.join(MODEL_DIR, "vectorizer.pkl")
FEATURE_NAMES_PATH = os.path.join(MODEL_DIR, "feature_names.pkl")

# =========================
# DOWNLOAD DATASET IF MISSING
# =========================
@st.cache_resource
def download_dataset():
    if not os.path.exists(DATASET_PATH):
        with st.spinner("Downloading dataset (~60MB) ..."):
            response = requests.get(DATASET_URL)
            with open(DATASET_PATH, "wb") as f:
                f.write(response.content)
        st.success("Dataset downloaded!")
    else:
        st.info("Dataset already present.")

# =========================
# FEATURE EXTRACTION
# =========================
def extract_features(url):
    features = {}
    features['url_length'] = len(url)
    features['https'] = 1 if 'https' in url else 0
    features['dot_count'] = url.count('.')
    features['hyphen_count'] = url.count('-')
    features['digit_count'] = sum(c.isdigit() for c in url)
    features['slash_count'] = url.count('/')
    features['special_char_count'] = len(re.findall(r'[@_!#$%^&*()<>?/|}{~:]', url))
    suspicious_words = [
        'login', 'verify', 'bank', 'secure', 'account',
        'update', 'free', 'bonus', 'signin', 'payment'
    ]
    features['suspicious_word_count'] = sum(word in url.lower() for word in suspicious_words)
    return features

# =========================
# TRAIN MODELS (SPARSE + SUBSAMPLING)
# =========================
@st.cache_resource
def train_models():
    # Create model directory at the very beginning
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    # Load dataset
    df = pd.read_csv(DATASET_PATH)
    
    # Optional: sample for faster training (adjust as needed)
    SAMPLE_SIZE = 50000
    if len(df) > SAMPLE_SIZE:
        df = df.sample(n=SAMPLE_SIZE, random_state=42).reset_index(drop=True)
        st.info(f"Using {SAMPLE_SIZE} sample rows for faster training.")
    
    # Create binary label
    df['label'] = df['type'].apply(lambda x: 0 if x == 'benign' else 1)
    
    # Extract handcrafted features
    feature_rows = []
    for url in df['url']:
        feature_rows.append(extract_features(url))
    feature_df = pd.DataFrame(feature_rows)
    
    # NLP features - sparse
    vectorizer = CountVectorizer(stop_words='english', max_features=100, lowercase=True)
    X_nlp_sparse = vectorizer.fit_transform(df['url'])
    
    # Combine features (sparse)
    X_dense_sparse = csr_matrix(feature_df.values)
    X = hstack([X_dense_sparse, X_nlp_sparse])
    y = df['label'].values
    
    # Save feature names for later inspection
    manual_names = list(feature_df.columns)
    word_names = vectorizer.get_feature_names_out().tolist()
    all_feature_names = manual_names + word_names
    joblib.dump(all_feature_names, FEATURE_NAMES_PATH)
    
    # Train test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Random Forest
    rf = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1, max_depth=20)
    rf.fit(X_train, y_train)
    rf_acc = accuracy_score(y_test, rf.predict(X_test))
    
    # Logistic Regression
    lr = LogisticRegression(max_iter=500, random_state=42, n_jobs=-1)
    lr.fit(X_train, y_train)
    lr_acc = accuracy_score(y_test, lr.predict(X_test))
    
    # Save models and vectorizer
    joblib.dump(rf, RF_MODEL_PATH)
    joblib.dump(lr, LR_MODEL_PATH)
    joblib.dump(vectorizer, VECTORIZER_PATH)
    
    return rf, lr, vectorizer, rf_acc, lr_acc, all_feature_names

# =========================
# PREDICTION FUNCTION
# =========================
def predict_url(url, model, vectorizer):
    feat = extract_features(url)
    feat_df = pd.DataFrame([feat])
    nlp_sparse = vectorizer.transform([url])
    combined = hstack([csr_matrix(feat_df.values), nlp_sparse])
    pred = model.predict(combined)[0]
    prob = model.predict_proba(combined)[0] if hasattr(model, "predict_proba") else None
    return pred, prob

# =========================
# MAIN STREAMLIT APP
# =========================
def main():
    st.set_page_config(page_title="Phishing URL Detector", layout="wide")
    st.title("🔒 AI-Powered Phishing URL Detection")
    st.markdown("Using **Random Forest** & **Logistic Regression** with URL features + NLP")
    
    # Download dataset if needed
    download_dataset()
    
    # Train or load models
    with st.spinner("Loading / training models (first time may take 1-2 minutes)..."):
        try:
            # Try to load existing artifacts
            rf = joblib.load(RF_MODEL_PATH)
            lr = joblib.load(LR_MODEL_PATH)
            vectorizer = joblib.load(VECTORIZER_PATH)
            feature_names = joblib.load(FEATURE_NAMES_PATH)
            st.success("✅ Models loaded from cache!")
            rf_acc = lr_acc = None
        except:
            st.info("Training models (this happens once)...")
            rf, lr, vectorizer, rf_acc, lr_acc, feature_names = train_models()
            st.success("✅ Training complete! Models saved.")
    
    # Sidebar
    st.sidebar.header("⚙️ Settings")
    model_choice = st.sidebar.selectbox("Select Model", ["Random Forest", "Logistic Regression"])
    
    if rf_acc is not None:
        st.sidebar.subheader("📊 Model Performance")
        st.sidebar.write(f"Random Forest: {rf_acc:.4f}")
        st.sidebar.write(f"Logistic Regression: {lr_acc:.4f}")
    else:
        st.sidebar.info("Performance metrics shown after first training.")
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🔍 How it works")
    st.sidebar.info(
        "Features: URL length, HTTPS, dots, hyphens, digits, slashes, special chars, "
        "suspicious keywords + 100 most common words (TF)."
    )
    
    # URL input area
    col1, col2 = st.columns([3, 1])
    with col1:
        url_input = st.text_input("Enter a URL to analyze:", value="http://verify-bank-login-security-update.com")
    with col2:
        predict_btn = st.button("🔍 Detect", type="primary", use_container_width=True)
    
    if predict_btn and url_input:
        with st.spinner("Analyzing..."):
            try:
                model = rf if model_choice == "Random Forest" else lr
                pred, prob = predict_url(url_input, model, vectorizer)
                
                st.markdown("---")
                col_res1, col_res2 = st.columns(2)
                with col_res1:
                    if pred == 0:
                        st.success("✅ **BENIGN** (Safe)")
                    else:
                        st.error("⚠️ **PHISHING / MALICIOUS** (Dangerous)")
                with col_res2:
                    if prob is not None:
                        confidence = prob[1] if pred == 1 else prob[0]
                        st.metric("Confidence", f"{confidence:.2%}")
                
                # Show extracted features
                with st.expander("📊 View extracted URL features"):
                    feats = extract_features(url_input)
                    feat_df = pd.DataFrame([feats]).T.reset_index()
                    feat_df.columns = ["Feature", "Value"]
                    st.dataframe(feat_df, use_container_width=True)
                
                # Show top NLP tokens
                with st.expander("📝 Top NLP tokens found in URL"):
                    nlp_vec = vectorizer.transform([url_input]).toarray()[0]
                    token_names = vectorizer.get_feature_names_out()
                    tokens_with_val = [(token_names[i], nlp_vec[i]) for i in range(len(nlp_vec)) if nlp_vec[i] > 0]
                    tokens_with_val.sort(key=lambda x: x[1], reverse=True)
                    if tokens_with_val:
                        st.write(pd.DataFrame(tokens_with_val[:10], columns=["Token", "Count"]))
                    else:
                        st.write("No common words from vocabulary detected.")
            
            except Exception as e:
                st.error(f"Prediction error: {e}")
    
    # Feature importance chart
    st.markdown("---")
    st.subheader("📈 Global Feature Importance (Random Forest)")
    if os.path.exists(RF_MODEL_PATH):
        rf_model = joblib.load(RF_MODEL_PATH)
        feature_names = joblib.load(FEATURE_NAMES_PATH)
        importances = rf_model.feature_importances_
        imp_df = pd.DataFrame({"Feature": feature_names, "Importance": importances})
        imp_df = imp_df.sort_values("Importance", ascending=False).head(20)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(data=imp_df, x="Importance", y="Feature", ax=ax, palette="viridis")
        ax.set_title("Top 20 Features (Random Forest)")
        st.pyplot(fig)
    else:
        st.info("Train the model first to see feature importance.")
    
    st.markdown("---")
    st.caption("Dataset: malicious_phish.csv | Model: RandomForest + LogisticRegression | Scikit-learn")

if __name__ == "__main__":
    main()