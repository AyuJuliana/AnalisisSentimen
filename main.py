import streamlit as st
import pandas as pd
import numpy as np
import re
import string
import joblib
from gensim.models import Word2Vec
from streamlit_option_menu import option_menu
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import matplotlib.pyplot as plt
import seaborn as sns

# =========================================================
# LOAD MODEL
# =========================================================
svm_model = joblib.load("svm_best_model.pkl")
w2v_model = Word2Vec.load("word2vec_embedding (1).model")   # HARUS MODEL YANG DIPAKAI SAAT TRAINING

# =========================================================
# LOAD SLANGWORDS & STOPWORDS
# =========================================================
def load_slangwords(path="slangwords.txt"):
    slang = {}
    try:
        import json
        slang = json.load(open(path, "r", encoding="utf-8"))
    except:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if "=" in line:
                    a, b = line.split("=", 1)
                    slang[a.strip()] = b.strip()
    return slang

def load_stopwords(path="stopwords.txt"):
    return set([
        x.strip()
        for x in open(path, "r", encoding="utf-8").read().splitlines()
        if x.strip() != ""
    ])

slang_dict = load_slangwords()
stopwords = load_stopwords()

# =========================================================
# PREPROCESSING
# =========================================================
factory = StemmerFactory()
stemmer = factory.create_stemmer()

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"[@#]\w+", "", text)
    text = re.sub(r"\d+", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    return text

def tokenize(text):
    return text.split()

def normalize_slang(tokens):
    return [slang_dict.get(t, t) for t in tokens]

def remove_stopword(tokens):
    return [t for t in tokens if t not in stopwords]

def stemming(tokens):
    return [stemmer.stem(t) for t in tokens]

def preprocess_pipeline(text):
    cleaned = clean_text(text)
    tokens = tokenize(cleaned)
    tokens = normalize_slang(tokens)
    # tokens = remove_stopword(tokens)
    # HANYA STEMMING JIKA DIPAKAI SAAT TRAINING
    # tokens = stemming(tokens)
    return cleaned, tokens

# =========================================================
# WORD2VEC VECTOR
# =========================================================
def get_w2v_vector(tokens, model, size=100):
    vec = np.zeros(size)
    count = 0

    for w in tokens:
        if w in model.wv:
            vec += model.wv[w]
            count += 1

    if count == 0:
        return np.zeros(size)

    return vec / count

# =========================================================
# PREDIKSI
# =========================================================
def predict_sentiment(text):
    cleaned, tokens = preprocess_pipeline(text)
    vector = get_w2v_vector(tokens, w2v_model, 100).reshape(1, -1)
    pred = svm_model.predict(vector)[0]
    return cleaned, tokens, pred

# =========================================================
# STREAMLIT LAYOUT
# =========================================================
st.set_page_config(layout="wide")

with st.sidebar:
    selected = option_menu(
        "Navigasi",
        ["Dashboard", "Input CSV", "Input Text"],
        icons=['bar-chart', 'filetype-csv', 'pencil-square'],
        default_index=0
    )

# =========================================================
# 1. DASHBOARD PAGE
# =========================================================
if selected == "Dashboard":
    st.title("📊 Dashboard Sentimen Word2Vec + SVM")

    df = pd.read_csv("data_preprocessed.csv")
    st.subheader("Dataset Preprocessed")
    st.dataframe(df, height=400)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📈 Distribusi Rating")
        fig, ax = plt.subplots()
        df["score"].value_counts().sort_index().plot(kind='bar', ax=ax)
        st.pyplot(fig)

    with col2:
        st.subheader("📉 Distribusi Label Sentimen")
        fig2, ax2 = plt.subplots()
        df["label"].value_counts().plot(kind='bar', color="orange", ax=ax2)
        st.pyplot(fig2)

# =========================================================
# 2. INPUT CSV PAGE
# =========================================================
elif selected == "Input CSV":
    st.title("📥 Analisis Sentimen dari File CSV")

    uploaded = st.file_uploader("Upload File CSV", type=["csv"])

    if uploaded:
        df_raw = pd.read_csv(uploaded)
        st.subheader("Data Mentah")
        st.dataframe(df_raw, height=400)

        if "content" not in df_raw.columns:
            st.error("CSV harus ada kolom: content")
        else:
            if st.button("🔍 Analisis"):
                # Preprocessing
                df_raw["clean"], df_raw["tokens"] = zip(*df_raw["content"].apply(preprocess_pipeline))

                # Vector Word2Vec
                df_raw["vector"] = df_raw["tokens"].apply(lambda x: get_w2v_vector(x, w2v_model, 100))

                # Prediksi
                df_raw["prediction"] = df_raw["vector"].apply(
                    lambda v: svm_model.predict(v.reshape(1, -1))[0]
                )

                st.subheader("📌 Hasil Preprocessing")
                st.dataframe(df_raw[["content", "clean", "tokens"]])

                st.subheader("📌 Hasil Prediksi Sentimen")
                st.dataframe(df_raw[["content", "prediction"]])

# =========================================================
# 3. INPUT TEXT PAGE (SAMA FORMAT DENGAN INPUT CSV)
# =========================================================
elif selected == "Input Text":
    st.title("✍️ Analisis Teks Manual")

    user_text = st.text_area("Masukkan teks ulasan:")

    if st.button("🔍 Analisis"):
        if user_text.strip() == "":
            st.warning("Teks tidak boleh kosong!")
        else:
            # Preprocess
            clean, tokens = preprocess_pipeline(user_text)

            # Vector Word2Vec
            vector = get_w2v_vector(tokens, w2v_model, 100).reshape(1, -1)

            # Predict
            pred = svm_model.predict(vector)[0]

            st.subheader("📌 Hasil Sentimen")
            st.success(f"Prediksi Sentimen: **{pred}**")

