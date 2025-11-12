import streamlit as st
import pandas as pd
import numpy as np
import re
import string
import joblib
import nltk

from gensim.models import Word2Vec
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from streamlit_option_menu import option_menu

# -------------------------------------------------------
# NLTK DOWNLOAD
# -------------------------------------------------------
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

# -------------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------------
st.set_page_config(
    page_title="Sentiment Analysis SVM + Word2Vec",
    layout="wide"
)

# -------------------------------------------------------
# LOAD MODEL
# -------------------------------------------------------
@st.cache_resource
def load_models():
    w2v_model = Word2Vec.load("w2v_model.pkl")
    svm_model= joblib.load("svm_w2v_model.pkl")
    return w2v_model, svm_model

w2v_model, svm_model = load_models()

# -------------------------------------------------------
# PREPROCESSING
# -------------------------------------------------------
factory = StemmerFactory()
stemmer = factory.create_stemmer()

stop_words = set(stopwords.words('indonesian'))
stop_words.update({
    'yg','dg','rt','dgn','ny','d','klo','kalo','amp','biar',
    'bikin','bilang','gak','ga','krn','nya'
})

def remove_emoji(text):
    emoji_pattern = re.compile("[" 
        u"\U0001F600-\U0001F64F"
        u"\U0001F300-\U0001F5FF"
        u"\U0001F680-\U0001F6FF"
        u"\U0001F1E0-\U0001F1FF"
        u"\U00002700-\U000027BF"
        u"\U0001F900-\U0001F9FF"
        u"\U0001FA70-\U0001FAFF"
        u"\U00002600-\U000026FF"
        u"\U00002000-\U000023FF"
        "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+|https\S+", '', text)
    text = re.sub(r'@\w+|#\w+', '', text)
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    return remove_emoji(text).strip()

def preprocess_text_detailed(text):
    """Preprocessing dengan detail setiap tahap"""
    steps = {}
    
    # Original
    steps['original'] = str(text)
    
    # Case folding & cleaning
    text_clean = clean_text(text)
    steps['cleaned'] = text_clean
    
    # Tokenization
    tokens = word_tokenize(text_clean)
    steps['tokenized'] = tokens
    
    # Stopword removal
    tokens_no_stop = [t for t in tokens if t not in stop_words]
    steps['stopword_removed'] = tokens_no_stop
    
    # Stemming
    tokens_stemmed = [stemmer.stem(t) for t in tokens_no_stop]
    steps['stemmed'] = tokens_stemmed
    
    return steps

def preprocess_text(text):
    """Preprocessing standar untuk prediksi"""
    text = clean_text(text)
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t not in stop_words]
    tokens = [stemmer.stem(t) for t in tokens]
    return tokens

def sentence_vector(tokens):
    word_vectors = [w2v_model.wv[w] for w in tokens if w in w2v_model.wv]
    if len(word_vectors) == 0:
        return np.zeros(w2v_model.vector_size)
    return np.mean(word_vectors, axis=0)

# -------------------------------------------------------
# SIDEBAR NAVIGASI
# -------------------------------------------------------
with st.sidebar:
    selected = option_menu(
        "Navigasi",
        ["Dashboards", "Upload CSV", "Input Teks"],
        default_index=0
    )

# -------------------------------------------------------
# DASHBOARD
# -------------------------------------------------------
if selected == "Dashboards":
    st.title("📊 Dashboards — Analisis Sentimen MyIM3")

    # Load Data
    df_data = pd.read_csv("datafix.csv")

    st.subheader("Dataset Overview")
    st.write("Total data:", len(df_data))
    st.dataframe(df_data, use_container_width=True)

    # Visualisasi jumlah rating
    if "score" in df_data.columns:
        st.subheader("📈 Distribusi Rating Pengguna")
        st.bar_chart(df_data["score"].value_counts().sort_index())

    if "label" in df_data.columns:
        st.subheader("📈 Distribusi Label Sentimen")
        st.bar_chart(df_data["label"].value_counts().sort_index())


# -------------------------------------------------------
# INPUT TEKS 
# -------------------------------------------------------
elif selected == "Input Teks":
    st.title("📝 Analisis Sentimen Teks Tunggal")
    
    # init session state
    if 'preprocessed_data' not in st.session_state:
        st.session_state.preprocessed_data = None
    if 'user_text' not in st.session_state:
        st.session_state.user_text = ""

    user_text = st.text_area("Masukkan teks ulasan:", height=150, value=st.session_state.user_text)

    # update session if user types new text (reset preprocessing)
    if user_text != st.session_state.user_text:
        st.session_state.user_text = user_text
        st.session_state.preprocessed_data = None

    col1, col2, col3 = st.columns([1,1,1])
    with col1:
        preprocess_btn = st.button("🔄 Preprocessing")
    with col2:
        analyze_btn = st.button(
            "🎯 Analisis Sentimen",
            disabled=(st.session_state.preprocessed_data is None)
        )
    with col3:
        if st.button("🗑️ Clear"):
            st.session_state.preprocessed_data = None
            st.session_state.user_text = ""
            st.experimental_rerun()

    # RUN PREPROCESSING
    if preprocess_btn:
        if not user_text.strip():
            st.warning("⚠️ Masukkan teks terlebih dahulu.")
        else:
            with st.spinner("Memproses preprocessing..."):
                steps = preprocess_text_detailed(user_text)  # fungsi yang mengembalikan dict langkah2
                # simpan ke session_state
                st.session_state.preprocessed_data = steps
            st.success("✅ Preprocessing selesai!")

    # Tampilkan hasil preprocessing (format serupa CSV)
    if st.session_state.preprocessed_data is not None:
        steps = st.session_state.preprocessed_data  # ambil dari session_state (penting!)
        st.markdown("---")
        st.subheader("🔍 Ringkasan Preprocessing")

        original = steps.get("original", "")
        tokenized = steps.get("tokenized", [])
        stopword_removed = steps.get("stopword_removed", [])
        stemmed = steps.get("stemmed", [])

        df_preview = pd.DataFrame({
            "Teks Original": [original],
            "Hasil Preprocessing (Tokens)": [" | ".join(tokenized)],
            "Teks Setelah Preprocessing": [" ".join(stemmed)]
        })

        st.dataframe(df_preview, use_container_width=True, height=150)

    # ANALISIS SENTIMEN (setelah preprocessing)
    if analyze_btn:
        if st.session_state.preprocessed_data is None:
            st.error("❌ Lakukan preprocessing terlebih dahulu.")
        else:
            with st.spinner("Menganalisis sentimen..."):
                tokens = st.session_state.preprocessed_data.get("stemmed", [])
                vec = sentence_vector(tokens).reshape(1, -1)

                pred = svm_model.predict(vec)[0]
                try:
                    proba = svm_model.predict_proba(vec)[0]
                except Exception:
                    proba = None

            st.markdown("---")
            st.subheader("🎯 Hasil Analisis Sentimen")
            col1, col2 = st.columns([2,1])
            with col1:
                if pred == 1:
                    st.success("✅ **Sentimen: POSITIF**")
                else:
                    st.error("❌ **Sentimen: NEGATIF**")
            with col2:
                if proba is not None:
                    st.metric("Tingkat Keyakinan", f"{proba.max()*100:.2f}%")
                    prob_df = pd.DataFrame({
                        "Label": ["Negatif", "Positif"],
                        "Prob (%)": [proba[0]*100, proba[1]*100]
                    }).set_index("Label")
                else:
                    st.info("Model tidak mendukung probabilitas (predict_proba).")



# -------------------------------------------------------
# UPLOAD CSV
# -------------------------------------------------------
elif selected == "Upload CSV":
    st.title("📁 Analisis Sentimen CSV")
    
    # Initialize session state for CSV
    if 'csv_preprocessed' not in st.session_state:
        st.session_state.csv_preprocessed = None
    if 'csv_data' not in st.session_state:
        st.session_state.csv_data = None

    uploaded = st.file_uploader("Upload file CSV 📄", type=["csv"])

    if uploaded:
        df = pd.read_csv(uploaded)
        st.session_state.csv_data = df

        if "content" not in df.columns:
            st.error("❌ Kolom 'content' tidak ditemukan!")
        else:
            st.write("**Preview Data**")
            st.dataframe(df.head(10), use_container_width=True, height=300)
            st.info(f"Total baris: {len(df)}")

            col1, col2, col3 = st.columns(3)
            
            with col1:
                preprocess_csv_btn = st.button("🔄 Preprocessing Data", type="secondary", use_container_width=True)
            
            with col2:
                analyze_csv_btn = st.button("🎯 Analisis Sentimen", type="primary", use_container_width=True,
                                           disabled=st.session_state.csv_preprocessed is None)
            
            # PREPROCESSING CSV
            if preprocess_csv_btn:
                with st.spinner("Memproses preprocessing... Harap tunggu..."):
                    progress_bar = st.progress(0)
                    
                    # Preprocessing
                    df["tokens"] = df["content"].astype(str).apply(preprocess_text)
                    progress_bar.progress(50)
                    
                    # Create preprocessed text column
                    df["preprocessed_text"] = df["tokens"].apply(lambda x: " ".join(x))
                    progress_bar.progress(100)
                    
                    st.session_state.csv_preprocessed = df
                
                st.success("✅ Preprocessing selesai!")
            
            # Display preprocessing results
            if st.session_state.csv_preprocessed is not None:
                st.markdown("---")
                st.subheader("🔍 Hasil Preprocessing")
                
                df_preprocessed = st.session_state.csv_preprocessed
                
                # Show comparison
                comparison_df = df_preprocessed[["content", "tokens", "preprocessed_text"]].head(20)
                comparison_df.columns = ["Teks Original", "Hasil Preprocessing","Teks Setelah Preprocessing"]
                
                st.dataframe(comparison_df, use_container_width=True, height=400)
                
                # Statistics
                col1, col2, col3 = st.columns(3)
                with col1:
                    avg_tokens = df_preprocessed["tokens"].apply(len).mean()
                    st.metric("Rata-rata Token per Review", f"{avg_tokens:.1f}")
                with col2:
                    total_tokens = df_preprocessed["tokens"].apply(len).sum()
                    st.metric("Total Token", f"{total_tokens:,}")
                with col3:
                    unique_tokens = len(set([token for tokens in df_preprocessed["tokens"] for token in tokens]))
                    st.metric("Unique Token", f"{unique_tokens:,}")
                
                # Download preprocessed data
                csv_preprocessed = df_preprocessed[["content", "preprocessed_text", "tokens"]].to_csv(index=False).encode("utf-8")
                st.download_button(
                    "📥 Download Data Preprocessing",
                    csv_preprocessed,
                    "data_preprocessed.csv",
                    "text/csv",
                    use_container_width=True
                )

            # SENTIMENT ANALYSIS CSV
            if analyze_csv_btn:
                if st.session_state.csv_preprocessed is None:
                    st.error("❌ Lakukan preprocessing terlebih dahulu!")
                else:
                    with st.spinner("Menganalisis sentimen... Harap tunggu..."):
                        progress_bar = st.progress(0)
                        
                        df_analysis = st.session_state.csv_preprocessed.copy()
                        
                        # Vectorization
                        df_analysis["vector"] = df_analysis["tokens"].apply(sentence_vector)
                        progress_bar.progress(33)
                        
                        # Prediction
                        X = np.vstack(df_analysis["vector"].values)
                        preds = svm_model.predict(X)
                        progress_bar.progress(66)
                        
                        df_analysis["sentiment"] = ["Positif" if p == 1 else "Negatif" for p in preds]
                        
                        # Probability
                        try:
                            proba = svm_model.predict_proba(X)
                            df_analysis["confidence"] = [p.max() * 100 for p in proba]
                        except:
                            df_analysis["confidence"] = None
                        
                        progress_bar.progress(100)

                    st.success("✅ Analisis Sentimen Selesai!")
                    
                    st.markdown("---")
                    st.subheader("📊 Hasil Analisis")
                    
                    # Statistics
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Data", len(df_analysis))
                    with col2:
                        positive_count = (df_analysis["sentiment"] == "Positif").sum()
                        positive_pct = (positive_count / len(df_analysis)) * 100
                        st.metric("Sentimen Positif", f"{positive_count} ({positive_pct:.1f}%)")
                    with col3:
                        negative_count = (df_analysis["sentiment"] == "Negatif").sum()
                        negative_pct = (negative_count / len(df_analysis)) * 100
                        st.metric("Sentimen Negatif", f"{negative_count} ({negative_pct:.1f}%)")
                    
                    # Result table
                    st.subheader("📋 Tabel Hasil")
                    if "confidence" in df_analysis.columns and df_analysis["confidence"] is not None:
                        result_df = df_analysis[["content", "preprocessed_text", "sentiment", "confidence"]]
                        result_df.columns = ["Teks Original", "Preprocessing", "Sentimen", "Confidence (%)"]
                    else:
                        result_df = df_analysis[["content", "preprocessed_text", "sentiment"]]
                        result_df.columns = ["Teks Original", "Preprocessing", "Sentimen"]
                    
                    st.dataframe(result_df, use_container_width=True, height=400)

                    # Download results
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        csv_result = df_analysis.to_csv(index=False).encode("utf-8")
                        st.download_button(
                            "📥 Download Hasil Lengkap",
                            csv_result,
                            "hasil_sentimen_lengkap.csv",
                            "text/csv",
                            use_container_width=True
                        )
                    
                    with col2:
                        if "confidence" in df_analysis.columns and df_analysis["confidence"] is not None:
                            summary_df = df_analysis[["content", "sentiment", "confidence"]]
                        else:
                            summary_df = df_analysis[["content", "sentiment"]]
                        
                        csv_summary = summary_df.to_csv(index=False).encode("utf-8")
                        st.download_button(
                            "📥 Download Ringkasan",
                            csv_summary,
                            "hasil_sentimen.csv",
                            "text/csv",
                            use_container_width=True
                        )