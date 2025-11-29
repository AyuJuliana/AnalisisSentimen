import streamlit as st
import pandas as pd
import numpy as np
import re
import string
import json
import joblib
from gensim.models import Word2Vec
from streamlit_option_menu import option_menu
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# ========== KONFIGURASI PAGE ==========
st.set_page_config(
    page_title="Dashboard Sentimen Word2Vec + SVM",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ========== CUSTOM CSS ==========
st.markdown("""
    <style>
    /* Dark Theme */
    .main {
        background-color: #1e1e1e;
        color: #ffffff;
    }
    
    /* Sidebar Dark */
    [data-testid="stSidebar"] {
        background-color: #252525;
    }
    
    /* Header Title */
    .dashboard-header {
        display: flex;
        align-items: center;
        gap: 20px;
        padding: 20px 0;
        border-bottom: 2px solid #333;
        margin-bottom: 30px;
    }
    
    .dashboard-title {
        font-size: 42px;
        font-weight: 700;
        color: #ffffff;
        margin: 0;
    }
    
    /* Table Styling */
    .dataframe {
        background-color: #2a2a2a !important;
        color: #ffffff !important;
        border-radius: 10px;
        overflow: hidden;
    }
    
    .dataframe thead tr th {
        background-color: #1a1a1a !important;
        color: #888888 !important;
        font-weight: 600;
        text-transform: uppercase;
        font-size: 12px;
        padding: 15px !important;
        border: none !important;
    }
    
    .dataframe tbody tr td {
        background-color: #2a2a2a !important;
        color: #ffffff !important;
        padding: 15px !important;
        border-bottom: 1px solid #333 !important;
    }
    
    .dataframe tbody tr:hover td {
        background-color: #333333 !important;
    }
    
    /* Section Title */
    .section-title {
        font-size: 28px;
        font-weight: 700;
        color: #ffffff;
        margin-top: 30px;
        margin-bottom: 20px;
    }
    
    /* Metric Card */
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 12px;
        color: white;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }
    
    /* Button Styling */
    .stButton>button {
        background-color: #ff4b4b;
        color: white;
        border-radius: 8px;
        padding: 12px 24px;
        font-weight: 600;
        border: none;
        transition: all 0.3s;
    }
    
    .stButton>button:hover {
        background-color: #ff3333;
        box-shadow: 0 4px 12px rgba(255, 75, 75, 0.4);
    }
    
    /* Hide Streamlit Branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Input Styling */
    .stTextInput>div>div>input,
    .stTextArea>div>div>textarea {
        background-color: #2a2a2a;
        color: #ffffff;
        border: 1px solid #444;
        border-radius: 8px;
    }
    
    /* File Uploader */
    [data-testid="stFileUploader"] {
        background-color: #2a2a2a;
        border-radius: 8px;
        padding: 20px;
    }
    </style>
""", unsafe_allow_html=True)

# ========== LOAD MODEL ==========
@st.cache_resource
def load_models():
    try:
        svm_model = joblib.load("svm_model.pkl")
        w2v_model = Word2Vec.load("word2vec_model.model")
        return svm_model, w2v_model
    except FileNotFoundError:
        st.error("❌ Model tidak ditemukan! Pastikan file 'svm_model.pkl' dan 'word2vec_model.model' ada di folder yang sama.")
        return None, None

svm_model, w2v_model = load_models()

# ========== LOAD SLANGWORDS & STOPWORDS ==========
@st.cache_data
def load_slangwords(path="slangwords.txt"):
    """Load slangwords dictionary from file"""
    slang = {}
    try:
        # Try loading as JSON first
        with open(path, "r", encoding="utf-8") as f:
            slang = json.load(f)
    except json.JSONDecodeError:
        # If not JSON, try key=value format
        try:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    if "=" in line:
                        a, b = line.split("=", 1)
                        slang[a.strip()] = b.strip()
        except Exception as e:
            st.error(f"❌ Error loading slangwords: {str(e)}")
    except FileNotFoundError:
        st.error("❌ File slangwords.txt tidak ditemukan! Silakan letakkan file di folder yang sama dengan app.")
    return slang

@st.cache_data
def load_stopwords(path="stopwords.txt"):
    """Load stopwords list from file"""
    try:
        with open(path, "r", encoding="utf-8") as f:
            stopwords_list = set([
                x.strip()
                for x in f.read().splitlines()
                if x.strip() != ""
            ])
        return stopwords_list
    except FileNotFoundError:
        return set()
    except Exception as e:
        return set()

# Load slangwords dan stopwords dari file
slang_dict = load_slangwords()
stopwords = load_stopwords()

# ========== PREPROCESSING ==========
factory = StemmerFactory()
stemmer = factory.create_stemmer()

def remove_emoji(text):
    emoji_pattern = re.compile(
        "["
        u"\U0001F600-\U0001F64F"
        u"\U0001F300-\U0001F5FF"
        u"\U0001F680-\U0001F6FF"
        u"\U0001F1E0-\U0001F1FF"
        u"\U00002700-\U000027BF"
        u"\U0001F900-\U0001F9FF"
        u"\U0001FA70-\U0001FAFF"
        u"\U00002600-\U000026FF"
        u"\U00002000-\U000023FF"
        "]+", flags=re.UNICODE
    )
    return emoji_pattern.sub('', text)

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"[@#]\w+", "", text)
    text = re.sub(r"\d+", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = remove_emoji(text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def tokenize(text):
    return text.split()

def normalize_slang(tokens):
    return [slang_dict.get(t, t) for t in tokens]

def remove_stopword(tokens):
    return [t for t in tokens if t not in stopwords]

def stemming(tokens):
    return [stemmer.stem(t) for t in tokens if t]

def preprocess_pipeline(text):
    cleaned = clean_text(text)
    tokens = tokenize(cleaned)
    tokens = normalize_slang(tokens)
    tokens = remove_stopword(tokens)
    tokens = stemming(tokens)
    return cleaned, tokens

# ========== WORD2VEC VECTOR ==========
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

# ========== PREDIKSI ==========
def predict_sentiment(text):
    cleaned, tokens = preprocess_pipeline(text)
    vector = get_w2v_vector(tokens, w2v_model, 100).reshape(1, -1)
    pred = svm_model.predict(vector)[0]
    label = 'positif' if pred == 1 else 'negatif'
    return cleaned, tokens, label

# ========== SIDEBAR ==========
with st.sidebar:
    selected = option_menu(
        "Navigasi",
        ["Dashboard", "Input CSV", "Input Text"],
        icons=['bar-chart', 'filetype-csv', 'pencil-square'],
        default_index=0,
        styles={
            "container": {"padding": "5px", "background-color": "#252525"},
            "icon": {"color": "#ffffff", "font-size": "18px"},
            "nav-link": {
                "font-size": "16px",
                "text-align": "left",
                "margin": "5px",
                "padding": "10px",
                "border-radius": "8px",
                "color": "#ffffff"
            },
            "nav-link-selected": {"background-color": "#ff4b4b"},
        }
    )

# ========== MENU DASHBOARD ==========
if selected == "Dashboard":
    st.markdown("""
        <div class="dashboard-header">
            <div style="font-size: 50px;">📊</div>
            <h1 class="dashboard-title">Dashboard Sentimen Word2Vec + SVM</h1>
        </div>
    """, unsafe_allow_html=True)
    
    try:
        df = pd.read_csv("data_preprocessed.csv")
        
        st.markdown('<p class="section-title">Dataset Preprocessed</p>', unsafe_allow_html=True)
        
        # Display data
        st.dataframe(df, use_container_width=True, height=500)
        
        # Metrics
        st.markdown("---")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
                <div class="metric-card">
                    <h3 style="margin:0; font-size:16px; opacity:0.8;">Total Data</h3>
                    <h1 style="margin:10px 0 0 0; font-size:36px;">{len(df)}</h1>
                </div>
            """, unsafe_allow_html=True)
        
        with col2:
            if 'label' in df.columns:
                positive = len(df[df['label'] == 'positif'])
                st.markdown(f"""
                    <div class="metric-card">
                        <h3 style="margin:0; font-size:16px; opacity:0.8;">Positif</h3>
                        <h1 style="margin:10px 0 0 0; font-size:36px;">{positive}</h1>
                    </div>
                """, unsafe_allow_html=True)
        
        with col3:
            if 'label' in df.columns:
                negative = len(df[df['label'] == 'negatif'])
                st.markdown(f"""
                    <div class="metric-card">
                        <h3 style="margin:0; font-size:16px; opacity:0.8;">Negatif</h3>
                        <h1 style="margin:10px 0 0 0; font-size:36px;">{negative}</h1>
                    </div>
                """, unsafe_allow_html=True)
        
        with col4:
            if 'score' in df.columns:
                avg_score = df['score'].mean()
                st.markdown(f"""
                    <div class="metric-card">
                        <h3 style="margin:0; font-size:16px; opacity:0.8;">Avg Score</h3>
                        <h1 style="margin:10px 0 0 0; font-size:36px;">{avg_score:.1f}</h1>
                    </div>
                """, unsafe_allow_html=True)
        
        # Visualizations
        st.markdown("---")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<p class="section-title">📈 Distribusi Rating</p>', unsafe_allow_html=True)
            if 'score' in df.columns:
                fig_score = px.bar(
                    df['score'].value_counts().sort_index(),
                    title="",
                    labels={'value': 'Jumlah', 'index': 'Score'}
                )
                fig_score.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='white'),
                    showlegend=False
                )
                st.plotly_chart(fig_score, use_container_width=True)
        
        with col2:
            st.markdown('<p class="section-title">📉 Distribusi Label Sentimen</p>', unsafe_allow_html=True)
            if 'label' in df.columns:
                fig_label = px.pie(
                    df['label'].value_counts(),
                    values=df['label'].value_counts().values,
                    names=df['label'].value_counts().index,
                    color_discrete_map={'positif': '#00CC96', 'negatif': '#EF553B'},
                    hole=0.4
                )
                fig_label.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='white')
                )
                st.plotly_chart(fig_label, use_container_width=True)
    
    except FileNotFoundError:
        st.error("❌ File 'data_preprocessed.csv' tidak ditemukan!")

# ========== MENU INPUT CSV ==========
elif selected == "Input CSV":
    st.markdown("""
        <div class="dashboard-header">
            <div style="font-size: 50px;">📁</div>
            <h1 class="dashboard-title">Analisis Sentimen dari File CSV</h1>
        </div>
    """, unsafe_allow_html=True)
    
    if svm_model is None or w2v_model is None:
        st.error("❌ Model tidak tersedia!")
        st.stop()
    
    uploaded = st.file_uploader("Upload File CSV", type=["csv"])
    
    if uploaded:
        df_raw = pd.read_csv(uploaded)
        
        st.markdown('<p class="section-title">Data Mentah</p>', unsafe_allow_html=True)
        st.dataframe(df_raw.head(20), use_container_width=True, height=400)
        
        if "content" not in df_raw.columns:
            st.error("❌ CSV harus ada kolom: content")
        else:
            if st.button("🔍 Analisis", type="primary", use_container_width=True):
                with st.spinner("Memproses analisis..."):
                    # Preprocessing
                    results = []
                    for text in df_raw["content"]:
                        cleaned, tokens = preprocess_pipeline(text)
                        vector = get_w2v_vector(tokens, w2v_model, 100)
                        pred = svm_model.predict(vector.reshape(1, -1))[0]
                        label = 'positif' if pred == 1 else 'negatif'
                        results.append({
                            'cleaned': cleaned,
                            'tokens': str(tokens),
                            'prediction': label
                        })
                    
                    df_results = pd.DataFrame(results)
                    df_raw['clean'] = df_results['cleaned']
                    df_raw['tokens'] = df_results['tokens']
                    df_raw['prediction'] = df_results['prediction']
                    
                    st.success("✅ Analisis selesai!")
                    
                    st.markdown('<p class="section-title">📌 Hasil Preprocessing</p>', unsafe_allow_html=True)
                    st.dataframe(df_raw[["content", "clean", "tokens"]].head(20), use_container_width=True, height=400)
                    
                    st.markdown('<p class="section-title">📌 Hasil Prediksi Sentimen</p>', unsafe_allow_html=True)
                    st.dataframe(df_raw[["content", "prediction"]].head(20), use_container_width=True, height=400)
                    
                    # Download button
                    csv = df_raw.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        "📥 Download Hasil CSV",
                        csv,
                        "hasil_prediksi.csv",
                        "text/csv",
                        key='download-csv',
                        use_container_width=True
                    )
                    
                    # Summary
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Dianalisis", len(df_raw))
                    with col2:
                        positif_count = len(df_raw[df_raw['prediction'] == 'positif'])
                        st.metric("Positif", positif_count)
                    with col3:
                        negatif_count = len(df_raw[df_raw['prediction'] == 'negatif'])
                        st.metric("Negatif", negatif_count)

# ========== MENU INPUT TEXT ==========
elif selected == "Input Text":
    st.markdown("""
        <div class="dashboard-header">
            <div style="font-size: 50px;">✍️</div>
            <h1 class="dashboard-title">Analisis Teks Manual</h1>
        </div>
    """, unsafe_allow_html=True)
    
    if svm_model is None or w2v_model is None:
        st.error("❌ Model tidak tersedia!")
        st.stop()
    
    user_text = st.text_area(
        "Masukkan teks ulasan:",
        placeholder="Contoh: Aplikasi sangat bagus dan mudah digunakan!",
        height=150
    )
    
    if st.button("🔍 Analisis", type="primary", use_container_width=True):
        if user_text.strip() == "":
            st.warning("⚠️ Teks tidak boleh kosong!")
        else:
            with st.spinner("Memproses..."):
                # Preprocess
                clean, tokens = preprocess_pipeline(user_text)
                
                # Vector Word2Vec
                vector = get_w2v_vector(tokens, w2v_model, 100).reshape(1, -1)
                
                # Predict
                pred = svm_model.predict(vector)[0]
                label = 'positif' if pred == 1 else 'negatif'
                
                st.markdown("---")
                
                # Display Result
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    if label == 'positif':
                        st.markdown("""
                            <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                                        padding: 40px; border-radius: 15px; text-align: center;">
                                <div style="font-size: 80px; margin-bottom: 10px;">😊</div>
                                <h2 style="margin: 0; color: white;">POSITIF</h2>
                            </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown("""
                            <div style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); 
                                        padding: 40px; border-radius: 15px; text-align: center;">
                                <div style="font-size: 80px; margin-bottom: 10px;">😞</div>
                                <h2 style="margin: 0; color: white;">NEGATIF</h2>
                            </div>
                        """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown('<p class="section-title">Detail Preprocessing</p>', unsafe_allow_html=True)
                    st.info(f"**Text Original:**\n\n{user_text}")
                    st.success(f"**Text Cleaned:**\n\n{clean}")
                    st.warning(f"**Tokens:**\n\n{' | '.join(tokens)}")
                    st.metric("Jumlah Token", len(tokens))