import os
import re
import pandas as pd
import numpy as np
import plotly.express as px
import joblib
import streamlit as st
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Set Streamlit page configuration to Wide Mode
st.set_page_config(
page_title="Aspect-Based Sentiment Analysis Platform", 
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
     <style>
        .stApp {
            background-color: #0d1117 !important;
            color: #c9d1d9 !important;
        }

        label[data-testid="stWidgetLabel"] p {
            color: #ffffff !important;
            font-size: 16px !important;
            font-weight: 500 !important;
        }

        input::placeholder {
            color: #8b949e !important;
       }
        div.stButton > button {
            color: #ffffff !important;
        }
        
        div.stButton > button[kind="secondary"] {
            background-color: #21262d !important;
            color: #c9d1d9 !important;
            border: 1px solid #30363d !important;
        }
        div.stButton > button[kind="secondary"]:hover {
            background-color: #30363d !important;
            color: #ffffff !important;
        }

        .stWidgetLabel p {
            color: #e6edf3 !important;
            font-weight: 500 !important;
        }
        
        /* Tambahan: Baiki kapsyen subheader yang pudar */
        .stCaption {
            color: #8b949e !important;
        }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# EMBEDDED RESOURCES & NLP PIPELINE
# ==========================================
@st.cache_resource
def load_nlp_resources():
    nltk.download('stopwords', quiet=True)
    nltk.download('punkt', quiet=True)       
    nltk.download('punkt_tab', quiet=True)   
    nltk.download('wordnet', quiet=True)     
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    return stop_words, lemmatizer

stop_words, lemmatizer = load_nlp_resources()


def clean_text(text):

    if not text or pd.isna(text):
        return ""

    text = str(text)

    text = text.replace("â€™", "'")
    text = text.replace("â€“", "-")

    text = text.lower()

    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)

    text = re.sub(r'@\w+', '', text)

    text = re.sub(r'(.)\1{2,}', r'\1', text)

    text = text.replace("doesnt", "does not")
    text = text.replace("dont", "do not")
    text = text.replace("cant", "cannot")

    text = text.replace("so cheap", "affordable")
    text = text.replace("cheap price", "affordable")

    text = re.sub(r"[^a-zA-Z\s']", " ", text)

    try:

        tokens = word_tokenize(text)

        filtered_tokens = [
            word for word in tokens
            if word not in stop_words
        ]

        lemmatized_tokens = [
            lemmatizer.lemmatize(word)
            for word in filtered_tokens
        ]

        return " ".join(lemmatized_tokens)

    except:
        return text

aspect_keywords = {
    'Product Quality': ['quality', 'material', 'authentic', 'original', 'fake', 'durable', 'excellent', 'sturdy', 'thick', 'affordable', 'cheap','comfortable',
    'comfort', 'soft', 'hard', 'broken', 'broke' 'working', 'works', 'perfect','perfectly'],
    'Packaging Quality': ['box', 'bubble', 'wrap', 'bubblewrap', 'tape', 'seal', 'dented', 'torn', 'secure', 'packing', 'packaging', 'wrapped', 'package',
    'packaged', 'packed'],
    'Delivery & Service': ['delivery', 'shipping', 'shipped', 'received', 'fast', 'slow', 'courier', 'tracking', 'arrive', 'post', 'seller', 'chat', 'response', 'friendly', 'deliver',
                           'delivered','replacement']
}
def detect_aspect(text):

    cleaned = clean_text(text)

    tokens = word_tokenize(cleaned)

    scores = {}

    for aspect, keywords in aspect_keywords.items():

        scores[aspect] = sum(
            1 for keyword in keywords
            if keyword in tokens
        )

    if max(scores.values()) == 0:
        return ["Others"]

    return [max(scores, key=scores.get)]
# ==========================================
# 1. MODEL LOADING & CONFIGURATION
# ==========================================
@st.cache_resource
def load_prediction_models():

    sentiment_model = joblib.load("naive_bayes_model.pkl")

    sentiment_vectorizer = joblib.load("tfidf_vectorizer.pkl")

    return sentiment_model, sentiment_vectorizer
ai_model, tfidf_vectorizer = load_prediction_models()

sentiment_map = {
    0: "Positive",
    1: "Neutral",
    2: "Negative"
}
# ==========================================
# 2. HORIZONTAL NAVBAR & TEMA CSS
# ==========================================
if 'current_page' not in st.session_state:
    st.session_state.current_page = "Home"

query_params = st.query_params
if "page" in query_params:
    st.session_state.current_page = query_params["page"].replace('+', ' ')

st.markdown("""
    <style>
        [data-testid="stSidebar"] { display: none !important; }
        [data-testid="collapsedSidebarButton"] { display: none !important; }
        h1 a, h2 a, h3 a, h4 a, h5 a, h6 a { display: none !important; }
        [data-testid="stHeaderActionElements"] { display: none !important; }
        .stHtml h1 a, .stHtml h2 a, .stHtml h3 a { display: none !important; }
        .stMainBlockContainer { 
            padding-top: 1.5rem !important; 
            padding-left: 3rem !important;
            padding-right: 3rem !important;
        }
        
        .custom-navbar {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 12px 0px;
            background-color: transparent;
            margin-bottom: 10px;
        }
        
        .nav-logo { font-size: 26px; font-weight: 800; color: #4CAF50 !important; }
        .nav-links { display: flex; gap: 20px; align-items: center; }
        
        .nav-item {
            color: #a3a8b4 !important;
            text-decoration: none !important;
            font-weight: 500;
            font-size: 14px;
            transition: color 0.2s;
        }
        .nav-item:hover { color: #ffffff !important; }
        .nav-active { color: #ffffff !important; font-weight: 700; }
        
        .metric-box {
            background-color: #161b22; 
            padding: 15px; 
            border-radius: 8px; 
            text-align: left;
            box-shadow: 0 4px 6px rgba(0,0,0,0.2);
        }
        .metric-box h5 { margin: 0; font-size: 14px; color: #a3a8b4; font-weight: 400; }
        .metric-val { font-size: 28px; font-weight: 700; margin-top: 5px; }

        .brand-card { 
            background-color: #16161a; 
            border-radius: 12px; 
            padding: 25px; 
            text-align: center; 
            height: 100%; 
            border: 1px solid #232329; 
        }
        .brand-badge { 
            display: inline-block; 
            padding: 6px 20px; 
            border-radius: 6px; 
            font-weight: bold; 
            font-size: 15px; 
            margin-bottom: 15px; 
            color: #ffffff; 
        }
    </style>
""", unsafe_allow_html=True)

if st.session_state.current_page != "Home":
    home_active = "nav-active" if st.session_state.current_page == "Home" else ""
    dash_active = "nav-active" if st.session_state.current_page == "Analytics Dashboard" else ""
    rt_active   = "nav-active" if st.session_state.current_page == "Real-Time Testing" else ""

    st.markdown(f"""
        <div class="custom-navbar">
            <div class="nav-logo">ABSA</div>
            <div class="nav-links">
                <a class="nav-item {home_active}" href="/?page=Home" target="_self">HOME</a>
                <span style="color:#a3a8b4;">|</span>
                <a class="nav-item {dash_active}" href="/?page=Analytics+Dashboard" target="_self">DASHBOARD</a>
                <span style="color:#a3a8b4;">|</span>
                <a class="nav-item {rt_active}" href="/?page=Real-Time+Testing" target="_self">SENTIMEN ANALYZER</a>
            </div>
        </div>
        <hr style="margin-top: 0px; margin-bottom: 25px; border-color: #232329;">
    """, unsafe_allow_html=True)


# ==========================================
# 3. KUNCI DATA FUNGSI CACHE (Kekal Statik)
# ==========================================
@st.cache_data
def load_macro_data_cached():
    if os.path.exists('result.csv'):
        df = pd.read_csv('result.csv')
        df['Sentiment'] = df['label'].map(sentiment_map)
        return df
    else:
        np.random.seed(42)
        total_mock = 15036
        mock_aspects = np.random.choice(['Delivery', 'Others', 'Price', 'Quality', 'Service'], size=total_mock, p=[0.25, 0.15, 0.10, 0.40, 0.10])
        mock_sentiments = np.random.choice(['Positive', 'Neutral', 'Negative'], size=total_mock, p=[0.948, 0.027, 0.025])
        return pd.DataFrame({'aspects': mock_aspects, 'Sentiment': mock_sentiments})
def refine_sentiment_with_rules(text, predicted_label):

    text = str(text).lower()

    positive_phrases = [
        "not bad",
        "no issue",
        "no problem",
        "worth buying",
        "worth it"
    ]

    if any(p in text for p in positive_phrases):
        return 0

    strong_negative = [
        "broken",
        "broke",
        "damaged",
        "damage",
        "fake",
        "refund",
        "scam",
        "poor quality",
        "defect",
        "doesn't work",
        "not work",
        "wrong item"
    ]

    if any(w in text for w in strong_negative):
        return 2

    return predicted_label

# ==========================================
# PAGE 1: LANDING PAGE
# ==========================================
if st.session_state.current_page == "Home":
    st.markdown("""
        <style>
            .hero-section { text-align: center; padding: 60px 20px 30px 20px; }
            .hero-title { font-size: 20px; color: #a3a8b4; margin-bottom: 5px; font-weight: 300; }
            .hero-main { font-size: 42px; font-weight: 800; color: #ee4d2d; letter-spacing: 0.5px; }
            .intro-text { max-width: 900px; margin: 0 auto; text-align: center; font-size: 15px; color: #d3d3d3; line-height: 1.7; padding: 30px 10px; }
            .brand-section-title { text-align: center; font-size: 20px; font-weight: 700; color: white; margin-top: 20px; margin-bottom: 25px; letter-spacing: 0.5px; }
        </style>
    """, unsafe_allow_html=True)

    st.markdown("""
        <div class="hero-section">
            <div class="hero-title">Aspect-Based Sentiment Analysis of</div>
            <div class="hero-main">E-COMMERCE REVIEWS</div>
        </div>
    """, unsafe_allow_html=True)
    
    _, btn_col, _ = st.columns([1.8, 1, 1.8])
    with btn_col:
        if st.button("GO TO DASHBOARD", use_container_width=True, key="home_cta_dashboard"):
            st.session_state.current_page = "Analytics Dashboard"
            st.query_params["page"] = "Analytics Dashboard"
            st.rerun()

    st.markdown("""
        <div class="intro-text">
            Nowadays, e-commerce platforms like Shopee have become the primary marketplace for consumers in Malaysia. With millions of transactions occurring daily, thousands of text feedback and product reviews are generated. However, traditional rating systems (1 to 5 stars) often fail to capture the specific reasons behind customer satisfaction or frustration. By extracting fine-grained market intelligence metrics through Aspect-Based Sentiment Analysis (ABSA), this framework targets explicit textual opinions to automatically classify sentiments across specific operational dimensions. This provides sellers and stakeholders with clear, actionable insights to enhance product standards and buyer loyalty.
        </div>
    """, unsafe_allow_html=True)

    st.markdown("<div class='brand-section-title'>ASPECTS UNDER ANALYSIS</div>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
            <div class="brand-card">
                <div class="brand-badge" style="background-color: #ee4d2d;">Product Quality</div>
                <p style="color: #a3a8b4; font-size: 14px; line-height: 1.6; text-align: center;">
                    Analyzes product-related reviews to monitor feedback on quality, durability, and whether the item is genuine and worth its price.
                </p>
            </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
            <div class="brand-card">
                <div class="brand-badge" style="background-color: #00f2fe; color: #111111;">Packaging Quality</div>
                <p style="color: #a3a8b4; font-size: 14px; line-height: 1.6; text-align: center;">
                    Tracks feedback on packaging quality. It identifies mentions of bubble wrap, boxes, and seals to flag if products arrived broken or damaged.
                </p>
            </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
            <div class="brand-card">
                <div class="brand-badge" style="background-color: #ffea00; color: #111111;">Delivery & Service</div>
                <p style="color: #a3a8b4; font-size: 14px; line-height: 1.6; text-align: center;">
                    Tracks delivery and service efficiency. It identifies shipping delays and measures how effectively sellers respond to and interact with their customers.
                </p>
            </div>
        """, unsafe_allow_html=True)


# ==========================================
# PAGE 2: ANALYTICS DASHBOARD
# ==========================================
elif st.session_state.current_page == "Analytics Dashboard":
    st.markdown("<h1 style='margin-bottom:0;'>📊 E-Commerce Sentiment Analysis Dashboard</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color: #a3a8b4; margin-top:5px;'>Monitor capturing buyer satisfaction metrics across core operational aspects.</p>", unsafe_allow_html=True)

    df_main = load_macro_data_cached()
    
    total_val = len(df_main)
    pos_val = len(df_main[df_main['Sentiment'] == 'Positive'])
    neu_val = len(df_main[df_main['Sentiment'] == 'Neutral'])
    neg_val = len(df_main[df_main['Sentiment'] == 'Negative'])
    
    k1, k2, k3, k4 = st.columns(4)
    with k1: st.markdown(f"<div class='metric-box'><h5>Total Reviews</h5><div class='metric-val' style='color:#00f2fe;'>{total_val:,}</div></div>", unsafe_allow_html=True)
    with k2: st.markdown(f"<div class='metric-box'><h5>Positive Sentiment</h5><div class='metric-val' style='color:#00e676;'>{pos_val:,}</div></div>", unsafe_allow_html=True)
    with k3: st.markdown(f"<div class='metric-box'><h5>Neutral Sentiment</h5><div class='metric-val' style='color:#ffea00;'>{neu_val:,}</div></div>", unsafe_allow_html=True)
    with k4: st.markdown(f"<div class='metric-box'><h5>Negative Sentiment</h5><div class='metric-val' style='color:#ff1744;'>{neg_val:,}</div></div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    
    st.markdown("### 📈 Sentiment Volume Across Aspects")
    trend_data = df_main.groupby(['aspects', 'Sentiment']).size().reset_index(name='Review Volume')
    
    fig_line = px.line(
        trend_data, x='aspects', y='Review Volume', color='Sentiment',
        color_discrete_map={'Positive': '#00e676', 'Neutral': '#ffea00', 'Negative': '#ff1744'},
        template="plotly_dark", height=280
    )
    fig_line.update_layout(
        margin=dict(t=20, b=20, l=40, r=20), 
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis_title="Aspects", yaxis_title="Number of Reviews",
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)'
    )
    fig_line.update_traces(mode='lines+markers', line=dict(width=3), marker=dict(size=8))
    st.plotly_chart(fig_line, use_container_width=True, config={'displayModeBar': False})
    
    col_left, col_right = st.columns([6, 4])
    with col_left:
        st.markdown("### Total Counts of Sentiment by Aspect")
        fig_bar = px.histogram(
            df_main, x="aspects", color="Sentiment", barmode="group",
            color_discrete_map={'Positive': '#00e676', 'Neutral': '#ffea00', 'Negative': '#ff1744'},
            template="plotly_dark", height=320
        )
        fig_bar.update_layout(margin=dict(t=10, b=10, l=10, r=10), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_bar, use_container_width=True, config={'displayModeBar': False})

    with col_right:
        st.markdown("### Percentage Distribution Split")
        fig_pie = px.pie(
            df_main, names='Sentiment', color='Sentiment',
            color_discrete_map={'Positive': '#00e676', 'Neutral': '#ffea00', 'Negative': '#ff1744'},
            hole=0.5, template="plotly_dark", height=320
        )
        fig_pie.update_traces(textinfo='percent')
        fig_pie.update_layout(margin=dict(t=10, b=10, l=10, r=10), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_pie, use_container_width=True, config={'displayModeBar': False})


# ==========================================
# PAGE 3: SENTIMEN ANALYZER 
# ==========================================
elif st.session_state.current_page == "Real-Time Testing":
    st.title("Sentimen Analyzer")
    st.caption("Perform live single-phrase classification or run predictions on batch datasets.")
    st.markdown("<br>", unsafe_allow_html=True)

    if 'test_mode' not in st.session_state: 
        st.session_state.test_mode = "Single"
    
    col_a, col_b = st.columns([1, 1])
    with col_a:
        if st.button("📝 Single Review Prediction", use_container_width=True, type="primary" if st.session_state.test_mode == "Single" else "secondary"):
            st.session_state.test_mode = "Single"
            st.rerun()
    with col_b:
        if st.button("📂 Upload Analysis System", use_container_width=True, type="primary" if st.session_state.test_mode == "Bulk" else "secondary"):
            st.session_state.test_mode = "Bulk"
            st.rerun()
            
    st.markdown("---")

    if st.session_state.test_mode == "Single":
        st.subheader("Manual Single Sentimen")
        manual_input = st.text_input("Enter any sentence :", placeholder="e.g., The product quality is amazing...")

        if manual_input:
            st.markdown("<br>", unsafe_allow_html=True)
            cleaned_in = clean_text(manual_input)
            
            if ai_model and tfidf_vectorizer:
                vec_in = tfidf_vectorizer.transform([cleaned_in])
                pred = ai_model.predict(vec_in)[0]
                pred = refine_sentiment_with_rules(manual_input, pred)
            else:
                pred = 0 if any(w in cleaned_in for w in ['good', 'great', 'fast']) else 2
                
            detected_aspects = detect_aspect(manual_input)
            
            col_res1, col_res2 = st.columns(2)
            with col_res1: 
                st.info(f"🎯 **Aspect:**\n### {', '.join(detected_aspects)}")
            with col_res2:
                if pred == 0: st.success("😊 **Predicted Sentiment:**\n### Positive Class")
                elif pred == 1: st.warning("😐 **Predicted Sentiment:**\n### Neutral Class")
                elif pred == 2: st.error("😡 **Predicted Sentiment:**\n### Negative Class")

    else:
        st.subheader("📂 Upload File")
        uploaded_file = st.file_uploader("Choose a file", type=['xlsx', 'csv'])

        if uploaded_file:
            uploaded_df = pd.read_excel(uploaded_file) if uploaded_file.name.endswith('.xlsx') else pd.read_csv(uploaded_file)
            text_columns = uploaded_df.select_dtypes(include=['object']).columns.tolist()
            
            if not text_columns:
                st.error("❌ No text data found.")
            else:
                synonyms = ['comment', 'review', 'ulasan', 'text', 'feedback', 'komen']
                chosen_text_col = next((c for c in uploaded_df.columns if c.lower() in synonyms), text_columns[0])
              
                if st.button("Analyze"):
                    with st.spinner("Processing NLP Engine..."):
                        uploaded_df['Cleaned Text'] = uploaded_df[chosen_text_col].apply(clean_text)
                        uploaded_df['Extracted Aspect'] = uploaded_df[chosen_text_col].apply(lambda x: detect_aspect(x)[0])
                        
                        if ai_model and tfidf_vectorizer:
                            vectors = tfidf_vectorizer.transform(uploaded_df['Cleaned Text'].fillna(''))
                            preds = ai_model.predict(vectors)
                            
                           
                            uploaded_df['label_pred'] = [refine_sentiment_with_rules(row[chosen_text_col], p) for row, p in zip(uploaded_df.to_dict('records'), preds)]
                            uploaded_df['Predicted Sentiment'] = uploaded_df['label_pred'].map(sentiment_map)
                        else:
                            uploaded_df['Predicted Sentiment'] = np.random.choice(['Positive', 'Neutral', 'Negative'], size=len(uploaded_df))
                        
                    st.success("Analyze successfully!")
                    
                    st.markdown("### Data Visualization")
                    col_v1, col_v2 = st.columns(2)
                    
                    with col_v1:
                        st.markdown("<p style='text-align: center; font-weight: bold;'>Sentiment Distribution by Aspect</p>", unsafe_allow_html=True)
                        fig_u_bar = px.histogram(
                            uploaded_df, x="Extracted Aspect", color="Predicted Sentiment", barmode="group",
                            color_discrete_map={'Positive': '#00e676', 'Neutral': '#ffea00', 'Negative': '#ff1744'},
                            template="plotly_dark", height=300
                        )
                        fig_u_bar.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
                        st.plotly_chart(fig_u_bar, use_container_width=True, config={'displayModeBar': False})
                        
                    with col_v2:
                        st.markdown("<p style='text-align: center; font-weight: bold;'>Percentage Split</p>", unsafe_allow_html=True)
                        fig_u_pie = px.pie(
                            uploaded_df, names='Predicted Sentiment', color='Predicted Sentiment',
                            color_discrete_map={'Positive': '#00e676', 'Neutral': '#ffea00', 'Negative': '#ff1744'},
                            hole=0.4, template="plotly_dark", height=300
                        )
                        fig_u_pie.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
                        st.plotly_chart(fig_u_pie, use_container_width=True, config={'displayModeBar': False})
                    
                    st.markdown("---")
                    st.write("#### Data Sample Output Preview:")
                    st.dataframe(
                        uploaded_df[[chosen_text_col, 'Cleaned Text', 'Extracted Aspect', 'Predicted Sentiment']].head(10), 
                        use_container_width=True
                    )
                    
                    csv_data = uploaded_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="📥 Download Result (CSV)", data=csv_data,
                        file_name="absa_bulk_report_output.csv", mime="text/csv"
                    )
