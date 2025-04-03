import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import time

# Set page config
st.set_page_config(
    page_title="SpamSensei",
    page_icon="🛡️",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Download NLTK data
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

# Initialize Porter Stemmer
ps = PorterStemmer()

# Preprocessing function
def transform_text(text):
    if not text.strip():
        return ""
    text = text.lower()
    text = nltk.word_tokenize(text)
    y = [i for i in text if i.isalnum()]
    y = [i for i in y if i not in stopwords.words('english') and i not in string.punctuation]
    y = [ps.stem(i) for i in y]
    return " ".join(y)

# Load models with caching
@st.cache_resource
def load_models():
    try:
        tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
        model = pickle.load(open('model.pkl', 'rb'))
        return tfidf, model
    except Exception as e:
        st.error(f"🚨 Failed to load models: {e}. Check 'vectorizer.pkl' and 'model.pkl'.")
        return None, None

tfidf, model = load_models()

# Custom CSS with enhanced result styling
st.markdown("""
    <style>
    .title { 
        font-size: 3em; 
        text-align: center; 
        font-weight: bold; 
        background: linear-gradient(90deg, #2c3e50, #3498db); 
        -webkit-background-clip: text; 
        color: transparent; 
    }
    .subtitle { 
        color: #7f8c8d; 
        text-align: center; 
        font-style: italic; 
        margin-bottom: 20px; 
    }
    .stButton>button { 
        background: linear-gradient(90deg, #2980b9, #3498db); 
        color: white; 
        border-radius: 12px; 
        padding: 12px 24px; 
        font-size: 16px; 
        border: none; 
        transition: transform 0.2s, box-shadow 0.2s; 
    }
    .stButton>button:hover { 
        transform: scale(1.05); 
        box-shadow: 0 4px 12px rgba(0,0,0,0.2); 
    }
    .stTextArea textarea { 
        border-radius: 15px; 
        padding: 15px; 
        font-size: 14px; 
        border: 2px solid #dcdcdc; 
        transition: border-color 0.3s; 
    }
    .stTextArea textarea:focus { 
        border-color: #3498db; 
    }
    .result-card { 
        padding: 20px 30px; 
        border-radius: 15px; 
        text-align: center; 
        margin-top: 25px; 
        box-shadow: 0 8px 16px rgba(0,0,0,0.15); 
        animation: slideUp 0.5s ease-out; 
        border-left: 6px solid; 
    }
    .result-header { 
        font-size: 1.6em; 
        font-weight: bold; 
        margin-bottom: 10px; 
    }
    .result-detail { 
        font-size: 1.2em; 
        color: #555; 
    }
    @keyframes slideUp { 
        from { opacity: 0; transform: translateY(30px); } 
        to { opacity: 1; transform: translateY(0); } 
    }
    .footer { 
        text-align: center; 
        color: #95a5a6; 
        font-size: 12px; 
        margin-top: 50px; 
    }
    </style>
""", unsafe_allow_html=True)

# Theme Switcher in Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    theme = st.selectbox("Theme", ["Light", "Dark"], index=0)
    if theme == "Dark":
        st.markdown("""
            <style>
            body { background-color: #2c3e50; color: #ecf0f1; }
            .stTextArea textarea { background-color: #34495e; color: #ecf0f1; border-color: #7f8c8d; }
            .subtitle, .footer { color: #bdc3c7; }
            .result-detail { color: #bdc3c7; }
            </style>
        """, unsafe_allow_html=True)

    st.header("ℹ️ About SpamSensei")
    st.write("""
        SpamSensei uses cutting-edge to protect you from spam and scams. 
       
    """)
    st.info("Created by Aman,Sangam,Piyush,Aditya | 2025")

# Header Section
st.markdown('<h1 class="title">🛡️ SpamSensei</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Your ultimate  guardian against unwanted messages</p>', unsafe_allow_html=True)

# Input Section
st.markdown("### 📬 Analyze Your Message")
input_sms = st.text_area(
    "",
    placeholder="Enter your SMS or email (e.g., 'Claim your prize now!')",
    height=150,
    help="Paste any message to scan for spam."
)

# Predict Button and Confidence Meter
col1, col2 = st.columns([2, 1])
with col1:
    predict_button = st.button('🔍 Scan Now')
with col2:
    st.write("")  # Spacer

if predict_button:
    if not input_sms.strip():
        st.warning("⚠️ Please enter a message to scan.")
    elif tfidf is None or model is None:
        st.error("🚨 Model loading failed.")
    else:
        with st.spinner("🔄 Analyzing..."):
            time.sleep(1)  # Simulate processing
            transformed_sms = transform_text(input_sms)
            vector_input = tfidf.transform([transformed_sms])
            result = model.predict(vector_input)[0]
            proba = model.predict_proba(vector_input)[0][1]  # Spam probability

            # Confidence Meter
            st.markdown("#### Confidence Level")
            st.progress(proba)
            st.caption(f"Spam Likelihood: {proba:.1%}")

            # Enhanced Result Display
            if result == 1:
                st.markdown(
                    f'<div class="result-card" style="background: linear-gradient(135deg, #fadbd8, #f5b7b1); border-left-color: #e74c3c;">'
                    f'<div class="result-header">🚨 Spam Detected!</div>'
                    f'<div class="result-detail">Confidence: {proba:.1f}%</div>'
                    f'<div class="result-detail">This message exhibits spam-like patterns.</div>'
                    f'</div>',
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f'<div class="result-card" style="background: linear-gradient(135deg, #d5f5e3, #a9dfbf); border-left-color: #27ae60;">'
                    f'<div class="result-header">✅ Safe Message</div>'
                    f'<div class="result-detail">Spam Risk: {proba:.1f}%</div>'
                    f'<div class="result-detail">This message appears legitimate.</div>'
                    f'</div>',
                    unsafe_allow_html=True
                )

            # Add to History
            if 'history' not in st.session_state:
                st.session_state.history = []
            st.session_state.history.append({
                'message': input_sms[:50] + ("..." if len(input_sms) > 50 else ""),
                'result': "Spam" if result == 1 else "Safe",
                'confidence': proba
            })

# History Section
if 'history' in st.session_state and st.session_state.history:
    with st.expander("📜 Scan History", expanded=False):
        for entry in reversed(st.session_state.history[-5:]):  # Show last 5 scans
            st.write(f"**Message**: {entry['message']}")
            st.write(f"**Result**: {entry['result']} (Confidence: {entry['confidence']:.1f}%)")
            st.divider()

# Footer
st.markdown('<p class="footer">Powered by AMAN | © 2025 SpamSensei</p>', unsafe_allow_html=True)