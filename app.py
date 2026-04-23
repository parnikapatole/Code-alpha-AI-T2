import streamlit as st
import json
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --- VERCEL COMPATIBILITY ---
def handler(event, context):
    return {'statusCode': 200, 'body': 'Streamlit is running.'}

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(page_title="AI Seismic FAQ", layout="centered")

with st.sidebar:
    st.header("📊 Project Details")
    st.markdown("**Framework:** MASF AI\n**Developer:** Parnika")

# --- 2. UI STYLING ---
st.markdown("""
    <style>
    .stApp { background-color: #121212; color: #E0E0E0; }
    .stTextInput > div > div > input { 
        background-color: #1E1E1E; color: #D4AF37; border: 1px solid #D4AF37; }
    .answer-box { border-left: 4px solid #D4AF37; padding: 20px; background-color: #1A1A1A; color: #C0C0C0; }
    </style>
""", unsafe_allow_html=True)

st.title("🏛️ AI Project Assistant")

# --- 3. DATA LOADING ---
json_path = os.path.join(os.path.dirname(__file__), 'faqs.json')

if not os.path.exists(json_path):
    st.error(f"❌ ERROR: 'faqs.json' not found at {json_path}")
else:
    with open(json_path, 'r') as f:
        data = json.load(f)
    questions = [item['question'] for item in data]
    answers = [item['answer'] for item in data]

    # --- 4. NLP ENGINE ---
    @st.cache_resource
    def load_nlp_model(qs):
        vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))
        matrix = vectorizer.fit_transform(qs)
        return vectorizer, matrix

    vectorizer, tfidf_matrix = load_nlp_model(questions)

    user_query = st.text_input("Enter your query here...")
    if user_query:
        query_vec = vectorizer.transform([user_query])
        similarity = cosine_similarity(query_vec, tfidf_matrix)
        match_idx = similarity.argmax()
        score = similarity[0][match_idx]

        if score > 0.25:
            st.markdown(f"<div class='answer-box'><b>Answer:</b><br>{answers[match_idx]}</div>", unsafe_allow_html=True)
            st.progress(float(score))
        else:
            st.warning("Confidence too low. Please rephrase.")

app = handler
