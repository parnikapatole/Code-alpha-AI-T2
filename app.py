app = None  # Dummy handler for Vercel deployment

import streamlit as st
import json
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --- 1. PAGE CONFIGURATION & SIDEBAR ---
st.set_page_config(page_title="AI Seismic FAQ", layout="centered")

with st.sidebar:
    st.header("📊 Project Details")
    st.markdown("""
    **Framework:** MASF AI  
    **Developer:** Parnika  
    **Status:** Active Research  
    """)
    st.info("Uses TF-IDF Vectorization & Cosine Similarity for high-precision matching.")
    st.subheader("Methodology")
    st.caption("- NLP Preprocessing\n- N-gram Analysis\n- Similarity Thresholding")

# --- 2. HIGH-END UI STYLING (Earth-Tone & Gold) ---
st.markdown("""
    <style>
    .stApp { background-color: #121212; color: #E0E0E0; }
    .stTextInput > div > div > input { 
        background-color: #1E1E1E; color: #D4AF37; border: 1px solid #D4AF37; border-radius: 0px; 
    }
    .answer-box { 
        border-left: 4px solid #D4AF37; 
        padding: 20px; 
        background-color: #1A1A1A; 
        margin-top: 20px; 
        font-size: 18px;
        color: #C0C0C0;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🏛️ AI Project Assistant")
st.write("Precision-focused FAQ system for seismic risk research.")

# --- 3. KNOWLEDGE BASE LOADING ---
if not os.path.exists('faqs.json'):
    st.error("❌ ERROR: 'faqs.json' file not found!")
else:
    try:
        with open('faqs.json', 'r') as f:
            data = json.load(f)
        questions = [item['question'] for item in data]
        answers = [item['answer'] for item in data]

        # --- 4. NLP ENGINE ---
        vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))
        tfidf_matrix = vectorizer.fit_transform(questions)

        # --- 5. USER INTERFACE ---
        user_query = st.text_input("Enter your query here...")

        if user_query:
            query_vec = vectorizer.transform([user_query])
            similarity = cosine_similarity(query_vec, tfidf_matrix)
            
            match_index = similarity.argmax()
            score = similarity[0][match_index]

            if score > 0.25:
                st.markdown(f"<div class='answer-box'><b>Answer:</b><br>{answers[match_index]}</div>", unsafe_allow_html=True)
                st.write(f"Confidence Level: {int(score*100)}%")
                st.progress(float(score))
            else:
                st.warning("Confidence too low. Please rephrase your question.")
                
    except json.JSONDecodeError:
        st.error("❌ ERROR: Your 'faqs.json' has a formatting error.")
    except Exception as e:
        st.error(f"❌ AN UNKNOWN ERROR OCCURRED: {e}")
        # This is the "hook" Vercel needs to run the app
app = None