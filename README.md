# 🏛️ AI Seismic FAQ Assistant
**Lead Developer:** [Your Name]  
**Project Goal:** Technical support bot for the Multi-Modal AI Seismic Forecasting (MASF) Framework.

## 🚀 Overview
This is a high-end NLP chatbot designed to provide instant answers regarding seismic risk probability. It uses mathematical vectorization to understand user intent rather than simple keyword matching.

## 🧠 How it Works
1. **Data:** Knowledge is stored in `faqs.json`.
2. **NLP Engine:** Uses **TF-IDF Vectorization** to convert text into numbers.
3. **Matching:** Uses **Cosine Similarity** to find the closest match between the user's question and the database.
4. **UI:** Built with **Streamlit** using a professional Earth-Tone & Gold theme.

## 🛠️ Setup Instructions
1. Activate virtual environment: `.\venv\Scripts\Activate.ps1`
2. Install dependencies: `pip install -r requirements.txt`
3. Run the app: `streamlit run app.py`

## 📈 Performance Metric
The system includes a **Confidence Gauge**. If the similarity score is below 25%, the bot will refuse to answer to ensure high precision and avoid misinformation.