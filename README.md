# SafeSpeech AI

ML system to detect hate speech in text using NLP and Machine Learning.

## Features
- Text preprocessing with NLTK
- TF-IDF vectorization
- Multiple ML models (Logistic Regression, Naive Bayes, Random Forest)
- Real-time detection via Streamlit

## Tech Stack
Python • NLTK • Scikit-learn • Streamlit • Pandas

## How to Run
```bash
pip install -r requirements.txt
python train_model.py
streamlit run app.py
```

## Model Performance
- Trained on labeled tweet dataset
- 85%+ accuracy
- TF-IDF with bigrams for feature extraction


   - Click **"Commit changes"** (green button)

---

## **Step 2: Deploy on Streamlit Cloud - 15 mins**

### **A) Sign Up:**

1. Go to **https://share.streamlit.io**
2. Click **"Sign in"**
3. Choose **"Continue with GitHub"**
4. **Authorize Streamlit** to access your GitHub

§ Hate Speech Detector (Live Demo | GitHub)
- Built NLP-based text classification system using Python to detect hate speech 
  achieving 85%+ accuracy on labeled tweet dataset
- Implemented text preprocessing pipeline (tokenization, lemmatization) and 
  TF-IDF vectorization with bigram features
- Trained and compared 3 ML models (Logistic Regression, Naive Bayes, Random 
  Forest) selecting best performer
- Deployed interactive Streamlit web application with real-time prediction and 
  confidence scoring
