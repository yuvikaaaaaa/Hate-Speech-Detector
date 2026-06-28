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
