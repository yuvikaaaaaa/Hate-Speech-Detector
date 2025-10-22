import streamlit as st
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

@st.cache_resource
def download_nltk_data():
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    return True

download_nltk_data()

@st.cache_resource
def load_models():
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
    return model, vectorizer

model, vectorizer = load_models()

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    words = text.split()
    words = [lemmatizer.lemmatize(w) for w in words if w not in stop_words]
    return ' '.join(words)

st.title("Hate Speech Detector")
st.write("Detect offensive content using Machine Learning")

user_input = st.text_area("Enter text to analyze:", height=150)

if st.button("Analyze"):
    if user_input:
        processed = preprocess_text(user_input)
        vectorized = vectorizer.transform([processed])
        prediction = model.predict(vectorized)[0]
        probability = model.predict_proba(vectorized)[0]
        
        if prediction == 1:
            st.error("Offensive Content Detected")
            confidence = probability[1] * 100
        else:
            st.success("Non-Offensive Content")
            confidence = probability[0] * 100
        
        st.metric("Confidence", f"{confidence:.1f}%")
    else:
        st.warning("Please enter text")