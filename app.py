import streamlit as st
import pickle
import re
import os
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Download NLTK data
@st.cache_resource
def download_nltk_data():
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('punkt', quiet=True)
    return True

download_nltk_data()

# Train model if not exists
@st.cache_resource
def load_or_train_model():
    # Check if model exists
    if os.path.exists('model.pkl') and os.path.exists('vectorizer.pkl'):
        with open('model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('vectorizer.pkl', 'rb') as f:
            vectorizer = pickle.load(f)
        return model, vectorizer
    
    # Train model
    st.info("Training model for first time... This will take 2-3 minutes.")
    
    # Sample training data (replace with your data.csv if you uploaded it)
    data = {
        'tweet': [
            'I love everyone here', 'You people are trash', 'Having a great day',
            'I hate you so much', 'This community is wonderful', 'Go kill yourself',
            'Thanks for the help', 'You are an idiot', 'Beautiful weather today',
            'Stupid loser', 'Great work team', 'You should die',
            'Happy birthday friend', 'Worthless piece of trash', 'Excellent presentation',
            'I hope you suffer', 'Nice to meet you', 'You disgust me',
            'Congratulations on your success', 'Pathetic human being'
        ] * 50,  # Repeat to have more data
        'class': [2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0] * 50
    }
    
    df = pd.DataFrame(data)
    df['label'] = df['class'].apply(lambda x: 0 if x == 2 else 1)
    
    # Preprocessing
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    
    def preprocess(text):
        text = text.lower()
        text = re.sub(r'http\S+', '', text)
        text = re.sub(r'@\w+', '', text)
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        words = text.split()
        words = [lemmatizer.lemmatize(w) for w in words if w not in stop_words]
        return ' '.join(words)
    
    texts = [preprocess(t) for t in df['tweet'].values]
    labels = df['label'].values
    
    X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2)
    
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X_train_vec = vectorizer.fit_transform(X_train)
    
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_vec, y_train)
    
    # Save model
    with open('model.pkl', 'wb') as f:
        pickle.dump(model, f)
    with open('vectorizer.pkl', 'wb') as f:
        pickle.dump(vectorizer, f)
    
    st.success("Model trained successfully!")
    return model, vectorizer

model, vectorizer = load_or_train_model()

# Preprocessing function
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

# UI
st.title("🛡️ Hate Speech Detector")
st.markdown("### Detect offensive and hateful content using Machine Learning")

with st.sidebar:
    st.header("ℹ️ About")
    st.write("""
    Uses NLP and ML to detect hate speech.
    
    **Tech Stack:**
    - Python, NLTK
    - TF-IDF + Logistic Regression
    - Streamlit
    """)

user_input = st.text_area("Enter text to analyze:", height=150, 
                          placeholder="Example: This is a sample text...")

if st.button("🔍 Analyze Text", type="primary"):
    if user_input:
        with st.spinner("Analyzing..."):
            processed = preprocess_text(user_input)
            vectorized = vectorizer.transform([processed])
            prediction = model.predict(vectorized)[0]
            probability = model.predict_proba(vectorized)[0]
            
            st.markdown("---")
            col1, col2 = st.columns(2)
            
            with col1:
                if prediction == 1:
                    st.error("⚠️ **Offensive Content Detected**")
                    confidence = probability[1] * 100
                else:
                    st.success("✅ **Non-Offensive Content**")
                    confidence = probability[0] * 100
                
                st.metric("Confidence Score", f"{confidence:.1f}%")
            
            with col2:
                st.write("**Probability:**")
                st.write(f"Non-Offensive: {probability[0]*100:.1f}%")
                st.write(f"Offensive: {probability[1]*100:.1f}%")
    else:
        st.warning("⚠️ Please enter some text")

st.markdown("---")
st.markdown("<div style='text-align: center; color: gray;'>Built with Python, NLTK, Scikit-learn, Streamlit</div>", 
            unsafe_allow_html=True)
