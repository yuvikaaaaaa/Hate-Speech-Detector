# Hate Speech Detector

ML system to detect hate speech in text using NLP and Machine Learning.

## Features
- Text preprocessing with NLTK
- TF-IDF vectorization
- Multiple ML models (Logistic Regression, Naive Bayes, Random Forest)
- Real-time detection via Streamlit

## Live Demo
[Deploying to Streamlit Cloud - Link coming soon]

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
```

   - Click **"Commit changes"** (green button)

---

## **Step 2: Deploy on Streamlit Cloud - 15 mins**

### **A) Sign Up:**

1. Go to **https://share.streamlit.io**
2. Click **"Sign in"**
3. Choose **"Continue with GitHub"**
4. **Authorize Streamlit** to access your GitHub

### **B) Deploy Your App:**

1. Click **"New app"** (big button)

2. **Fill in the form:**
   - **Repository:** Select `your-username/hate-speech-detector`
   - **Branch:** `main`
   - **Main file path:** `app.py`

3. Click **"Deploy!"** (bottom right)

4. **Wait 5-10 minutes** - Watch the logs, it will:
   - Install packages
   - Download NLTK data
   - Start your app

5. **You'll get a URL** like:
```
   https://your-username-hate-speech-detector.streamlit.app
```

6. **COPY THIS URL!**

---

## **Step 3: Update README with Live Link - 3 mins**

1. Go back to your GitHub repo
2. Click on `README.md`
3. Click **pencil icon** (edit)
4. Change this line:
```
   [Deploying to Streamlit Cloud - Link coming soon]
```
   To:
```
   [🔗 Live Demo](PASTE-YOUR-STREAMLIT-URL-HERE)
```
5. Click **"Commit changes"**

---

## **Step 4: Update Your Resume - 5 mins**

**Open your resume document and add:**
```
§ Hate Speech Detector (Live Demo | GitHub)
- Built NLP-based text classification system using Python to detect hate speech 
  achieving 85%+ accuracy on labeled tweet dataset
- Implemented text preprocessing pipeline (tokenization, lemmatization) and 
  TF-IDF vectorization with bigram features
- Trained and compared 3 ML models (Logistic Regression, Naive Bayes, Random 
  Forest) selecting best performer
- Deployed interactive Streamlit web application with real-time prediction and 
  confidence scoring
