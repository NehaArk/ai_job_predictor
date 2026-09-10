import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="AI Job Experience Level Predictor", layout="wide")
st.title("AI Job Salaries — Experience Level Predictor")
st.write("Upload a CSV with a `job_title` column to get experience-level predictions.")

@st.cache_resource
def load_model():
    clf = joblib.load("experience_level_classifier.pkl")
    vectorizer = joblib.load("tfidf_vectorizer.pkl")
    return clf, vectorizer

try:
    clf, vectorizer = load_model()
    model_loaded = True
except FileNotFoundError:
    model_loaded = False
    st.error("Model files not found.")

uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader("Preview")
    st.dataframe(df.head())

    if "job_title" not in df.columns:
        st.error("CSV needs a `job_title` column.")
    elif model_loaded:
        X_tfidf = vectorizer.transform(df["job_title"].astype(str))
        df["predicted_experience_level"] = clf.predict(X_tfidf)
        df["prediction_confidence"] = clf.predict_proba(X_tfidf).max(axis=1).round(3)

        st.subheader("Predictions")
        st.dataframe(df[["job_title", "predicted_experience_level", "prediction_confidence"]])
        st.bar_chart(df["predicted_experience_level"].value_counts())

        csv_out = df.to_csv(index=False).encode("utf-8")
        st.download_button("Download predictions", csv_out, "predictions.csv", "text/csv")
else:
    st.info("Waiting for a CSV upload.")
