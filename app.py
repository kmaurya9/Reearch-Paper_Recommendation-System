import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

@st.cache_data
def load_data():
    df = pd.read_json("/Users/kshitij/ML project/Dataset.json", lines=True)
    df = df.sample(n=100000, random_state=42)
    df = df.reset_index(drop=True)
    df = df[["title", "abstract", "categories"]]
    df["abstract"] = df["abstract"].str.replace("\n", " ")
    df["abstract"] = df["abstract"].str.strip()
    df["abstract"] = df["abstract"].str.lower()
    return df

@st.cache_resource
def build_models(df):
    tfidf_vec = TfidfVectorizer(max_features=5000, stop_words="english", ngram_range=(1, 2))
    count_vec = CountVectorizer(max_features=5000, stop_words="english", ngram_range=(1, 2))
    tfidf_matrix = tfidf_vec.fit_transform(df["abstract"])
    count_matrix = count_vec.fit_transform(df["abstract"])
    return tfidf_vec, count_vec, tfidf_matrix, count_matrix

df = load_data()
tfidf_vec, count_vec, tfidf_matrix, count_matrix = build_models(df)

st.title("Research Paper Recommendation System")
st.write("Search across 100,000+ arXiv research papers")

query = st.text_input("Enter your research interest:")

if query:
    tfidf_scores = cosine_similarity(tfidf_vec.transform([query]), tfidf_matrix).flatten()
    count_scores = cosine_similarity(count_vec.transform([query]), count_matrix).flatten()
    mixed_scores = (tfidf_scores + count_scores) / 2

    top_indices = mixed_scores.argsort()[-5:][::-1]

    st.subheader(f"Top 5 results for: '{query}'")
    for i in top_indices:
        st.subheader(df["title"][i])
        st.write(f"**Score:** {mixed_scores[i]:.4f} | **Category:** {df['categories'][i]}")
        st.write(df["abstract"][i][:300])
        st.divider()
