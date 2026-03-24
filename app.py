from sentence_transformers import SentenceTransformer
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
def build_bert_model():
    model = SentenceTransformer("all-MiniLM-L6-v2")
    return model

@st.cache_data
def build_embeddings(_model, abstracts):
    return _model.encode(abstracts, show_progress_bar=True, batch_size=256)

df = load_data()
bert_model = build_bert_model()
bert_embeddings = build_embeddings(bert_model, df["abstract"].tolist())

st.title("Research Paper Recommendation System")
st.write("Search across 100,000+ arXiv research papers using BERT semantic search")

query = st.text_input("Enter your research interest:")

if query:
    query_embedding = bert_model.encode([query])
    scores = cosine_similarity(query_embedding, bert_embeddings).flatten()
    top_indices = scores.argsort()[-5:][::-1]

    st.subheader(f"Top 5 results for: '{query}'")
    for i in top_indices:
        st.subheader(df["title"][i])
        st.write(f"**Score:** {scores[i]:.4f} | **Category:** {df['categories'][i]}")
        st.write(df["abstract"][i][:300])
        st.divider()
