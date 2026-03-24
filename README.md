# Research Paper Recommendation System

> Built an end-to-end NLP recommendation engine over **100K+ arXiv research papers** (sampled from 3M+), benchmarking 4 approaches from TF-IDF to BERT Sentence Transformers — achieving **91% domain accuracy** with semantic embeddings.

## What it does

Enter a research interest (e.g. "deep learning for natural language processing") and the system returns the top 5 most relevant research papers from 100,000+ arXiv papers — ranked by similarity score.

## How it works

1. **Text Processing** — abstracts are cleaned (lowercased, stripped, newlines removed)
2. **Vectorization** — TF-IDF and CountVectorizer convert text into numerical vectors with bigrams and stop word removal
3. **BERT Embeddings** — Sentence Transformers encode abstracts into 384-dimensional semantic vectors
4. **Similarity** — cosine similarity between the query embedding and all paper embeddings

## Results

| Approach | Category Accuracy | Domain Accuracy |
|----------|------------------|-----------------|
| TF-IDF | 0.10 | 0.15 |
| CountVectorizer | 0.49 | 0.77 |
| Mixed | 0.50 | 0.78 |
| **BERT (best)** | **0.73** | **0.91** |

## Resume

- Built an end-to-end NLP recommendation engine over **100K+ arXiv research papers** (sampled from 3M+), using TF-IDF, CountVectorizer, and BERT Sentence Transformers to surface topic-relevant papers from a user's search query.
- Benchmarked 4 approaches (TF-IDF, CountVectorizer, Mixed, BERT) — BERT achieved **91% domain accuracy**, a **6x improvement** over TF-IDF baseline, by leveraging semantic embeddings instead of word frequency.
- Designed a multi-label evaluation framework using **set intersection across 175 arXiv categories**, enabling quantitative comparison across model configurations.
- Deployed a **Streamlit web interface** allowing users to input research interests and receive ranked recommendations with similarity scores and abstracts in real time.

## Tech Stack

- Python, Pandas, Scikit-learn, Sentence Transformers, Streamlit
- Dataset: arXiv research papers (3M+ papers)

## How to run

```bash
pip install pandas scikit-learn streamlit sentence-transformers
streamlit run app.py
```
