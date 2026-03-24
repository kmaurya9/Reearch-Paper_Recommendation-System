# Research Paper Recommendation System

> Built an end-to-end NLP recommendation engine over **100K+ arXiv research papers** (sampled from 3M+), using TF-IDF and CountVectorizer with bigram extraction and stop word filtering to surface topic-relevant papers from a user's search query.

## What it does

Enter a research interest (e.g. "deep learning for natural language processing") and the system returns the top 5 most relevant research papers from 100,000+ arXiv papers — ranked by similarity score.

## How it works

1. **Text Processing** — abstracts are cleaned (lowercased, stripped, newlines removed)
2. **Vectorization** — TF-IDF and CountVectorizer convert text into numerical vectors with bigrams and stop word removal
3. **Similarity** — cosine similarity between the query vector and all paper vectors
4. **Mixed Model** — scores from both vectorizers are averaged for best results

## Results

| Approach | Category Accuracy | Domain Accuracy |
|----------|------------------|-----------------|
| TF-IDF | 0.10 | 0.15 |
| CountVectorizer | 0.49 | 0.77 |
| **Mixed (best)** | **0.50** | **0.78** |

## Resume

- Built an end-to-end NLP recommendation engine over **100K+ arXiv research papers** (sampled from 3M+), using TF-IDF and CountVectorizer with bigram extraction and stop word filtering to surface topic-relevant papers from a user's search query.
- Compared 3 vectorization approaches (TF-IDF, CountVectorizer, Mixed) — Mixed model achieved **78% domain accuracy and 50% category accuracy**, outperforming TF-IDF alone (15% domain accuracy).
- Designed a multi-label evaluation framework using **set intersection across 175 arXiv categories**, enabling quantitative comparison across model configurations.
- Deployed a **Streamlit web interface** allowing users to input research interests and receive ranked recommendations with similarity scores and abstracts in real time.

## Tech Stack

- Python, Pandas, Scikit-learn, Streamlit
- Dataset: arXiv research papers (3M+ papers)

## How to run

```bash
pip install pandas scikit-learn streamlit
streamlit run app.py
```
