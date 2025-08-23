import streamlit as st
from sentence_transformers import SentenceTransformer

# Cache for embeddings
@st.cache_resource
def get_embedding_cache():
    return {}

embedding_cache = get_embedding_cache()

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
def emb_text(embedding_model, text: str):
    if text in embedding_cache:
        return embedding_cache[text]
    else:
        embedding = embedding_model.encode(text)
        embedding_cache[text] = embedding
        return embedding
    
if __name__ == '__main__':
    sample_1 = "this is a test"
    embedding = emb_text(sample_1)
    print(embedding)
