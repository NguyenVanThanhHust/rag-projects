import os
import streamlit as st

st.set_page_config(layout="wide")

from milvus_utils import get_milvus_client, get_search_results
from all_llm import get_llm_answer
from encoder import embedding_model, emb_text
from dotenv import load_dotenv

load_dotenv()
COLLECTION_NAME = os.getenv("COLLECTION_NAME")
MILVUS_ENDPOINT = os.getenv("MILVUS_ENDPOINT")
MILVUS_TOKEN = os.getenv("MILVUS_TOKEN")

# Logo
st.markdown(
    """
    <style>
    .title {
        text-align: center;
        font-size: 50px;
        font-weight: bold;
        margin-bottom: 20px;
    }
    .description {
        text-align: center;
        font-size: 18px;
        color: gray;
        margin-bottom: 40px;
    }
    </style>
    <div class="title">RAG Demo</div>
    <div class="description">
        This chatbot is built with Milvus vector database, supported by OpenAI text embedding model.<br>
        It supports conversation based on knowledge from the Milvus development guide document.
    </div>
    """,
    unsafe_allow_html=True,
)

# Get clients
milvus_client = get_milvus_client(uri=MILVUS_ENDPOINT)

from openai import OpenAI

# Initialize the client with the local Ollama server URL
client = OpenAI(
    base_url='http://localhost:11434/v1',
    api_key='ollama',  # A placeholder key is required but not used
)

# Now, use the client as you normally would
response = client.chat.completions.create(
    model="qwen3:4b",  # Use the name of the model you pulled
    messages=[
        {"role": "user", "content": "Tell me about replacing OpenAI with local models."}
    ]
)

retrieved_lines_with_distances = []

with st.form("my_form"):
    question = st.text_area("Enter your question:")
    # Sample question: what is the hardware requirements specification if I want to build Milvus and run from source code?
    submitted = st.form_submit_button("Submit")

    if question and submitted:
        # Generate query embedding
        query_vector = emb_text(embedding_model, question)
        # Search in Milvus collection
        search_res = get_search_results(
            milvus_client, COLLECTION_NAME, query_vector, ["text"]
        )

        # Retrieve lines and distances
        retrieved_lines_with_distances = [
            (res["entity"]["text"], res["distance"]) for res in search_res[0]
        ]

        # Create context from retrieved lines
        context = "\n".join(
            [
                line_with_distance[0]
                for line_with_distance in retrieved_lines_with_distances
            ]
        )
        answer = get_llm_answer(ll, context, question)

        # Display the question and response in a chatbot-style box
        st.chat_message("user").write(question)
        st.chat_message("assistant").write(answer)


# Display the retrieved lines in a more readable format
st.sidebar.subheader("Retrieved Lines with Distances:")
for idx, (line, distance) in enumerate(retrieved_lines_with_distances, 1):
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"**Result {idx}:**")
    st.sidebar.markdown(f"> {line}")
    st.sidebar.markdown(f"*Distance: {distance:.2f}*")