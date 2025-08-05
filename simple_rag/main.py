import os
from fastapi import FastAPI, UploadFile, File, HTTPException
from typing import List, Union
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse
import shutil
from pathlib import Path

from llama_index.core import SimpleDirectoryReader, StorageContext, VectorStoreIndex
from llama_index.core.node_parser import SentenceWindowNodeParser
from llama_index.core.postprocessor import (
    MetadataReplacementPostProcessor,
    SentenceTransformerRerank,
)
from llama_index.core.settings import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.llms.openai import OpenAI
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
import torch

from dotenv import load_dotenv
load_dotenv()

torch.classes.__path__ = [] # add this line to manually set it to empty.

def initialize_llm():
    # Settings.llm = Ollama("qwen3:4b", request_timeout=30.0)
    Settings.llm = OpenAI(model="gpt-4o-mini")
    embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5",
                                       device="cuda",
                                        backend="openvino",  # OpenVINO is very strong on CPUs
                                        revision="refs/pr/16",  # BAAI/bge-small-en-v1.5 itself doesn't have an OpenVINO model currently, but there's a PR with it that we can load: https://huggingface.co/BAAI/bge-small-en-v1.5/discussions/16
                                        model_kwargs={
                                            "file_name": "openvino_model_qint8_quantized.xml"
                                        }
                                       )
    # embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
    Settings.embed_model = embed_model


def create_index(documents_paths, prefix):
    documents = SimpleDirectoryReader(input_files=documents_paths).load_data()
    node_parser = SentenceWindowNodeParser.from_defaults(
        window_size=3, 
        window_metadata_key="window",
        original_text_metadata_key="original_text"
    )

    nodes = node_parser.get_nodes_from_documents(documents=documents)

    client = QdrantClient(url="http://localhost:6333")
    collection_name = prefix

    vector_store = QdrantVectorStore(
        client=client,
        collection_name=collection_name,
        enable_hybrid=True
    )

    store_context = StorageContext.from_defaults(vector_store=vector_store)

    index = VectorStoreIndex(
        nodes,
        storage_context=store_context
    )

    return index

def query_index(index, query):
    query_engine = index.as_query_engine(
        similarity_top_k=6,
        llm=Settings.llm,
        node_postprocessors=[
            MetadataReplacementPostProcessor(target_metadata_key="window"),
            SentenceTransformerRerank(top_n=2, model="BAAI/bge-reranker-base"),
        ],
        vector_store_query_mode="hybrid",
        alpha=0.5,
    )

    response = query_engine.query(query)
    return response

class ChatEngine:
    def __init__(self, index) -> None:
        self.index = index
        self.conversation_history = []

    def chat(self, query):
        self.conversation_history.append({"role": "user", "content": query})
        response = query_index(self.index, query)
        self.conversation_history.append({"role": "assistant", "content": response})
        return response
    
initialize_llm()

chat_alls = dict()

app = FastAPI()
@app.post("/uploadfile/")
async def create_upload_file(file: UploadFile = File(...)):
    """
    Uploads a single file and saves it to the server.
    """
    try:
        uploaded_file_name = str(file.filename)
        prefix = uploaded_file_name.split(".")[0]
        os.makedirs(prefix, exist_ok=True)
        file_location = Path(prefix) / file.filename
        with open(file_location, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        document_paths = [file_location]
        index = create_index(document_paths, prefix)
        chat_engine = ChatEngine(index)
        chat_alls[prefix] = chat_engine
        with open(file_location, "wb") as buffer:
            # Using shutil.copyfileobj for efficient streaming
            # It reads chunks from the UploadFile.file object and writes them
            # to the buffer, avoiding loading the entire file into memory at once.
            shutil.copyfileobj(file.file, buffer)
        return {"filename": file.filename, "message": f"File '{file.filename}' uploaded successfully to {file_location}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not upload file: {e}")
    finally:
        await file.close() # Ensure the uploaded file is closed

@app.get("/query/")
async def query_rag_system(video_name_prefix: str, query: str):
    try:
        # Call the RAG system to process the query and get a response
        response = chat_alls[video_name_prefix].chat(query)
        response_str = response.response
        return {"query": query, "response": response_str}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))