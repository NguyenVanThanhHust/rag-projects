from glob import glob
from sentence_transformers import SentenceTransformer
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

text_lines = []

for file_path in glob("milvus_docs/en/faq/*.md", recursive=True):
    with open(file_path, "r") as file:
        file_text = file.read()

    text_lines += file_text.split("# ")


embedding_cache = dict()
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

def emb_text(embedding_model, text: str):
    if text in embedding_cache:
        return embedding_cache[text]
    else:
        embedding = embedding_model.encode(text)
        embedding_cache[text] = embedding
        return embedding
test_embedding = emb_text(embedding_model, "This is a test")
embedding_dim = len(test_embedding)
print(embedding_dim)
print(test_embedding[:10])

# Load data into Milvus
from pymilvus import MilvusClient
milvus_client = MilvusClient(uri="./milvus_demo.db")
collection_name = "my_rag_collection"

# check if the collection is existing and drop it if it does
if milvus_client.has_collection(collection_name):
    milvus_client.drop_collection(collection_name)

# create the collection
milvus_client.create_collection(
    collection_name=collection_name,
    dimension=embedding_dim,
    metric_type="COSINE",
    # we don't need strong consistency for this demo 
    # Bounded is enough
    consistency_level="Strong", 
)

# Insert data
data = []
for i, text in enumerate(text_lines):
    data.append({
        "id": i,
        "vector": emb_text(embedding_model, text),
        "text": text,
    })

milvus_client.insert(collection_name=collection_name, data=data)

# Visualize embeeding for vector search
question = "How is data stored in Milvus?"
question_embedding = emb_text(embedding_model, question)

# Search for similar vectors
results = milvus_client.search(
    collection_name=collection_name,
    data=[question_embedding],
    limit=5,
    output_fields=["text"],
)

# Print the results
for result in results:
    print(result)

# Reduce embedding dimension
import pandas as pd
import numpy as np
from sklearn.manifold import TSNE

data.append({"id": len(data), "vector": question_embedding, "text": question})
embeddings = []
for gp in data:
    embeddings.append(gp["vector"])

X = np.array(embeddings, dtype=np.float32)
tsne = TSNE(random_state=0, max_iter=1000)
tsne_results = tsne.fit_transform(X)

df_tsne = pd.DataFrame(tsne_results, columns=["TSNE1", "TSNE2"])

import matplotlib.pyplot as plt
import seaborn as sns

similar_ids = [gp["id"] for gp in results[0]]

df_norm = df_tsne[:-1]

df_query = pd.DataFrame(df_tsne.iloc[-1]).T

similar_points = df_tsne[df_tsne.index.isin(similar_ids)]

fig, ax = plt.subplots(figsize=(8, 6))  # Set figsize

sns.set_style("darkgrid", {"grid.color": ".6", "grid.linestyle": ":"})

sns.scatterplot(
    data=df_tsne, x="TSNE1", y="TSNE2", color="blue", label="All knowledge", ax=ax
)

sns.scatterplot(
    data=similar_points,
    x="TSNE1",
    y="TSNE2",
    color="red",
    label="Similar knowledge",
    ax=ax,
)

sns.scatterplot(
    data=df_query, x="TSNE1", y="TSNE2", color="green", label="Query", ax=ax
)

plt.title("Scatter plot of knowledge using t-SNE")
plt.xlabel("TSNE1")
plt.ylabel("TSNE2")

plt.axis("equal")

plt.legend()

plt.savefig("tsne.png")
