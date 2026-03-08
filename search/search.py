import faiss
import numpy as np
import pickle
from sentence_transformers import SentenceTransformer

print("Loading FAISS index...")
index = faiss.read_index("search_index.faiss")

print("Loading documents...")
with open("documents.pkl", "rb") as f:
    documents = pickle.load(f)

print("Loading embedding model...")
model = SentenceTransformer("all-MiniLM-L6-v2")

print("Semantic search ready!")

while True:
    query = input("\nEnter search query (or 'exit'): ")

    if query.lower() == "exit":
        break

    query_embedding = model.encode([query])

    k = 5
    distances, indices = index.search(np.array(query_embedding), k)

    print("\nTop results:\n")

    for i, idx in enumerate(indices[0]):
        print(f"Result {i+1}:")
        print(documents[idx][:500])
        print("\n------------------------\n")