import faiss
import numpy as np
import pickle
from fastapi import FastAPI
from sentence_transformers import SentenceTransformer

from cache.semantic_cache import SemanticCache

app = FastAPI()

print("Loading FAISS index...")
index = faiss.read_index("search_index.faiss")

print("Loading documents...")
with open("documents.pkl", "rb") as f:
    documents = pickle.load(f)

print("Loading clustering...")
cluster_probs = np.load("cluster_probs.npy")

print("Loading embedding model...")
model = SentenceTransformer("all-MiniLM-L6-v2")

cache = SemanticCache(threshold=0.9)


@app.post("/query")
def query_api(data: dict):

    query = data["query"]

    query_embedding = model.encode(query)

    cached_query, result, score = cache.lookup(query_embedding)

    if cached_query is not None:

        return {
            "query": query,
            "cache_hit": True,
            "matched_query": cached_query,
            "similarity_score": float(score),
            "result": result
        }

    distances, indices = index.search(
        np.array([query_embedding]),
        1
    )

    doc_id = int(indices[0][0])

    result = documents[doc_id]

    dominant_cluster = int(np.argmax(cluster_probs[doc_id]))

    cache.add(query, query_embedding, result)

    return {
        "query": query,
        "cache_hit": False,
        "matched_query": None,
        "similarity_score": None,
        "result": result,
        "dominant_cluster": dominant_cluster
    }


@app.get("/cache/stats")
def cache_stats():
    return cache.stats()


@app.delete("/cache")
def clear_cache():
    cache.clear()
    return {"message": "cache cleared"}