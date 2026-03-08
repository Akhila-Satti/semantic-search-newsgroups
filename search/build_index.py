import faiss
import numpy as np

print("Loading embeddings...")

embeddings = np.load("embeddings.npy")

dimension = embeddings.shape[1]

print("Vector dimension:", dimension)

print("Building FAISS index...")

index = faiss.IndexFlatL2(dimension)

index.add(embeddings)

print("Total vectors indexed:", index.ntotal)

print("Saving index...")

faiss.write_index(index, "search_index.faiss")

print("Index saved as search_index.faiss")