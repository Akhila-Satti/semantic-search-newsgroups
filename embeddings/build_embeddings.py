import numpy as np
import pickle
from sentence_transformers import SentenceTransformer
from utils.load_dataset import load_dataset


def build_embeddings():

    documents, labels, categories = load_dataset()

    print("Loading embedding model...")

    model = SentenceTransformer("all-MiniLM-L6-v2")

    print("Generating embeddings...")

    embeddings = model.encode(
        documents,
        show_progress_bar=True,
        batch_size=64
    )

    embeddings = np.array(embeddings)

    print("Embedding shape:", embeddings.shape)

    print("Saving files...")

    np.save("embeddings.npy", embeddings)

    with open("documents.pkl", "wb") as f:
        pickle.dump(documents, f)

    with open("labels.pkl", "wb") as f:
        pickle.dump(labels, f)

    print("Saved:")
    print("embeddings.npy")
    print("documents.pkl")
    print("labels.pkl")


if __name__ == "__main__":
    build_embeddings()