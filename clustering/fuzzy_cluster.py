import numpy as np
import pickle
from sklearn.mixture import GaussianMixture

print("Loading embeddings...")

embeddings = np.load("embeddings.npy")

print("Running fuzzy clustering...")

n_clusters = 20   # adjustable parameter

gmm = GaussianMixture(
    n_components=n_clusters,
    covariance_type="tied",
    random_state=42
)

gmm.fit(embeddings)

cluster_probs = gmm.predict_proba(embeddings)

print("Cluster probability shape:", cluster_probs.shape)

print("Saving cluster distributions...")

np.save("cluster_probs.npy", cluster_probs)

with open("gmm_model.pkl", "wb") as f:
    pickle.dump(gmm, f)

print("Fuzzy clustering complete")