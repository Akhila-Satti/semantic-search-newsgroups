import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


class SemanticCache:

    def __init__(self, threshold=0.9):

        self.threshold = threshold

        self.query_embeddings = []
        self.queries = []
        self.results = []

        self.hit_count = 0
        self.miss_count = 0

    def lookup(self, query_embedding):

        if len(self.query_embeddings) == 0:
            self.miss_count += 1
            return None, None, None

        similarities = cosine_similarity(
            [query_embedding],
            self.query_embeddings
        )[0]

        best_idx = np.argmax(similarities)
        best_score = similarities[best_idx]

        if best_score >= self.threshold:

            self.hit_count += 1

            return (
                self.queries[best_idx],
                self.results[best_idx],
                best_score
            )

        self.miss_count += 1

        return None, None, None

    def add(self, query, embedding, result):

        self.queries.append(query)
        self.query_embeddings.append(embedding)
        self.results.append(result)

    def stats(self):

        total = self.hit_count + self.miss_count

        hit_rate = self.hit_count / total if total > 0 else 0

        return {
            "total_entries": len(self.queries),
            "hit_count": self.hit_count,
            "miss_count": self.miss_count,
            "hit_rate": hit_rate
        }

    def clear(self):

        self.query_embeddings = []
        self.queries = []
        self.results = []

        self.hit_count = 0
        self.miss_count = 0