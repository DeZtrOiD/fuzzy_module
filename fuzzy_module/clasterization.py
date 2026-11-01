
from .fuzzy_num import create_fuzzy_exp
from .fuzzy_set import ConvexFuzzySet
from typing import List, Tuple
import numpy as np
from sklearn.metrics import silhouette_score
from sklearn_extra.cluster import KMedoids
# from validclust import dunn


def generate_fuzzy_epxs(num_objects: int) -> List[ConvexFuzzySet]:
    exp_lambda = np.random.uniform(0.1, 5.0, size=num_objects)
    shift = np.random.uniform(-10.0, 10.0, size=num_objects)
    num_levels = np.random.randint(5, 51, size=num_objects)
    normalization_exp = np.random.choice([0, 1], size=num_objects)
    normalization_set = np.random.choice([0, 1], size=num_objects)
    reversed = np.random.choice([0, 1], size=num_objects)

    a1 = np.random.uniform(0.01, 1.0, size=num_objects)
    a2 = np.random.uniform(0.01, 1.0, size=num_objects)

    min_alpha = np.minimum(a1, a2)
    max_alpha = np.maximum(a1, a2)

    same_mask = (min_alpha == max_alpha)
    max_alpha[same_mask] = np.minimum(1.0, max_alpha[same_mask] + 0.05)
    min_alpha[same_mask] = np.maximum(0.01, min_alpha[same_mask] - 0.05)

    fuzzy_array: List[ConvexFuzzySet] = []
    for i in range(num_objects):
        fuzzy_array.append(create_fuzzy_exp(
            exp_lambda[i],
            shift[i],
            max_alpha[i],
            min_alpha[i],
            num_levels[i],
            normalization_exp[i],
            normalization_set[i],
            reversed[i]
        ))

    return fuzzy_array


def build_distance_matrix(fuzzy_sets: List[ConvexFuzzySet]) -> np.ndarray:
    n = len(fuzzy_sets)
    distance_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(i + 1, n):
            dist = ConvexFuzzySet.distance(fuzzy_sets[i], fuzzy_sets[j])
            distance_matrix[i, j] = dist
            distance_matrix[j, i] = dist

    return distance_matrix


def cluster_by_count(fuzzy_sets: List[ConvexFuzzySet], k: int) -> List[int]:
    distance_matrix = build_distance_matrix(fuzzy_sets)

    kmedoids = KMedoids(n_clusters=k, metric='precomputed', random_state=42)
    labels = kmedoids.fit_predict(distance_matrix)

    return labels.tolist()


def cluster_by_criterion(fuzzy_sets: List[ConvexFuzzySet], max_k: int = 10,
                         criterion: str = "silhouette") -> Tuple[List[int], int]:

    distance_matrix = build_distance_matrix(fuzzy_sets)
    n_samples = len(fuzzy_sets)

    if criterion != "silhouette":
        raise ValueError(f"Undefined criterion: {criterion}")

    best_score = -1.0
    best_labels = []
    best_k = 2

    upper_bound = min(max_k + 1, n_samples)

    for k in range(2, upper_bound):
        kmedoids = KMedoids(n_clusters=k, metric='precomputed', random_state=42)
        labels = kmedoids.fit_predict(distance_matrix)

        score = silhouette_score(distance_matrix, labels, metric='precomputed')

        if score > best_score:
            best_score = score
            best_labels = labels.tolist()
            best_k = k

    return best_labels, best_k
