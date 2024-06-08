import numpy as np
import random

def initialize_centroids(data, k):
    indices = np.random.choice(data.shape[0], k, replace=False)
    return data[indices]

def euclidean_distance(a, b):
    return np.sqrt(np.sum((a - b) ** 2))

def assign_clusters(data, centroids):
    distances = np.sqrt(((data - centroids[:, np.newaxis])**2).sum(axis=2))
    return np.argmin(distances, axis=0)

def calculate_centroids(data, labels, k):
    return np.array([data[labels == i].mean(axis=0) for i in range(k)])

def kmeans(data, k, max_iterations=100):
    centroids = initialize_centroids(data, k)
    for _ in range(max_iterations):
        labels = assign_clusters(data, centroids)
        new_centroids = calculate_centroids(data, labels, k)
        if np.allclose(centroids, new_centroids):
            break
        centroids = new_centroids
    return centroids, labels

def calculate_total_wcss(data, centroids, labels):
    return sum(np.sum((data[labels == i] - centroids[i])**2) for i in range(len(centroids)))

def custom_kmeans(data, k, n_runs=10):
    best_wcss = float('inf')
    best_centroids = None
    best_labels = None

    for _ in range(n_runs):
        centroids, labels = kmeans(data, k)
        current_wcss = calculate_total_wcss(data, centroids, labels)
        if current_wcss < best_wcss:
            best_wcss = current_wcss
            best_centroids = centroids
            best_labels = labels

    return best_centroids, best_labels
