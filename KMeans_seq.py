import random

def initialize_centroids(data, k):
    random_indices = random.sample(range(len(data)), k)
    centroids = [data[i] for i in random_indices]
    return centroids

def euclidean_distance(a, b):
    return sum((x - y) ** 2 for x, y in zip(a, b)) ** 0.5

def assign_clusters(data, centroids):
    clusters = [[] for _ in centroids]
    for point in data:
        distances = [euclidean_distance(point, centroid) for centroid in centroids]
        min_distance_index = distances.index(min(distances))
        clusters[min_distance_index].append(point)
    return clusters

def calculate_centroids(clusters):
    centroids = []
    for cluster in clusters:
        centroid = [sum(dim) / len(cluster) for dim in zip(*cluster)]
        centroids.append(centroid)
    return centroids

def kmeans(data, k, max_iterations=100):
    centroids = initialize_centroids(data, k)
    for _ in range(max_iterations):
        clusters = assign_clusters(data, centroids)
        new_centroids = calculate_centroids(clusters)
        if new_centroids == centroids:
            break
        centroids = new_centroids
    return centroids, clusters

def calculate_total_wcss(clusters, centroids):
    total_wcss = 0
    for i, cluster in enumerate(clusters):
        total_wcss += sum(euclidean_distance(point, centroids[i]) ** 2 for point in cluster)
    return total_wcss

def custom_kmeans(data, k, n_runs=10):
    best_wcss = float('inf')
    best_centroids = None
    best_labels = []

    for _ in range(n_runs):
        centroids, clusters = kmeans(data, k)
        current_wcss = calculate_total_wcss(clusters, centroids)
        if current_wcss < best_wcss:
            best_wcss = current_wcss
            best_centroids = centroids
            best_labels = [None] * len(data)
            for idx, cluster in enumerate(clusters):
                for point in cluster:
                    best_labels[data.index(point)] = idx

    return best_centroids, best_labels
