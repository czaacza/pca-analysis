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
    labels = []
    for point in data:
        distances = [euclidean_distance(point, centroid) for centroid in centroids]
        min_distance_index = distances.index(min(distances))
        labels.append(min_distance_index)
    return centroids, labels

def custom_kmeans(data, k):
    return kmeans(data, k)
