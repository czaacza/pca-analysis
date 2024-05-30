import numpy as np
import threading

def assign_cluster(data, centers, labels, start_idx, end_idx):
    """Assign a subset of data points to the nearest cluster center."""
    for i in range(start_idx, end_idx):
        distances = np.linalg.norm(data[i] - centers, axis=1)
        labels[i] = np.argmin(distances)

def update_center(data, labels, new_centers, k, lock):
    """Update the cluster centers."""
    for i in range(k):
        selected_data = data[labels == i]
        if len(selected_data) > 0:
            with lock:
                new_centers[i] = selected_data.mean(axis=0)

def custom_kmeans(data, k, max_iter=50, num_threads=4):
    np.random.seed(42)
    centers = data[np.random.choice(data.shape[0], k, replace=False)]
    labels = np.zeros(data.shape[0], dtype=int)
    thread_list = []
    lock = threading.Lock()

    for _ in range(max_iter):
        # Assign clusters in parallel
        step = data.shape[0] // num_threads
        for i in range(num_threads):
            start_idx = i * step
            end_idx = data.shape[0] if i == num_threads - 1 else (i + 1) * step
            thread = threading.Thread(target=assign_cluster, args=(data, centers, labels, start_idx, end_idx))
            thread_list.append(thread)
            thread.start()

        for thread in thread_list:
            thread.join()

        # Update centers in parallel
        new_centers = np.zeros_like(centers)
        thread_list = []
        for i in range(num_threads):
            thread = threading.Thread(target=update_center, args=(data, labels, new_centers, k, lock))
            thread_list.append(thread)
            thread.start()

        for thread in thread_list:
            thread.join()

        if np.allclose(centers, new_centers):
            break
        centers = new_centers.copy()

    return centers, labels

