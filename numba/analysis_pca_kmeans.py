import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import time
from PCA_numba import custom_pca
from Kmeans_numba import custom_kmeans

def main():
    full_start = time.time()
    data = pd.read_csv('../datasets/oof.csv')

    data.columns.values[0] = "id"

    cleaned_data = data.dropna(how='all').reset_index(drop=True)

    # features = cleaned_data.columns[100:]
    features = cleaned_data.columns[1:]

    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(cleaned_data[features])
    scaled_data = scaled_data[:5000]
    print('scaled_data shape:', scaled_data.shape)

    print("Starting PCA")
    pca_start = time.time()
    principal_components, explained_variances = custom_pca(scaled_data, n_components=2)
    pca_end = time.time()

    explained_variance_sum = sum(explained_variances)
    explained_variance = [var / explained_variance_sum for var in explained_variances]

    optimal_k = 5

    print("Starting KMeans")
    kmeans_start = time.time()
    centers, labels = custom_kmeans(principal_components, optimal_k)
    kmeans_end = time.time()

    full_end = time.time()

    pca_time = (pca_end - pca_start).__str__().replace('.', ',')
    kmeans_time = (kmeans_end - kmeans_start).__str__().replace('.', ',')
    full_time = (full_end - full_start).__str__().replace('.', ',')
    print("PCA_time", "KMeans_time", "Full_time")
    print(pca_time, kmeans_time, full_time)

    plt.figure(figsize=(10, 6))
    plt.scatter(principal_components[:, 0], principal_components[:, 1], c=labels, cmap='viridis', s=50, alpha=0.6)
    plt.scatter([center[0] for center in centers], [center[1] for center in centers], c='red', s=200, alpha=0.75, marker='X')  # Cluster centers
    if len(data['id']) == len(principal_components):
        for i, txt in enumerate(data['id']):
            plt.annotate(txt, (principal_components[i, 0], principal_components[i, 1]))
    plt.title('Drug Response Clustering Based on PCA and KMeans (Numba)')
    plt.xlabel(f'PC 1 [{(explained_variance[0] * 100):.2f}%]')
    plt.ylabel(f'PC 2 [{(explained_variance[1] * 100):.2f}%]')
    plt.grid(True)
    plt.show()

if __name__ == '__main__':
    main()
