import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from PCA import custom_pca
from KMeans import custom_kmeans
import time

def main():
    immigration_data = pd.read_csv('./datasets/EU_Immigrants.csv')
    print('immigration_data head', immigration_data.head())

    # Dropping rows with all null values and resetting the index
    cleaned_data = immigration_data.dropna(how='all').reset_index(drop=True)

    # Selecting numerical columns (excluding country names)
    features = cleaned_data.columns[1:]

    # Standardizing the data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(cleaned_data[features])

    # measure PCA time
    pca_start = time.time()
    principal_components, explained_variances = custom_pca(scaled_data, n_components=2)
    pca_end = time.time()
    print('PCA time:', pca_end - pca_start)

    explained_variance = explained_variances / np.sum(explained_variances)

    print(explained_variance, principal_components.shape)

    # Applying K-means clustering with the optimal number of clusters found
    optimal_k = 3  # Assuming from previous elbow method

    # measure KMeans time
    kmeans_start = time.time()
    centers, labels = custom_kmeans(principal_components, optimal_k)
    kmeans_end = time.time()
    print('KMeans time:', kmeans_end - kmeans_start)

    # Plotting the clusters
    plt.figure(figsize=(10, 6))
    plt.scatter(principal_components[:, 0], principal_components[:, 1], c=labels, cmap='viridis', s=50, alpha=0.6)
    plt.scatter(centers[:, 0], centers[:, 1], c='red', s=200, alpha=0.75, marker='X')  # Cluster centers
    for i, txt in enumerate(cleaned_data['EU COUNTRIES']):
        plt.annotate(txt, (principal_components[i, 0], principal_components[i, 1]))
    plt.title('Clusters of EU Countries Based on Immigration Statistics 2022')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.grid(True)
    plt.show()



if __name__ == '__main__':
    main()
