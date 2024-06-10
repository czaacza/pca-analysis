import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np
import time
from PCA_seq import custom_pca
from KMeans_seq import custom_kmeans

def main():
    full_start = time.time()
    data = pd.read_csv('./datasets/EU_Immigrants.csv')

    data.columns.values[0] = "id"
    cleaned_data = data.dropna(how='all').reset_index(drop=True)
    country_names = cleaned_data['id']
    features = cleaned_data.columns[1:]
    
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(cleaned_data[features])
    # scaled_data = scaled_data[:5000]  # Assume sufficient data
    print('scaled_data shape:', scaled_data.shape)

    pca_start = time.time()
    principal_components, explained_variances = custom_pca(scaled_data, n_components=2)
    principal_components = np.array(principal_components)  # Ensure this is an array
    pca_end = time.time()

    print("explained_variances", explained_variances)

    optimal_k = 5
    kmeans_start = time.time()
    centers, labels = custom_kmeans(principal_components.tolist(), optimal_k)  # Ensure data compatibility
    kmeans_end = time.time()
    full_end = time.time()

    print("centers", centers, "labels", labels)

    pca_time = (pca_end - pca_start).__str__().replace('.', ',')
    kmeans_time = (kmeans_end - kmeans_start).__str__().replace('.', ',')
    full_time = (full_end - full_start).__str__().replace('.', ',')
    print("PCA_time", "KMeans_time", "Full_time")
    print(pca_time, kmeans_time, full_time)

    plt.figure(figsize=(10, 6))
    plt.scatter(principal_components[:, 0], principal_components[:, 1], c=labels, cmap='viridis', s=50, alpha=0.6)
    plt.scatter([center[0] for center in centers], [center[1] for center in centers], c='red', s=200, alpha=0.75, marker='X')
    
    for i, country in enumerate(country_names):
        plt.annotate(country, (principal_components[i, 0], principal_components[i, 1]), fontsize=8, alpha=0.75)
    
    plt.title('Clusters of EU Countries Based on Immigration Statistics 2022')
    plt.xlabel(f'PC 1 [{(explained_variances[0] / sum(explained_variances) * 100):.2f}%]')
    plt.ylabel(f'PC 2 [{(explained_variances[1] / sum(explained_variances) * 100):.2f}%]')
    plt.grid(True)
    plt.show()

if __name__ == '__main__':
    main()
