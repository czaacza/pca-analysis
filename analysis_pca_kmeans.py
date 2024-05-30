import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import time
from PCA_seq import custom_pca
from KMeans_seq import custom_kmeans

def main():
    immigration_data = pd.read_csv('./datasets/EU_Immigrants.csv')
    # print('immigration_data head', immigration_data.head())

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

    explained_variance_sum = sum(explained_variances)
    explained_variance = [var / explained_variance_sum for var in explained_variances]    

    sum_of_variances = sum(explained_variance)
    for (i, value) in enumerate(explained_variance):
        percent = value / sum_of_variances * 100
        if percent > 1:
            print(f'PC {i + 1} (of chosen): {percent:.2f}%')


    # Applying K-means clustering with the optimal number of clusters found
    optimal_k = 3  # Assuming from previous elbow method

    # measure KMeans time
    kmeans_start = time.time()
    centers, labels = custom_kmeans(principal_components, optimal_k)
    kmeans_end = time.time()
    print('KMeans sequential time:', kmeans_end - kmeans_start)

    # Convert lists to lists of lists for plotting
    centers = [list(center) for center in centers]
    principal_components = [list(pc) for pc in principal_components]

    # Plotting the clusters
    plt.figure(figsize=(10, 6))
    plt.scatter([pc[0] for pc in principal_components], [pc[1] for pc in principal_components], c=labels, cmap='viridis', s=50, alpha=0.6)
    plt.scatter([center[0] for center in centers], [center[1] for center in centers], c='red', s=200, alpha=0.75, marker='X')  # Cluster centers
    for i, txt in enumerate(cleaned_data['EU COUNTRIES']):
        plt.annotate(txt, (principal_components[i][0], principal_components[i][1]))
    plt.title('Clusters of EU Countries Based on Immigration Statistics 2022')
    plt.xlabel('PC 1 {}'.format(f'[{(explained_variance[0] * 100):.2f}%]'))
    plt.ylabel('PC 2 {}'.format(f'[{(explained_variance[1] * 100):.2f}%]'))
    plt.grid(True)
    plt.show()

if __name__ == '__main__':
    main()
