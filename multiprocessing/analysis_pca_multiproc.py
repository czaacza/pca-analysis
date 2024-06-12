# analysis_pca_async.py
import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import time
import asyncio
from PCA_multiproc import PCA_SVD_multiprocessing, custom_pca_multiprocessing


def main():
    immigration_data = pd.read_csv('../datasets/oof.csv')

    # Dropping rows with all null values and resetting the index
    cleaned_data = immigration_data.dropna(how='all').reset_index(drop=True)

    # Selecting numerical columns (excluding country names)
    features = cleaned_data.columns[150:]

    # Standardizing the data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(cleaned_data[features])

    scaled_data = scaled_data[:10000]

    print('scaled_data shape:', scaled_data.shape)

    # measure PCA time
    print("Starting PCA")
    pca_start = time.time()
    principal_components, explained_variances = custom_pca_multiprocessing(scaled_data, n_components=2)
    pca_end = time.time()

    if principal_components is not None and explained_variances is not None:
        print('principal_components:', principal_components)
        print('explained_variances:', explained_variances)

        explained_variance_sum = sum(explained_variances)
        explained_variance = [var / explained_variance_sum for var in explained_variances]

        sum_of_variances = sum(explained_variance)
        for (i, value) in enumerate(explained_variance):
            percent = value / sum_of_variances * 100
            print(f'PC {i + 1} (of chosen): {percent:.2f}%')

        print('PCA parallel time:', pca_end - pca_start)

if __name__ == '__main__':
    main()
