import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import time
from numba.PCA_numba import custom_pca

def main():
    immigration_data = pd.read_csv('../datasets/EU_Immigrants.csv')

    cleaned_data = immigration_data.dropna(how='all').reset_index(drop=True)

    features = cleaned_data.columns[1:]

    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(cleaned_data[features])

    # scaled_data = scaled_data[:10000]

    print('scaled_data shape:', scaled_data.shape)

    print("Starting PCA")
    pca_start = time.time()
    principal_components, explained_variances = custom_pca(scaled_data, n_components=2)
    pca_end = time.time()

    print('principal_components:', principal_components)
    print('explained_variances:', explained_variances)

    explained_variance_sum = sum(explained_variances)
    explained_variance = [var / explained_variance_sum for var in explained_variances]    

    sum_of_variances = sum(explained_variance)
    for (i, value) in enumerate(explained_variance):
        percent = value / sum_of_variances * 100
        print(f'PC {i + 1} (of chosen): {percent:.2f}%')
    
    print('PCA sequential time:', pca_end - pca_start)


if __name__ == '__main__':
    main()
