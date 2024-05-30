import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import time
from PCA_seq import custom_pca

def main():
    immigration_data = pd.read_csv('./datasets/EU_Immigrants.csv')

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
    
    print('PCA sequential time:', pca_end - pca_start)


if __name__ == '__main__':
    main()
