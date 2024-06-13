import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import time
from mpi4py import MPI
from PCA_mpi import PCA_SVD_MPI

def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    if rank == 0:
        immigration_data = pd.read_csv('../datasets/oof.csv')

        cleaned_data = immigration_data.dropna(how='all').reset_index(drop=True)

        features = cleaned_data.columns[150:]

        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(cleaned_data[features])

        scaled_data = scaled_data[:10000]

        print('scaled_data shape:', scaled_data.shape)
    else:
        scaled_data = None

    scaled_data = comm.bcast(scaled_data, root=0)

    if rank == 0:
        print("Starting PCA")
        pca_start = time.time()

    pca = PCA_SVD_MPI(n_components=2)
    components, explained_variances = pca.custom_pca_mpi(scaled_data)

    if rank == 0:
        pca_end = time.time()

        if components is not None and explained_variances is not None:
            print('principal_components:', components)
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
