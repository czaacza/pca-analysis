import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

class PCA_SVD:
    def __init__(self, n_components):
        self.n_components = n_components
        self.components_ = None
        self.explained_variance_ = None

    def mean(self, X):
        with ThreadPoolExecutor() as executor:
            mean_values = list(executor.map(np.mean, zip(*X)))
        return mean_values

    def center_data(self, X, mean):
        with ThreadPoolExecutor() as executor:
            centered_data = list(executor.map(self._center_row, X, [mean]*len(X)))
        return centered_data

    def _center_row(self, row, mean):
        return [x - m for x, m in zip(row, mean)]

    def transpose(self, X):
        return list(map(list, zip(*X)))

    def multiply(self, A, B):
        B_T = self.transpose(B)
        with ThreadPoolExecutor() as executor:
            result = list(executor.map(self._multiply_row, A, [B_T]*len(A)))
        return result

    def _multiply_row(self, A_row, B_T):
        return [sum(a * b for a, b in zip(A_row, B_col)) for B_col in B_T]

    def covariance_matrix(self, X):
        n_samples = len(X)
        X_transposed = self.transpose(X)
        cov_matrix = [[0] * len(X_transposed) for _ in range(len(X_transposed))]
        with ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(self._cov_element, X_transposed, i, j, n_samples)
                for i in range(len(X_transposed))
                for j in range(len(X_transposed))
            ]
            for future in as_completed(futures):
                i, j, value = future.result()
                cov_matrix[i][j] = value
        return cov_matrix

    def _cov_element(self, X_transposed, i, j, n_samples):
        value = sum(X_transposed[i][k] * X_transposed[j][k] for k in range(n_samples)) / (n_samples - 1)
        return i, j, value

    def svd(self, A):
        def normalize(v):
            norm = sum(x ** 2 for x in v) ** 0.5
            return [x / norm for x in v]

        def dot_product(v1, v2):
            return sum(x * y for x, y in zip(v1, v2))

        def matrix_vector_multiply(M, v):
            with ThreadPoolExecutor() as executor:
                result = list(executor.map(self._multiply_vector, M, [v]*len(M)))
            return result

        def power_iteration(M, num_simulations):
            b_k = [1.0] * len(M)
            for _ in range(num_simulations):
                b_k = matrix_vector_multiply(M, b_k)
                b_k = normalize(b_k)
            eigenvalue = dot_product(matrix_vector_multiply(M, b_k), b_k)
            eigenvector = b_k
            return eigenvalue, eigenvector

        eigenvalues, eigenvectors = [], []
        for _ in range(len(A)):
            eigenvalue, eigenvector = power_iteration(A, 100)
            eigenvalues.append(eigenvalue)
            eigenvectors.append(eigenvector)
            A = self.subtract(A, self.outer_product(eigenvector, eigenvector, eigenvalue))
        return eigenvalues, eigenvectors

    def _multiply_vector(self, M_row, v):
        return sum(M_row[j] * v[j] for j in range(len(v)))

    def subtract(self, A, B):
        with ThreadPoolExecutor() as executor:
            result = list(executor.map(self._subtract_row, A, B))
        return result

    def _subtract_row(self, A_row, B_row):
        return [A_row[j] - B_row[j] for j in range(len(A_row))]

    def outer_product(self, v1, v2, scalar):
        with ThreadPoolExecutor() as executor:
            result = list(executor.map(self._outer_product_element, range(len(v1)), [v1]*len(v1), [v2]*len(v1), [scalar]*len(v1)))
        return result

    def _outer_product_element(self, i, v1, v2, scalar):
        return [v1[i] * v2[j] * scalar for j in range(len(v2))]

    def fit(self, X):
        mean = self.mean(X)
        X_centered = self.center_data(X, mean)
        cov_matrix = self.covariance_matrix(X_centered)
        eigenvalues, eigenvectors = self.svd(cov_matrix)
        sorted_indices = sorted(range(len(eigenvalues)), key=lambda k: eigenvalues[k], reverse=True)

        sum_of_variances = sum(eigenvalues)
        for (i, value) in enumerate(eigenvalues):
            percent = value / sum_of_variances * 100
            print(f'PC {i + 1} (of all): {percent:.2f}%')

        self.components_ = [eigenvectors[i] for i in sorted_indices[:self.n_components]]
        self.explained_variance_ = [eigenvalues[i] for i in sorted_indices[:self.n_components]]
        return self

    def transform(self, X):
        mean = self.mean(X)
        X_centered = self.center_data(X, mean)
        return self.multiply(X_centered, self.transpose(self.components_))

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

def custom_pca(scaled_data, n_components=2):
    pca = PCA_SVD(n_components=n_components)
    principal_components = pca.fit_transform(scaled_data)
    explained_variances = pca.explained_variance_
    return principal_components, explained_variances

def list_shape(lst):
    if isinstance(lst, list):
        if not lst:
            return (0,)
        elif isinstance(lst[0], list):
            return (len(lst), len(lst[0]))
        else:
            return (len(lst),)
    else:
        return ()
